"""
data_loader.py
==============
Downloads NHANES 2017-2018 (.XPT) component files and merges them into a
single analysis-ready CSV keyed on SEQN.

Usage:
    python src/data_loader.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional
import pandas as pd
import requests
import yaml
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load the project YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_xpt(file_stub: str, base_url: str, raw_dir: str, retries: int = 3) -> Optional[str]:
    """
    Download a single NHANES .XPT file if not already present.
    Handles CDC's case-sensitive URLs.
    """
    os.makedirs(raw_dir, exist_ok=True)
    local_path = os.path.join(raw_dir, f"{file_stub}.XPT")

    if os.path.exists(local_path):
        logger.info("Cached file found, skipping download: %s", local_path)
        return local_path

    # Try both URL formats (CDC is case-sensitive)
    url_variants = [
        f"{base_url}{file_stub}.XPT",
        base_url.replace("/nhanes/", "/Nhanes/") + file_stub + ".XPT",
    ]

    for attempt in range(retries):
        for url in url_variants:
            try:
                logger.info("Downloading %s -> %s", url, local_path)
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()
                
                # Write file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.info("Successfully downloaded %s", file_stub)
                return local_path
                
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning("URL not found: %s", url)
                    continue
                logger.warning("HTTP error for %s: %s", url, e)
            except Exception as e:
                logger.warning("Download attempt %d failed for %s: %s", attempt + 1, file_stub, e)
            
            time.sleep(2 ** attempt)  # Exponential backoff

    logger.error("Failed to download %s after %d attempts", file_stub, retries)
    return None


def read_xpt(local_path: str) -> pd.DataFrame:
    """Read a SAS transport (.XPT) file into a DataFrame."""
    logger.info("Reading %s", local_path)
    
    # Try pandas first
    try:
        return pd.read_sas(local_path, format="xport", encoding="utf-8")
    except Exception as e:
        logger.warning("Pandas read failed: %s", e)
    
    # Try xport as fallback
    try:
        import xport
        with open(local_path, 'rb') as f:
            data = xport.v56.load(f)
            return pd.DataFrame(data)
    except ImportError:
        logger.warning("xport package not available")
    except Exception as e:
        logger.warning("xport read failed: %s", e)
    
    # Try pyreadstat
    try:
        import pyreadstat
        df, meta = pyreadstat.read_xport(local_path)
        return df
    except ImportError:
        logger.warning("pyreadstat package not available")
    except Exception as e:
        logger.warning("pyreadstat read failed: %s", e)
    
    raise ValueError(f"Could not read {local_path} with any available method")


def download_and_merge(config: dict) -> pd.DataFrame:
    """Download all configured NHANES components and inner-join on SEQN."""
    nhanes_cfg = config["nhanes"]
    base_url = nhanes_cfg["base_url"]
    raw_dir = config["paths"]["raw_dir"]
    merge_key = nhanes_cfg["merge_key"]
    file_stubs: List[str] = list(nhanes_cfg["files"].keys())

    merged_df: Optional[pd.DataFrame] = None
    component_shapes: Dict[str, tuple] = {}

    for stub in file_stubs:
        local_path = download_xpt(stub, base_url, raw_dir)
        if local_path is None:
            logger.warning("Skipping %s due to download failure", stub)
            continue
            
        df = read_xpt(local_path)
        component_shapes[stub] = df.shape

        if merged_df is None:
            merged_df = df
        else:
            try:
                merged_df = merged_df.merge(df, on=merge_key, how="inner")
                logger.info("Merged %s: %d rows", stub, len(merged_df))
            except KeyError:
                logger.warning("SEQN not found in %s, skipping", stub)
                continue

    logger.info("Component shapes: %s", component_shapes)
    logger.info("Final merged shape: %s", merged_df.shape if merged_df is not None else "None")

    if merged_df is None or merged_df.empty:
        raise ValueError("Merge produced an empty dataset. Check component availability.")

    return merged_df


def save_merged(df: pd.DataFrame, config: dict) -> str:
    """Persist the merged dataset to the processed data directory."""
    out_path = config["paths"]["merged_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved merged dataset to %s (%d rows, %d cols)", out_path, *df.shape)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and merge NHANES 2017-2018 data.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    merged = download_and_merge(config)
    save_merged(merged, config)


if __name__ == "__main__":
    main()