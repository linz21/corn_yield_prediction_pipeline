"""
USDA NASS Quickstats data ingestion for corn yield prediction.

Usage:
    python src/data/ingest.py --api-key YOUR_KEY --states IL IA IN OH MN
    python src/data/ingest.py --demo   # downloads a small sample without an API key

USDA NASS API key: https://quickstats.nass.usda.gov/api  (free, instant)
"""

import os
import argparse
import requests
import pandas as pd
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Top corn-producing states (covers ~80% of US production)
# ---------------------------------------------------------------------------
TOP_CORN_STATES = [
    "ILLINOIS", "IOWA", "INDIANA", "OHIO", "MINNESOTA",
    "NEBRASKA", "KANSAS", "MISSOURI", "SOUTH DAKOTA", "WISCONSIN",
]

USDA_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"


def fetch_usda(api_key: str, state: str, year_start: int = 2000, year_end: int = 2023) -> pd.DataFrame:
    """Fetch corn yield data from USDA NASS Quickstats API for one state."""
    params = {
        "key": api_key,
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "state_name": state,
        "freq_desc": "ANNUAL",
        "year__GE": year_start,
        "year__LE": year_end,
        "format": "JSON",
    }
    log.info(f"Fetching {state} {year_start}-{year_end} ...")
    resp = requests.get(USDA_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or len(data["data"]) == 0:
        log.warning(f"  No data returned for {state}")
        return pd.DataFrame()
    df = pd.DataFrame(data["data"])
    log.info(f"  {len(df)} rows fetched for {state}")
    return df


def clean_usda(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename relevant columns; cast types."""
    cols = {
        "year": "year",
        "state_name": "state",
        "commodity_desc": "commodity",
        "county_name": "county",
        "Value": "yield_bu_per_acre",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Cast yield to numeric — USDA sometimes returns " (D)" for suppressed values
    df["yield_bu_per_acre"] = pd.to_numeric(df["yield_bu_per_acre"].str.replace(",", ""), errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["yield_bu_per_acre", "year"])
    df["state"] = df["state"].str.title()
    return df


def make_demo_dataset(out_path: Path) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset when no API key is available.
    Useful for setting up the pipeline before getting a USDA key.
    """
    import numpy as np
    rng = np.random.default_rng(42)

    states = ["Illinois", "Iowa", "Indiana", "Ohio", "Minnesota",
              "Nebraska", "Kansas", "Missouri", "South Dakota", "Wisconsin"]
    years = list(range(2000, 2024))

    rows = []
    # State-level base yields (bu/acre) — realistic 2023 figures
    base_yield = {
        "Illinois": 195, "Iowa": 202, "Indiana": 188,
        "Ohio": 183, "Minnesota": 197, "Nebraska": 180,
        "Kansas": 160, "Missouri": 165, "South Dakota": 145, "Wisconsin": 170,
    }

    for state in states:
        for year in years:
            # Trend: ~1.5 bu/acre/year improvement + noise
            trend = (year - 2000) * 1.5
            weather_shock = rng.normal(0, 8)   # good/bad year
            base = base_yield[state] - (2023 - year) * 1.5 + trend + weather_shock

            rows.append({
                "year": year,
                "state": state,
                "commodity": "Corn",
                "county": "State Total",
                "yield_bu_per_acre": round(max(base, 80), 1),
                "planted_acres": int(rng.integers(800_000, 13_000_000)),
                "precip_inches": round(rng.uniform(12, 45), 1),
                "temp_avg_f": round(rng.uniform(45, 68), 1),
                "soil_organic_matter": round(rng.uniform(1.5, 5.5), 2),
                "ph_level": round(rng.uniform(5.8, 7.2), 1),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Demo dataset saved: {out_path}  ({len(df)} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest USDA NASS corn yield data")
    parser.add_argument("--api-key", type=str, default=os.getenv("USDA_API_KEY"),
                        help="USDA NASS API key (or set USDA_API_KEY env var)")
    parser.add_argument("--states", nargs="+", default=TOP_CORN_STATES,
                        help="State names to fetch (uppercase)")
    parser.add_argument("--year-start", type=int, default=2000)
    parser.add_argument("--year-end", type=int, default=2023)
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic demo data (no API key needed)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "corn_yield_raw.csv"

    if args.demo:
        log.info("Running in DEMO mode — generating synthetic data ...")
        make_demo_dataset(out_path)
        return

    if not args.api_key:
        log.error(
            "No API key found. Either:\n"
            "  1. Get a free key at https://quickstats.nass.usda.gov/api and pass --api-key KEY\n"
            "  2. Set USDA_API_KEY environment variable\n"
            "  3. Run with --demo to generate synthetic data immediately"
        )
        raise SystemExit(1)

    all_dfs = []
    for state in args.states:
        df = fetch_usda(args.api_key, state, args.year_start, args.year_end)
        if not df.empty:
            all_dfs.append(clean_usda(df))

    if not all_dfs:
        log.error("No data fetched. Check API key and state names.")
        raise SystemExit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_path, index=False)
    log.info(f"\nSaved {len(combined)} rows → {out_path}")
    log.info(f"States: {combined['state'].nunique()}  |  Years: {combined['year'].min()}–{combined['year'].max()}")


if __name__ == "__main__":
    main()
