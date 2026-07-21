"""
Creates a deliberately drifted version of the data to verify the drift
detector actually catches real distributional shift, not just running
without errors.

Usage:
    python scripts/inject_test_drift.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(99)

df = pd.read_csv("data/processed/corn_yield_features.csv")
drifted = df.copy()

# Simulate a realistic drift scenario: a shift in planted_acres distribution
# (e.g. a new reporting methodology, or genuinely different farming patterns
# in a new season) plus a shift in the yield lag feature
if "planted_acres" in drifted.columns:
    drifted["planted_acres"] = drifted["planted_acres"] * 1.8 + np.random.normal(0, 5000, len(drifted))

if "yield_bu_per_acre_lag1" in drifted.columns:
    drifted["yield_bu_per_acre_lag1"] = drifted["yield_bu_per_acre_lag1"] * 0.6

if "year" in drifted.columns:
    drifted["year"] = drifted["year"] + 50  # push into unrealistic future years

if "yield_3yr_avg" in drifted.columns:
    drifted["yield_3yr_avg"] = drifted["yield_3yr_avg"] * 0.4

out_path = Path("data/processed/drifted_test_data.csv")
drifted.to_csv(out_path, index=False)
print(f"Drifted test dataset saved: {out_path}")
print(f"Original planted_acres mean: {df['planted_acres'].mean():.0f}")
print(f"Drifted planted_acres mean: {drifted['planted_acres'].mean():.0f}")
