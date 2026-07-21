"""
Creates the reference dataset used as the baseline for drift detection.
This is a frozen snapshot of the training data distribution — future
incoming data gets compared against this to detect distributional shift.

Usage:
    python scripts/create_reference_dataset.py
"""

import pandas as pd
from pathlib import Path
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

processed_path = Path(cfg["data"]["processed_dir"]) / "corn_yield_features.csv"
df = pd.read_csv(processed_path)

feature_cols = (
    [c for c in cfg["features"]["numeric"] if c in df.columns]
    + [c for c in cfg["features"]["categorical"] if c in df.columns]
)

reference = df[feature_cols + ["yield_bu_per_acre"]].copy()

out_dir = Path("data/reference")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "reference_data.csv"
reference.to_csv(out_path, index=False)

print(f"Reference dataset saved: {out_path}  ({len(reference)} rows)")
print(f"Columns: {list(reference.columns)}")
