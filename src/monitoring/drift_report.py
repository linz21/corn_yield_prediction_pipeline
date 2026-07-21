"""
Data drift and quality monitoring using Evidently AI.

Compares incoming/current data against the frozen reference dataset to
detect distributional shift — e.g. a new growing season with unusual
weather patterns, a new state added to the pipeline, or a data quality
issue upstream (nulls, out-of-range values).

Usage:
    python src/monitoring/drift_report.py --current data/processed/corn_yield_features.csv
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_reference_data(cfg: dict) -> pd.DataFrame:
    ref_path = Path("data/reference/reference_data.csv")
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} not found. Run scripts/create_reference_dataset.py first."
        )
    return pd.read_csv(ref_path)


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    feature_cols = [c for c in current.columns if c in reference.columns]

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference[feature_cols], current_data=current[feature_cols])

    out_dir = Path("reports/drift")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = out_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))
    log.info(f"Drift report saved: {html_path}")

    result = report.as_dict()

    drift_detected = False
    try:
        for metric in result["metrics"]:
            if metric.get("metric") == "DatasetDriftMetric":
                drift_detected = metric["result"].get("dataset_drift", False)
                break
    except (KeyError, IndexError):
        log.warning("Could not parse drift_detected flag from report — inspect HTML manually.")

    # Also extract per-column drift detail — a single important feature
    # drifting is meaningful even if it doesn't cross the aggregate threshold
    drifted_columns = []
    try:
        for metric in result["metrics"]:
            if metric.get("metric") == "DataDriftTable":
                for col_name, col_result in metric["result"]["drift_by_columns"].items():
                    if col_result.get("drift_detected"):
                        drifted_columns.append(col_name)
    except (KeyError, IndexError):
        log.warning("Could not parse per-column drift detail.")

    return {
        "timestamp": timestamp,
        "drift_detected": drift_detected,
        "drifted_columns": drifted_columns,
        "report_path": str(html_path),
        "n_reference_rows": len(reference),
        "n_current_rows": len(current),
    }


def alert_if_drifted(summary: dict):
    if summary["drift_detected"] or summary["drifted_columns"]:
        log.warning(
            f"⚠ DRIFT SIGNAL at {summary['timestamp']}. "
            f"Dataset-level drift: {summary['drift_detected']}. "
            f"Individual drifted columns: {summary['drifted_columns']}. "
            f"Report: {summary['report_path']}"
        )
    else:
        log.info(f"No drift detected at {summary['timestamp']}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--current", type=str, required=True,
                        help="Path to current data CSV to compare against reference")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    reference = load_reference_data(cfg)
    current = pd.read_csv(args.current)

    summary = run_drift_report(reference, current, cfg)
    alert_if_drifted(summary)

    print(summary)


if __name__ == "__main__":
    main()
