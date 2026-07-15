"""
Baseline XGBoost model with MLflow experiment tracking.

Usage:
    python src/models/train.py
    python src/models/train.py --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bayesian-style prediction intervals via bootstrap
# This is Linlin's key differentiator — calibrated uncertainty estimates
# ---------------------------------------------------------------------------


def bootstrap_prediction_intervals(
    pipeline, X_train, y_train, X_test,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate PREDICTION intervals (not just confidence intervals) using bootstrap
    resampling + residual noise injection.

    A prediction interval must account for TWO sources of uncertainty:
      1. Model uncertainty — captured by retraining on bootstrap resamples
      2. Irreducible noise — captured by adding residual variance from held-out data

    Returns:
        point_pred   : point predictions from the full model
        lower_bound  : lower bound of the CI
        upper_bound  : upper bound of the CI
    """
    point_pred = pipeline.predict(X_test)
    boot_preds = np.zeros((n_bootstrap, len(X_test)))

    # First, compute residuals from the full model on a held-out slice of training data
    # to estimate the irreducible noise level
    from sklearn.model_selection import train_test_split as tts
    X_resid_train, X_resid_holdout, y_resid_train, y_resid_holdout = tts(
        X_train, y_train, test_size=0.2, random_state=99
    )
    resid_model = Pipeline(steps=[
        ("pre", pipeline.named_steps["pre"]),
        ("model", XGBRegressor(n_estimators=200, max_depth=6, random_state=99)),
    ])
    resid_model.fit(X_resid_train, y_resid_train)
    residuals = y_resid_holdout.values - resid_model.predict(X_resid_holdout)
    residual_std = np.std(residuals)

    rng = np.random.default_rng(42)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(X_train), len(X_train))
        X_b = X_train.iloc[idx]
        y_b = y_train.iloc[idx]
        boot_model = Pipeline(steps=[
            ("pre", pipeline.named_steps["pre"]),
            ("model", XGBRegressor(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=i
            )),
        ])
        boot_model.fit(X_b, y_b)
        base_preds = boot_model.predict(X_test)
        # Add irreducible noise sampled from the residual distribution —
        # this is what converts a confidence interval into a prediction interval
        noise = rng.normal(0, residual_std, size=len(X_test))
        boot_preds[i] = base_preds + noise

    alpha = 1 - confidence
    lower = np.percentile(boot_preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)
    return point_pred, lower, upper


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data"]["processed_dir"]) / cfg["data"]["processed_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}.\n"
            "Run first:  python src/features/build_features.py"
        )
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows from {path}")
    return df


def build_pipeline(cfg: dict) -> Pipeline:
    numeric_features  = cfg["features"]["numeric"]
    cat_features      = cfg["features"]["categorical"]
    model_params      = cfg["model"]["params"]

    # Filter to only columns that exist in the data (some may not be in demo set)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
        ],
        remainder="drop",
    )

    model = XGBRegressor(**model_params)

    return Pipeline(steps=[
        ("pre", preprocessor),
        ("model", model),
    ])


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # MAPE is undefined when y_true is 0 — exclude those rows for this metric only
    nonzero_mask = np.abs(y_true) > 1e-6
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    return {
        "rmse"    : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae"     : float(mean_absolute_error(y_true, y_pred)),
        "r2"      : float(r2_score(y_true, y_pred)),
        "mape"    : mape,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap CI (faster, for quick experiments)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = load_data(cfg)

    target   = cfg["data"]["target_col"]
    seed     = cfg["data"]["random_seed"]

    # ── Feature / target split ──────────────────────────────────────────────
    feature_cols = (
        [c for c in cfg["features"]["numeric"] if c in df.columns]
        + [c for c in cfg["features"]["categorical"] if c in df.columns]
    )
    X = df[feature_cols]
    y = df[target]

    # ── Train / val / test split ────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=cfg["data"]["val_size"] / (1 - cfg["data"]["test_size"]),
        random_state=seed,
    )
    log.info(f"Split — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    # ── MLflow tracking ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    run_name = args.run_name or f"xgboost-baseline"

    with mlflow.start_run(run_name=run_name):
        # Log config
        mlflow.log_params(cfg["model"]["params"])
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Build and train
        pipeline = build_pipeline(cfg)
        pipeline.fit(X_train, y_train)

        # Evaluate on val and test
        val_metrics  = evaluate(y_val, pipeline.predict(X_val))
        test_metrics = evaluate(y_test, pipeline.predict(X_test))

        # Log metrics
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        log.info(f"Val  RMSE={val_metrics['rmse']:.2f}  R²={val_metrics['r2']:.3f}")
        log.info(f"Test RMSE={test_metrics['rmse']:.2f}  R²={test_metrics['r2']:.3f}")

        # ── Bayesian bootstrap confidence intervals ──────────────────────────
        if not args.no_bootstrap:
            log.info("Computing bootstrap prediction intervals ...")
            n_boot = cfg["bayesian_ci"]["n_bootstrap"]
            conf   = cfg["bayesian_ci"]["confidence_level"]
            preds, lower, upper = bootstrap_prediction_intervals(
                pipeline, X_train, y_train, X_test, n_boot, conf
            )
            ci_width = float(np.mean(upper - lower))
            coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))
            mlflow.log_metric("ci_width_mean", ci_width)
            mlflow.log_metric("ci_coverage", coverage)
            log.info(f"Bootstrap CI — mean width: {ci_width:.2f} bu/acre | coverage: {coverage:.1%}")

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")
        log.info(f"Model logged to MLflow: {model_uri}")

    log.info("Run complete. Start the MLflow UI with:  mlflow ui")


if __name__ == "__main__":
    main()
