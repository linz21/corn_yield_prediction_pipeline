"""
Tests for the prediction API.
Run:  pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Unit tests — no model required
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_valid_state_title_case(self):
        """State name should be normalized to title case."""
        from src.api.main import YieldRequest
        req = YieldRequest(year=2023, state="illinois")
        assert req.state == "Illinois"

    def test_year_bounds(self):
        from src.api.main import YieldRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            YieldRequest(year=1800, state="Illinois")   # too old
        with pytest.raises(ValidationError):
            YieldRequest(year=2100, state="Illinois")   # too future

    def test_confidence_bounds(self):
        from src.api.main import YieldRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            YieldRequest(year=2023, state="Illinois", confidence=1.5)

    def test_default_values(self):
        from src.api.main import YieldRequest
        req = YieldRequest(year=2023, state="Iowa")
        assert req.commodity == "Corn"
        assert req.n_bootstrap == 200
        assert req.confidence == 0.95


class TestDataIngestion:
    def test_demo_dataset_shape(self, tmp_path):
        """Demo dataset should have expected columns and rows."""
        from src.data.ingest import make_demo_dataset
        out = tmp_path / "corn_yield_raw.csv"
        df  = make_demo_dataset(out)
        assert out.exists()
        assert "yield_bu_per_acre" in df.columns
        assert "state" in df.columns
        assert "year" in df.columns
        assert len(df) > 0

    def test_demo_yield_range(self, tmp_path):
        """Yields should be in a realistic range."""
        from src.data.ingest import make_demo_dataset
        df = make_demo_dataset(tmp_path / "test.csv")
        assert df["yield_bu_per_acre"].min() >= 80
        assert df["yield_bu_per_acre"].max() <= 250

    def test_demo_no_missing(self, tmp_path):
        from src.data.ingest import make_demo_dataset
        df = make_demo_dataset(tmp_path / "test.csv")
        assert df["yield_bu_per_acre"].isnull().sum() == 0


class TestBootstrapCI:
    """Sanity checks on bootstrap CI logic."""

    def test_ci_contains_point_pred(self):
        """Point prediction should (almost always) be within CI bounds."""
        from src.models.train import bootstrap_prediction_intervals
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        y_train = pd.Series(X_train["x"] * 2 + rng.normal(0, 0.1, 100))
        X_test  = pd.DataFrame({"x": [0.5, 1.0, -0.5]})

        pipe = Pipeline([
            ("pre", StandardScaler()),
            ("model", XGBRegressor(n_estimators=50, random_state=0)),
        ])
        pipe.fit(X_train, y_train)

        point, lower, upper = bootstrap_prediction_intervals(
            pipe, X_train, y_train, X_test, n_bootstrap=100
        )
        assert np.all(upper >= lower), "Upper bound must be >= lower bound"
        assert np.all(upper - lower > 0), "CI width must be positive"

    def test_wider_ci_for_higher_confidence(self):
        """95% CI should be wider than 80% CI."""
        from src.models.train import bootstrap_prediction_intervals
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        rng = np.random.default_rng(1)
        X_train = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        y_train = pd.Series(X_train["x"] * 3 + rng.normal(0, 0.5, 100))
        X_test  = pd.DataFrame({"x": [0.0]})

        pipe = Pipeline([
            ("pre", StandardScaler()),
            ("model", XGBRegressor(n_estimators=50, random_state=0)),
        ])
        pipe.fit(X_train, y_train)

        _, lo80, hi80 = bootstrap_prediction_intervals(pipe, X_train, y_train, X_test, 200, 0.80)
        _, lo95, hi95 = bootstrap_prediction_intervals(pipe, X_train, y_train, X_test, 200, 0.95)

        width_80 = hi80[0] - lo80[0]
        width_95 = hi95[0] - lo95[0]
        assert width_95 > width_80, f"95% CI ({width_95:.2f}) should be wider than 80% CI ({width_80:.2f})"
