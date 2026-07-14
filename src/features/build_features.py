"""
Feature engineering for corn yield prediction.
Adds derived features that improve model performance.
"""

import pandas as pd
import numpy as np


def add_lag_features(df: pd.DataFrame, group_col: str = "state", 
                      target_col: str = "yield_bu_per_acre") -> pd.DataFrame:
    """Add previous year's yield as a feature — strong predictor in agriculture."""
    df = df.sort_values([group_col, "year"]).copy()
    df[f"{target_col}_lag1"] = df.groupby(group_col)[target_col].shift(1)
    df[f"{target_col}_lag1"] = df[f"{target_col}_lag1"].fillna(df[target_col].mean())
    return df


def add_weather_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features — precip and temperature rarely act independently."""
    df = df.copy()
    if "precip_inches" in df.columns and "temp_avg_f" in df.columns:
        df["precip_temp_ratio"] = df["precip_inches"] / (df["temp_avg_f"] + 1e-5)
        df["heat_stress"] = np.where(df["temp_avg_f"] > 62, df["temp_avg_f"] - 62, 0)
    return df


def add_soil_quality_index(df: pd.DataFrame) -> pd.DataFrame:
    """Composite soil quality score from organic matter + pH."""
    df = df.copy()
    if "soil_organic_matter" in df.columns and "ph_level" in df.columns:
        # Optimal corn pH is 6.0-6.8
        df["ph_deviation"] = np.abs(df["ph_level"] - 6.4)
        df["soil_quality_index"] = (
            df["soil_organic_matter"] * 0.6 - df["ph_deviation"] * 0.4
        )
    return df


def add_regional_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 3-year average yield per state — smooths out yearly noise."""
    df = df.sort_values(["state", "year"]).copy()
    df["yield_3yr_avg"] = (
        df.groupby("state")["yield_bu_per_acre"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = add_lag_features(df)
    df = add_weather_interaction(df)
    df = add_soil_quality_index(df)
    df = add_regional_trend(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/corn_yield_raw.csv")
    df_features = build_all_features(df)
    df_features.to_csv("data/processed/corn_yield_features.csv", index=False)
    print(f"Features added. Shape: {df.shape} → {df_features.shape}")
    print(f"New columns: {set(df_features.columns) - set(df.columns)}")