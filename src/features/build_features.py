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

def add_crop_failure_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag complete crop failure years — informative signal, not noise."""
    df = df.copy()
    df["was_crop_failure"] = (df["yield_bu_per_acre"] == 0).astype(int)
    return df

def drop_incomplete_current_year(df: pd.DataFrame, year_col: str = "year",
                                  min_completeness_ratio: float = 0.5) -> pd.DataFrame:
    """Drop the most recent year if it looks like a still-in-progress growing
    season rather than complete data -- USDA reports partial results for the
    current season before harvest finishes, and mixing that with complete
    historical years isn't a fair comparison (early estimates aren't the
    same as final yields), and would distort whichever split it lands in.

    Uses a general row-count heuristic (far fewer rows than a typical year)
    rather than a hardcoded year, so this keeps working correctly without
    manual updates as time passes.
    """
    year_counts = df[year_col].value_counts().sort_index()
    if len(year_counts) < 2:
        return df  # not enough years to compare against

    most_recent_year = year_counts.index.max()
    most_recent_count = year_counts.loc[most_recent_year]
    other_years_median = year_counts.drop(most_recent_year).median()

    if most_recent_count < min_completeness_ratio * other_years_median:
        print(f"Dropping year {most_recent_year}: only {most_recent_count} rows "
              f"vs. a typical {other_years_median:.0f} for other years -- looks "
              f"like a still-in-progress growing season, not complete data.")
        df = df[df[year_col] != most_recent_year].copy()

    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = add_lag_features(df)
    df = add_weather_interaction(df)
    df = add_soil_quality_index(df)
    df = add_regional_trend(df)
    df = add_crop_failure_flag(df)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Process demo/synthetic data instead of real production data")
    args = parser.parse_args()

    if args.demo:
        raw_path, out_path = "data/raw/corn_yield_raw_demo.csv", "data/processed/corn_yield_features_demo.csv"
    else:
        raw_path, out_path = "data/raw/corn_yield_raw.csv", "data/processed/corn_yield_features.csv"

    df = pd.read_csv(raw_path)
    df = drop_incomplete_current_year(df)
    df_features = build_all_features(df)
    df_features.to_csv(out_path, index=False)
    print(f"Features added. Shape: {df.shape} → {df_features.shape}")
    print(f"New columns: {set(df_features.columns) - set(df.columns)}")