"""
Exploratory Data Analysis — Corn Yield Dataset
Run:  jupyter nbconvert --to notebook --execute notebooks/01_eda.py
Or:   python notebooks/01_eda.py
"""

# %% [markdown]
# # Corn Yield EDA
# **Author:** Linlin Zhang  |  **Project:** corn-yield-prediction

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path("data/raw/corn_yield_raw.csv")
OUT_DIR   = Path("notebooks/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.dtypes)

# %% [markdown]
# ## 1. Basic overview

# %%
print("\n--- HEAD ---")
print(df.head())

print("\n--- DESCRIBE ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- UNIQUE STATES ---")
print(df["state"].value_counts())

# %% [markdown]
# ## 2. Yield distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["yield_bu_per_acre"], bins=40, color="#185FA5", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Yield (bu/acre)")
axes[0].set_ylabel("Count")
axes[0].set_title("Yield Distribution")
axes[0].axvline(df["yield_bu_per_acre"].mean(), color="red", linestyle="--", label=f"Mean: {df['yield_bu_per_acre'].mean():.1f}")
axes[0].legend()

# Boxplot by state
states = df["state"].unique()
data_by_state = [df[df["state"] == s]["yield_bu_per_acre"].values for s in states]
axes[1].boxplot(data_by_state, labels=[s[:4] for s in states], vert=True)
axes[1].set_ylabel("Yield (bu/acre)")
axes[1].set_title("Yield by State")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_yield_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: 01_yield_distribution.png")

# %% [markdown]
# ## 3. Yield trend over time

# %%
annual = df.groupby("year")["yield_bu_per_acre"].agg(["mean", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(annual["year"],
                annual["mean"] - annual["std"],
                annual["mean"] + annual["std"],
                alpha=0.2, color="#185FA5", label="±1 std")
ax.plot(annual["year"], annual["mean"], "o-", color="#185FA5", linewidth=2, label="Mean yield")

# Annotate notable dip years (drought)
for year, val in zip(annual["year"], annual["mean"]):
    if val < annual["mean"].quantile(0.15):
        ax.annotate(f"{year}", xy=(year, val), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=9, color="red")

ax.set_xlabel("Year")
ax.set_ylabel("Yield (bu/acre)")
ax.set_title("US Corn Yield Trend (2000–2023)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_yield_trend.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: 02_yield_trend.png")

# %% [markdown]
# ## 4. Feature correlations (if extra features present)

# %%
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) > 2:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    plt.colorbar(im, ax=ax)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_correlation_matrix.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 03_correlation_matrix.png")

    # Key correlations with target
    target_corr = corr["yield_bu_per_acre"].drop("yield_bu_per_acre").sort_values(ascending=False)
    print("\nCorrelations with yield_bu_per_acre:")
    print(target_corr.to_string())
else:
    print("Only numeric cols found:", numeric_cols, "— skipping correlation matrix")

# %% [markdown]
# ## 5. Summary stats for README

# %%
print("\n=== KEY STATS FOR README ===")
print(f"Total records    : {len(df):,}")
print(f"States           : {df['state'].nunique()}")
print(f"Year range       : {int(df['year'].min())}–{int(df['year'].max())}")
print(f"Mean yield       : {df['yield_bu_per_acre'].mean():.1f} bu/acre")
print(f"Std yield        : {df['yield_bu_per_acre'].std():.1f} bu/acre")
print(f"Min yield        : {df['yield_bu_per_acre'].min():.1f} bu/acre")
print(f"Max yield        : {df['yield_bu_per_acre'].max():.1f} bu/acre")
print(f"Missing values   : {df.isnull().sum().sum()}")
print("===========================")
