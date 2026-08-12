"""
===============================================================================
Module 2: Exploratory Data Analysis (EDA) & Statistical Inference
===============================================================================
Input:  Data/amazon_cleaned.csv
Outputs:
  - Saved Visualizations:
      1. reports/figures/01_target_distribution.png
      2. reports/figures/02_correlation_heatmap.png
      3. reports/figures/03_category_ratings_boxplot.png
===============================================================================
"""

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Ensure target directory exists for output plots
os.makedirs("reports/figures", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Cleaned Dataset with Explicit Type Hinting
# -----------------------------------------------------------------------------
DATA_PATH = "Data/amazon_cleaned.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "amazon_cleaned.csv"

# Explicit casting to DataFrame resolves PyCharm TextFileReader warnings
df: pd.DataFrame = pd.read_csv(DATA_PATH)

print("=" * 70)
print(f"LOADED DATASET: {df.shape[0]} rows, {df.shape[1]} columns")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2. Target Variable Analysis (rating)
# -----------------------------------------------------------------------------
rating_series: pd.Series = df["rating"]
mean_rating: float = float(rating_series.mean())
median_rating: float = float(rating_series.median())
std_rating: float = float(rating_series.std())
skew_rating: float = float(rating_series.skew())
kurt_rating: float = float(rating_series.kurtosis())

# D'Agostino-Pearson Normality Test
stat_norm, p_norm = stats.normaltest(rating_series)

print("\n--- 1. TARGET VARIABLE SUMMARY ('rating') ---")
print(f"Mean Rating:               {mean_rating:.3f}")
print(f"Median Rating:             {median_rating:.3f}")
print(f"Standard Deviation:        {std_rating:.3f}")
print(f"Skewness:                  {skew_rating:.3f}")
print(f"Kurtosis:                  {kurt_rating:.3f}")
print(f"Normality Test Statistic:  {stat_norm:.3f} (p-value = {p_norm:.4e})")

# Visualizing Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    df["rating"],
    kde=True,
    ax=axes[0],
    color="darkblue",
    bins=25,
    stat="density",
)
axes[0].axvline(
    mean_rating, color="red", linestyle="--", label=f"Mean ({mean_rating:.2f})"
)
axes[0].axvline(
    median_rating,
    color="green",
    linestyle="-.",
    label=f"Median ({median_rating:.2f})",
)
axes[0].set_title("Rating Distribution with KDE Curve", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Product Rating (1-5 Scale)")
axes[0].set_ylabel("Density")
axes[0].legend()

sns.boxplot(x=df["rating"], ax=axes[1], color="skyblue")
axes[1].set_title("Rating Boxplot (Outlier Detection)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Product Rating")

plt.tight_layout()
plt.savefig("reports/figures/01_target_distribution.png", dpi=300)
plt.close()
print("Saved: reports/figures/01_target_distribution.png")

# -----------------------------------------------------------------------------
# 3. Correlation Analysis (Pearson & Spearman)
# -----------------------------------------------------------------------------
numerical_features: List[str] = [
    "rating",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count",
    "log_price",
    "log_actual_price",
    "popularity_score",
    "price_premium",
    "price_ratio",
    "category_price_index",
    "category_discount_index",
    "specification_density",
]

corr_pearson: pd.DataFrame = df[numerical_features].corr(method="pearson")
corr_spearman: pd.DataFrame = df[numerical_features].corr(method="spearman")

print("\n--- 2. FEATURE CORRELATIONS WITH TARGET ('rating') ---")
corr_target_df: pd.DataFrame = pd.DataFrame(
    {
        "Pearson_r": corr_pearson["rating"],
        "Spearman_rho": corr_spearman["rating"],
    }
).drop("rating")

print(corr_target_df.sort_values(by="Spearman_rho", ascending=False))

# Plot Correlation Heatmap
plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
sns.heatmap(
    corr_pearson,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Pearson Correlation Matrix Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/02_correlation_heatmap.png", dpi=300)
plt.close()
print("Saved: reports/figures/02_correlation_heatmap.png")

# -----------------------------------------------------------------------------
# 4. Statistical Hypothesis Testing
# -----------------------------------------------------------------------------
print("\n--- 3. STATISTICAL HYPOTHESIS TESTING ---")

# A. Welch's t-test: Branded vs. Non-Branded Products
branded: pd.Series = df[df["is_branded"] == 1]["rating"]
unbranded: pd.Series = df[df["is_branded"] == 0]["rating"]

t_stat, p_val_t = stats.ttest_ind(branded, unbranded, equal_var=False)

print("\nA. Welch's t-test (Top 5 Brands vs. Long-Tail Brands):")
print(f"   - Top 5 Brands Mean Rating:      {branded.mean():.3f} (n={len(branded)})")
print(f"   - Long-Tail Brands Mean Rating:  {unbranded.mean():.3f} (n={len(unbranded)})")
print(f"   - t-statistic: {t_stat:.4f}, p-value: {p_val_t:.4f}")

# B. One-Way ANOVA: Category Rating Variance across Category Level 1
cat_counts: pd.Series = df["category_level_1"].value_counts()
major_categories: List[str] = cat_counts[cat_counts >= 10].index.tolist()
cat_groups = [
    df[df["category_level_1"] == cat]["rating"] for cat in major_categories
]

f_stat, p_val_anova = stats.f_oneway(*cat_groups)

print("\nB. One-Way ANOVA (Rating Variance across Major Category Level 1):")
print(f"   - Evaluated Categories: {major_categories}")
print(f"   - F-statistic: {f_stat:.4f}, p-value: {p_val_anova:.4e}")

# Plot Boxplot for Category Ratings
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df[df["category_level_1"].isin(major_categories)],
    x="category_level_1",
    y="rating",
    palette="Set2",
)
plt.title(
    "Rating Distribution by Primary Category Level 1",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Category Level 1")
plt.ylabel("Rating")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("reports/figures/03_category_ratings_boxplot.png", dpi=300)
plt.close()
print("Saved: reports/figures/03_category_ratings_boxplot.png")

# -----------------------------------------------------------------------------
# 5. Multicollinearity Diagnostic (VIF Matrix Inverse Formula)
# -----------------------------------------------------------------------------
print("\n--- 4. MULTICOLLINEARITY ANALYSIS (VIF) ---")


def calculate_vif(dataframe: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculates Variance Inflation Factor using correlation matrix inversion."""
    corr_sub: np.ndarray = dataframe[features].corr().values
    vif_values: np.ndarray = np.diag(np.linalg.inv(corr_sub))
    return pd.DataFrame({"Feature": features, "VIF": vif_values}).sort_values(
        by="VIF", ascending=False
    )


# Step 1: Candidate Feature Set
candidate_features: List[str] = [
    "log_price",
    "log_actual_price",
    "discount_percentage",
    "popularity_score",
    "price_ratio",
    "category_price_index",
    "category_discount_index",
    "specification_density",
]

vif_initial: pd.DataFrame = calculate_vif(df, candidate_features)
print("\nInitial VIF (Full Feature Candidate Set):")
print(vif_initial.to_string(index=False))

# Step 2: Pruned Feature Set (VIF < 2.0)
pruned_features: List[str] = [
    "log_price",
    "popularity_score",
    "price_ratio",
    "category_price_index",
    "specification_density",
]

vif_pruned: pd.DataFrame = calculate_vif(df, pruned_features)
print("\nPruned VIF (Recommended Non-Collinear Feature Set):")
print(vif_pruned.to_string(index=False))

print("\n" + "=" * 70)
print("MODULE 2 COMPLETE: Figures saved to 'reports/figures/'")
print("=" * 70)