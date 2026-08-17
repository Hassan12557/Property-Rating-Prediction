"""
===============================================================================
Module 3: Train-Test Split, Categorical Encoding & Feature Scaling
===============================================================================
Input:  Data/amazon_cleaned.csv
Outputs:
  - Processed Matrices:
      1. Data/processed/X_train.csv
      2. Data/processed/X_test.csv
      3. Data/processed/y_train.csv
      4. Data/processed/y_test.csv
  - Serialized Artifacts:
      1. models/preprocessor.joblib
===============================================================================
"""

import os
from typing import List, Tuple
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure output directories exist
os.makedirs("Data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Dataset & Define Feature Subsets
# -----------------------------------------------------------------------------
DATA_PATH = "Data/amazon_cleaned.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "amazon_cleaned.csv"

df: pd.DataFrame = pd.read_csv(DATA_PATH)

TARGET: str = "rating"

# Selected numerical features after VIF multicollinearity pruning (VIF < 2.0)
NUMERICAL_FEATURES: List[str] = [
    "log_price",
    "popularity_score",
    "price_ratio",
    "category_price_index",
    "specification_density",
]

# Categorical features for one-hot encoding
CATEGORICAL_FEATURES: List[str] = ["category_level_1", "is_branded"]

# -----------------------------------------------------------------------------
# 2. Categorical Consolidation (Handling Rare Classes)
# -----------------------------------------------------------------------------
# Consolidate rare categories (<10 observations) into 'Other' to prevent
# single-instance split errors and zero-variance columns in train/test splits
cat_counts: pd.Series = df["category_level_1"].value_counts()
rare_categories: pd.Index = cat_counts[cat_counts < 10].index
df["category_level_1"] = df["category_level_1"].replace(rare_categories, "Other")

X: pd.DataFrame = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y: pd.Series = df[TARGET]

print("=" * 70)
print(f"INPUT DATASET SHAPE: {X.shape[0]} rows, {X.shape[1]} raw feature columns")
print(f"CATEGORY DISTRIBUTION:\n{df['category_level_1'].value_counts()}")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3. Stratified Train-Test Split (80/20)
# -----------------------------------------------------------------------------
# Stratifying on category_level_1 preserves category balance across splits
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["category_level_1"],
)

print(f"\nTrain Set Size: {X_train_raw.shape[0]} samples")
print(f"Test Set Size:  {X_test_raw.shape[0]} samples")

# -----------------------------------------------------------------------------
# 4. Feature Pipeline Construction & Fitting
# -----------------------------------------------------------------------------
# Constructing ColumnTransformer to scale numbers and encode categories
preprocessor: ColumnTransformer = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)

# Fit pipeline strictly on training data to avoid data leakage
X_train_array = preprocessor.fit_transform(X_train_raw)
X_test_array = preprocessor.transform(X_test_raw)

# Extract post-encoding feature names
encoded_cat_names: List[str] = list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
)
all_feature_names: List[str] = NUMERICAL_FEATURES + encoded_cat_names

# Convert transformed numpy arrays back to DataFrames for inspection & saving
X_train_processed = pd.DataFrame(X_train_array, columns=all_feature_names)
X_test_processed = pd.DataFrame(X_test_array, columns=all_feature_names)

# -----------------------------------------------------------------------------
# 5. Save Processed Artifacts
# -----------------------------------------------------------------------------
X_train_processed.to_csv("Data/processed/X_train.csv", index=False)
X_test_processed.to_csv("Data/processed/X_test.csv", index=False)
y_train.to_csv("Data/processed/y_train.csv", index=False)
y_test.to_csv("Data/processed/y_test.csv", index=False)

# Save fitted ColumnTransformer object for downstream inference/serving
joblib.dump(preprocessor, "models/preprocessor.joblib")

print("\n--- PROCESSED FEATURE MATRIX COLUMNS ---")
for idx, feature in enumerate(all_feature_names, 1):
    print(f"{idx:02d}. {feature}")

print("\n" + "=" * 70)
print("MODULE 3 COMPLETE: Datasets saved to 'Data/processed/'")
print("Pipeline serialized to 'models/preprocessor.joblib'")
print("=" * 70)