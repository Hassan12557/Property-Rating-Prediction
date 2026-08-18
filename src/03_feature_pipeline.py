"""
===============================================================================
Module 3: Train-Test Split, Categorical Encoding & Feature Scaling
===============================================================================
Input:  Data/amazon_cleaned.csv (located at Project Root)
Outputs (Outside 'src/'):
  - Processed Matrices:
      1. Data/processed/X_train.csv
      2. Data/processed/X_test.csv
      3. Data/processed/y_train.csv
      4. Data/processed/y_test.csv
  - Serialized Artifacts:
      1. models/preprocessor.joblib
===============================================================================
"""

from pathlib import Path
from typing import List
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# 1. Directory Anchoring (Ensures Data/ & models/ sit outside 'src/')
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()

# Step up one level if script is executed inside 'src/'
if SCRIPT_PATH.parent.name == "src":
    PROJECT_ROOT = SCRIPT_PATH.parent.parent
else:
    PROJECT_ROOT = SCRIPT_PATH.parent

DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create target directories at root level if they do not exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Load Cleaned Dataset & Define Feature Subsets
# -----------------------------------------------------------------------------
INPUT_FILE = DATA_DIR / "amazon_cleaned.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find 'amazon_cleaned.csv' at '{INPUT_FILE}'. "
        "Please run Module 1 first to generate the cleaned dataset."
    )

df: pd.DataFrame = pd.read_csv(INPUT_FILE)

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
# 3. Categorical Consolidation (Handling Rare Classes)
# -----------------------------------------------------------------------------
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
# 4. Stratified Train-Test Split (80/20)
# -----------------------------------------------------------------------------
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
# 5. Feature Pipeline Construction & Fitting
# -----------------------------------------------------------------------------
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

# Fit pipeline strictly on training set to prevent data leakage
X_train_array = preprocessor.fit_transform(X_train_raw)
X_test_array = preprocessor.transform(X_test_raw)

# Reconstruct feature column names post-encoding
encoded_cat_names: List[str] = list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
)
all_feature_names: List[str] = NUMERICAL_FEATURES + encoded_cat_names

X_train_processed = pd.DataFrame(X_train_array, columns=all_feature_names)
X_test_processed = pd.DataFrame(X_test_array, columns=all_feature_names)

# -----------------------------------------------------------------------------
# 6. Save Processed Artifacts Outside 'src/'
# -----------------------------------------------------------------------------
X_train_processed.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test_processed.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_DIR /"y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR /"y_test.csv", index=False)

joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

print("\n--- PROCESSED FEATURE MATRIX COLUMNS ---")
for idx, feature in enumerate(all_feature_names, 1):
    print(f"{idx:02d}. {feature}")

print("\n" + "=" * 70)
print(f"MODULE 3 COMPLETE: Datasets saved to '{PROCESSED_DIR}'")
print(f"Pipeline serialized to '{MODELS_DIR / 'preprocessor.joblib'}'")
print("=" * 70)