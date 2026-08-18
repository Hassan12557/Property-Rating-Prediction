"""
===============================================================================
Module 6: Feature Importance Analysis & Interpretation
===============================================================================
Input:
  - models/best_model_tuned.joblib
  - Data/processed/X_train.csv
Outputs:
  - Prints ranked feature importances
===============================================================================
"""

from pathlib import Path
import joblib
import pandas as pd

# Path Resolution
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "src" else SCRIPT_PATH.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model_tuned.joblib"
X_TRAIN_PATH = PROJECT_ROOT / "Data" / "processed" / "X_train.csv"

# Load Artifacts
model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(X_TRAIN_PATH)

# Extract Importances
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print("=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE RANKING")
print("=" * 70)
for rank, (feature, score) in enumerate(importances.items(), 1):
    print(f"{rank:02d}. {feature:<35} : {score:.4f} ({score * 100:.2f}%)")
print("=" * 70)