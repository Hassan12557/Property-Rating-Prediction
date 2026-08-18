"""
===============================================================================
Module 4: Multi-Model Training, Cross-Evaluation & Model Serialization
===============================================================================
Input:
  - Data/processed/X_train.csv
  - Data/processed/X_test.csv
  - Data/processed/y_train.csv
  - Data/processed/y_test.csv
Outputs (Outside 'src/'):
  - Serialized Best Model: models/best_model.joblib
===============================================================================
"""

from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. Directory Anchoring
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()

if SCRIPT_PATH.parent.name == "src":
    PROJECT_ROOT = SCRIPT_PATH.parent.parent
else:
    PROJECT_ROOT = SCRIPT_PATH.parent

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# -----------------------------------------------------------------------------
# 2. Load Processed Datasets
# -----------------------------------------------------------------------------
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()

print("=" * 70)
print(f"LOADED DATASETS: Train Shape={X_train.shape}, Test Shape={X_test.shape}")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3. Model Zoo Initialization
# -----------------------------------------------------------------------------
models: Dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

results = []
best_score = -float("inf")
best_model_name = ""
best_model_obj = None

# -----------------------------------------------------------------------------
# 4. Train & Evaluate Candidate Models
# -----------------------------------------------------------------------------
for name, model in models.items():
    # Fit model on training split
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Train Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)

    # Test Metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    results.append({
        "Model": name,
        "Train RMSE": round(train_rmse, 4),
        "Test RMSE": round(test_rmse, 4),
        "Test MAE": round(test_mae, 4),
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
    })

    # Track best performing model on test R²
    if test_r2 > best_score:
        best_score = test_r2
        best_model_name = name
        best_model_obj = model

# -----------------------------------------------------------------------------
# 5. Output Comparison Matrix & Save Best Model
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values(by="Test R²", ascending=False)

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(results_df.to_string(index=False))

# Save top performing model artifact
save_path = MODELS_DIR / "best_model.joblib"
joblib.dump(best_model_obj, save_path)

print("\n" + "=" * 70)
print(f"BEST PERFORMING MODEL: {best_model_name} (Test R² = {best_score:.4f})")
print(f"Serialized top model artifact to: '{save_path}'")
print("=" * 70)