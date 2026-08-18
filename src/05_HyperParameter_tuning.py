"""
===============================================================================
Module 5: Hyperparameter Optimization & Cross-Validation
===============================================================================
Input:
  - Data/processed/X_train.csv
  - Data/processed/y_train.csv
  - Data/processed/X_test.csv
  - Data/processed/y_test.csv
Outputs (Outside 'src/'):
  - Serialized Tuned Model: models/best_model_tuned.joblib
===============================================================================
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

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
print("STARTING HYPERPARAMETER TUNING (RANDOM FOREST REGRESSOR)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3. Define Regularization Hyperparameter Grid
# -----------------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10, 15],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", "log2"],
}

base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# 5-Fold Cross-Validation Grid Search targeting R² scoring
grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 4. Evaluate Tuned Model on Test Split
# -----------------------------------------------------------------------------
best_rf = grid_search.best_estimator_

y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n--- TUNING RESULTS ---")
print(f"Optimal Hyperparameters: {grid_search.best_params_}")
print(f"Cross-Validation R² Score: {grid_search.best_score_:.4f}")
print(f"Train R² Score:            {train_r2:.4f}")
print(f"Test R² Score:             {test_r2:.4f}")
print(f"Test RMSE:                 {test_rmse:.4f}")
print(f"Test MAE:                  {test_mae:.4f}")

# -----------------------------------------------------------------------------
# 5. Save Serialized Tuned Model
# -----------------------------------------------------------------------------
save_path = MODELS_DIR / "best_model_tuned.joblib"
joblib.dump(best_rf, save_path)

print("\n" + "=" * 70)
print(f"MODULE 5 COMPLETE: Optimized model serialized to '{save_path}'")
print("=" * 70)