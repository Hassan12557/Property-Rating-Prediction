"""
===============================================================================
Module 7: End-to-End Inference Pipeline
===============================================================================
Inputs:
  - models/preprocessor.joblib
  - models/best_model_tuned.joblib
Output:
  - Real-time rating predictions for new raw product inputs
===============================================================================
"""

from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd

# Directory Anchoring
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "src" else SCRIPT_PATH.parent

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_tuned.joblib"

# Load Pipeline Artifacts
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)


def predict_product_rating(sample_input: Dict[str, Any]) -> float:
    """Transforms raw product payload and returns predicted rating."""
    input_df = pd.DataFrame([sample_input])

    # Transform raw features using fitted ColumnTransformer
    X_processed = preprocessor.transform(input_df)

    # Generate prediction
    prediction = model.predict(X_processed)[0]
    return float(round(prediction, 2))


if __name__ == "__main__":
    # Test Payload matching raw engineered feature requirements
    sample_product = {
        "log_price": 3.25,
        "popularity_score": 4.12,
        "price_ratio": 0.85,
        "category_price_index": 1.02,
        "specification_density": 0.45,
        "category_level_1": "Electronics",
        "is_branded": 1,
    }

    predicted_rating = predict_product_rating(sample_product)

    print("=" * 70)
    print(f"SAMPLE INPUT: {sample_product}")
    print(f"PREDICTED RATING: {predicted_rating} / 5.0 stars")
    print("=" * 70)