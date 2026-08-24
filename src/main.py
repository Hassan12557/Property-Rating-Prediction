"""
===============================================================================
FastAPI Model Server for Product Rating Prediction
===============================================================================
Execute from project root:
    uvicorn main:app --reload
===============================================================================
"""

from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. Path Anchoring & Global Pipeline Initialization
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "src" else SCRIPT_PATH.parent

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_tuned.joblib"

app = FastAPI(
    title="Product Rating Prediction API",
    description="REST API for serving Random Forest ratings predictions on e-commerce products.",
    version="1.0.0",
)

# Load pipeline models at server startup
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as err:
    raise RuntimeError(f"Failed to load model artifacts from path. Error: {err}")


# -----------------------------------------------------------------------------
# 2. Pydantic Schemas for Request & Response Validation
# -----------------------------------------------------------------------------
class ProductPredictionInput(BaseModel):
    actual_price: float = Field(..., gt=0, example=49.99, description="Raw item listing price ($)")
    category_avg_price: float = Field(..., gt=0, example=45.00, description="Average category price ($)")
    discount_pct: float = Field(default=0.0, ge=0, le=100, example=15.0, description="Discount percentage")
    popularity_score: float = Field(
        ..., ge=1.0, le=5.0, example=4.12, description="Review volume & engagement score"
    )
    specification_density: float = Field(
        ..., ge=0.01, le=1.0, example=0.45, description="Listing detail completeness ratio"
    )
    category_level_1: Literal["Electronics", "Computers&Accessories", "Home&Kitchen", "OfficeProducts", "Other"] = (
        Field(default="Electronics", description="Top-level taxonomy category")
    )
    is_branded: int = Field(default=1, ge=0, le=1, description="Binary indicator: 1 for Branded, 0 for Generic")


class PredictionOutput(BaseModel):
    predicted_rating: float
    confidence_interval: str
    status: str


# -----------------------------------------------------------------------------
# 3. API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "preprocessor_loaded": preprocessor is not None}


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_rating(payload: ProductPredictionInput):
    try:
        # Calculate engineered features under the hood
        log_price = float(np.log1p(payload.actual_price))
        price_ratio = float(1.0 - (payload.discount_pct / 100.0))
        category_price_index = (
            float(payload.actual_price / payload.category_avg_price) if payload.category_avg_price > 0 else 1.0
        )

        input_dict = {
            "log_price": log_price,
            "popularity_score": payload.popularity_score,
            "price_ratio": price_ratio,
            "category_price_index": category_price_index,
            "specification_density": payload.specification_density,
            "category_level_1": payload.category_level_1,
            "is_branded": payload.is_branded,
        }

        # Vectorize and transform
        input_df = pd.DataFrame([input_dict])
        X_prep = preprocessor.transform(input_df)

        # Generate prediction bound between 1.0 and 5.0
        prediction = float(np.clip(model.predict(X_prep)[0], 1.0, 5.0))
        rating_rounded = round(prediction, 2)

        return PredictionOutput(
            predicted_rating=rating_rounded,
            confidence_interval=f"{rating_rounded:.2f} ± 0.17",
            status="success",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")