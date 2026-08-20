"""
===============================================================================
Streamlit Interactive Product Rating Predictor
===============================================================================
Execute from project root:
    streamlit run app.py
===============================================================================
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Product Rating Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Artifact Loader
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "src" else SCRIPT_PATH.parent

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_tuned.joblib"


@st.cache_resource
def load_pipeline():
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        st.error("Model artifacts not found. Please verify the `models/` folder.")
        st.stop()
    prep = joblib.load(PREPROCESSOR_PATH)
    mdl = joblib.load(MODEL_PATH)
    return prep, mdl


preprocessor, model = load_pipeline()

# -----------------------------------------------------------------------------
# 3. Sidebar: User Input Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Product Parameters")

# Product Metadata
category = st.sidebar.selectbox(
    "Category Level 1",
    ["Electronics", "Computers&Accessories", "Home&Kitchen", "OfficeProducts", "Other"],
    index=0,
)

is_branded_selection = st.sidebar.radio("Brand Type", ["Branded", "Generic"], index=0)
is_branded = 1 if is_branded_selection == "Branded" else 0

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Pricing Dynamics")

actual_price = st.sidebar.number_input("Actual Price ($)", min_value=1.0, max_value=5000.0, value=49.99, step=5.0)
category_avg_price = st.sidebar.number_input(
    "Category Avg Price ($)", min_value=1.0, max_value=5000.0, value=45.00, step=5.0
)
discount_pct = st.sidebar.slider("Discount Percentage (%)", min_value=0, max_value=90, value=15)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Metadata & Engagement")

popularity_score = st.sidebar.slider(
    "Popularity Score (Review Engagement)", min_value=1.0, max_value=5.0, value=4.10, step=0.05
)

specification_density = st.sidebar.slider(
    "Specification Density (Listing Depth)", min_value=0.05, max_value=1.00, value=0.45, step=0.05
)

# Feature Transformations (Calculated Under the Hood)
log_price = np.log1p(actual_price)
price_ratio = 1.0 - (discount_pct / 100.0)
category_price_index = actual_price / category_avg_price if category_avg_price > 0 else 1.0

# -----------------------------------------------------------------------------
# 4. Main Layout & Interactive Dashboard
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⭐ E-Commerce Product Rating Predictor</div>', unsafe_allow_html=True)
st.caption("Machine Learning inference platform powered by Random Forest Regression.")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("### 📥 Derived Feature Profile")

    # Display engineered features in clean metric layout
    m1, m2 = st.columns(2)
    m1.metric("Log Price", f"{log_price:.2f}")
    m2.metric("Price Ratio", f"{price_ratio:.2f}")

    m3, m4 = st.columns(2)
    m3.metric("Category Price Index", f"{category_price_index:.2f}")
    m4.metric("Spec Density", f"{specification_density:.2f}")

    predict_btn = st.button("🚀 Calculate Rating Prediction")

# Predict & Visualize
if predict_btn or "predicted_rating" not in st.session_state:
    input_dict = {
        "log_price": log_price,
        "popularity_score": popularity_score,
        "price_ratio": price_ratio,
        "category_price_index": category_price_index,
        "specification_density": specification_density,
        "category_level_1": category,
        "is_branded": is_branded,
    }

    input_df = pd.DataFrame([input_dict])
    X_prep = preprocessor.transform(input_df)
    st.session_state.predicted_rating = float(np.clip(model.predict(X_prep)[0], 1.0, 5.0))

with col_right:
    st.markdown("### 🎯 Predicted Customer Rating")
    pred_val = st.session_state.predicted_rating

    # Plotly Radial Gauge Chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Stars", "font": {"size": 18, "color": "#94A3B8"}},
            number={"suffix": " / 5.0", "font": {"size": 36, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [1, 5], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#06B6D4"},
                "bgcolor": "#0F172A",
                "borderwidth": 2,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [1, 3.0], "color": "#EF4444"},
                    {"range": [3.0, 4.0], "color": "#F59E0B"},
                    {"range": [4.0, 5.0], "color": "#10B981"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"💡 **Expected Model Margin:** ~{pred_val:.2f} ± 0.17 Rating Points (MAE)")

# -----------------------------------------------------------------------------
# 5. Feature Comparison Radar Chart
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🕸️ Feature Intensity Profile")

radar_categories = ["Popularity", "Price Ratio", "Category Index", "Spec Density"]
raw_values = [popularity_score / 5.0, price_ratio, min(category_price_index / 2.0, 1.0), specification_density]

fig_radar = go.Figure()
fig_radar.add_trace(
    go.Scatterpolar(
        r=raw_values,
        theta=radar_categories,
        fill="toself",
        name="Current Input",
        line_color="#4F46E5",
    )
)
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    height=320,
    margin=dict(l=40, r=40, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_radar, use_container_width=True)