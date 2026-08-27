 # 🏡 Real Estate Property Rating Prediction API & Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Deploy](https://img.shields.io/badge/Deployed-Back4app%20%7C%20Streamlit-blueviolet?style=for-the-badge)](https://propertyratingprediction-3xgfmwm0.b4a.run/docs)

An end-to-end Machine Learning solution designed to predict real estate property ratings based on market metrics, spatial feature density, and pricing structures. The system integrates a trained Machine Learning model with a high-performance **FastAPI microservice** and an intuitive **Streamlit interactive dashboard**.
## 🚀 Live Deployment Links

* 🖥️ **Interactive Web App (Streamlit Cloud):https://property-rating-prediction-vq3aowxtfkhfp5s5xf2eab.streamlit.app/
* ⚙️ **API Documentation & Swagger UI (Back4app):** [https://propertyratingprediction-3xgfmwm0.b4a.run/docs](https://propertyratingprediction-3xgfmwm0.b4a.run/docs)
* 🟢 **Backend API Health Check Endpoint:** `GET https://propertyratingprediction-3xgfmwm0.b4a.run/`
---

## 📌 Project Overview

Accurate property rating prediction enables real estate buyers, sellers, and platforms to estimate market value and customer perception based on property attributes and price dynamics. 

This project covers the full end-to-end Machine Learning workflow:
1. **Data Preprocessing & Model Training:** Automated feature scaling, categorical encoding, and model pipeline serialization using `scikit-learn` and `joblib`.
2. **REST API Microservice:** Built using `FastAPI` to serve real-time predictions with strict Pydantic payload validation and automatic Swagger documentation.
3. **Interactive Web Application:** Built using `Streamlit` to allow non-technical users to input property variables and receive instant rating predictions.
4. **Containerization & Deployment:** Dockerized and deployed across **Back4app Containers** (API Backend) and **Streamlit Community Cloud** (Frontend Dashboard).

---

## 🏗️ Technical Architecture

```text
┌────────────────────────────────┐       ┌─────────────────────────────────┐
│     Streamlit User Interface   │ ───►  │     FastAPI Microservice API    │
│  (Streamlit Community Cloud)   │ HTTP  │       (Back4app Containers)     │
└────────────────────────────────┘ POST  └─────────────────────────────────┘
                │                                         │
                ▼                                         ▼
   Interactive Input Controls                   Scikit-Learn ML Model
  (Price, Discount, Density, etc.)            (property_rating_model.joblib)



Property-Rating-Prediction/
├── models/
│   └── property_rating_model.joblib  # Serialized scikit-learn model & scaler
├── src/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application & REST endpoints
│   ├── app.py                        # Streamlit dashboard interface
│   └── pipeline.py                   # Feature preprocessing & pipeline logic
├── Dockerfile                        # Multi-stage Docker deployment build
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation




🚀 Live Demo & API Access
Live FastAPI Documentation (Swagger UI): https://propertyratingprediction-3xgfmwm0.b4a.run/docs

Health Check Endpoint: GET https://propertyratingprediction-3xgfmwm0.b4a.run/

⚡ Quick Start (Local Setup)
1. Prerequisites
Python 3.11+

Git

Docker (optional for containerized execution)

2. Clone the Repository
Bash
git clone [https://github.com/Hassan12557/Property-Rating-Prediction.git](https://github.com/Hassan12557/Property-Rating-Prediction.git)
cd Property-Rating-Prediction
3. Virtual Environment Setup
Bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS:
source venv/bin/activate
# Activate on Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
4. Run Locally
Option A: Run the FastAPI Microservice
Bash
uvicorn src.main:app --reload --port 8000
Visit http://localhost:8000/docs in your browser.

Option B: Run the Streamlit Interactive Dashboard
Bash
streamlit run src/app.py
Visit http://localhost:8501 in your browser.

🐳 Running with Docker
Build the Container
Bash
docker build -t property-rating-app .
Run the Container
Bash
docker run -p 8501:8501 property-rating-app
📡 API Reference
POST /predict
Sends property feature values to receive a predicted rating score.

Request Body (JSON)
JSON
{
  "actual_price": 49.99,
  "category_avg_price": 45.0,
  "discount_pct": 15.0,
  "popularity_score": 4.12,
  "specification_density": 0.45,
  "category_level_1": "Electronics",
  "is_branded": 1
}
Response (JSON)
JSON
{
  "status": "success",
  "predicted_rating": 4.35,
  "model_version": "1.0.0"
}
Example Usage via cURL
Bash
curl -X 'POST' \
  '[https://propertyratingprediction-3xgfmwm0.b4a.run/predict](https://propertyratingprediction-3xgfmwm0.b4a.run/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
    "actual_price": 49.99,
    "category_avg_price": 45.0,
    "discount_pct": 15.0,
    "popularity_score": 4.12,
    "specification_density": 0.45,
    "category_level_1": "Electronics",
    "is_branded": 1
  }'
🛠️ Key Technical Solutions & Challenges Overcome
Container Port Dynamics on Back4app:

Issue: Cloud container environments pass dynamic environment variables for $PORT.

Solution: Configured Docker entry points to evaluate ${PORT:-8501} dynamically across deployment environments.

Streamlit Subdirectory Execution:

Issue: Standard Docker execution expected app.py in the root folder, triggering File does not exist errors.

Solution: Updated CMD path specification to explicitly point to src/app.py.

Dependency Optimization:

Issue: Missing runtime packages (joblib) caused startup crashes during container health checks.

Solution: Audited and updated requirements.txt to strictly pin all inference runtime dependencies.

👨‍💻 Author
Hassan Raza

Role: Data Scientist & Machine Learning Engineer

Certifications: Google Data Analytics Professional Certificate | Google Advanced Data Analytics | CompTIA Data+

GitHub: @Hassan12557
 
