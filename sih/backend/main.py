from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict_latest as predict_latest_from_data
from src.inference.predict import predict_risk


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Mine Subsidence AI API",
    description="Real-time mine subsidence risk prediction API",
    version="1.0"
)

# In-memory store for the latest real sensor transmission
latest_sensor_state: Optional[Dict[str, Any]] = None


# ============================================================
# INPUT SCHEMA
# ============================================================

class SensorData(BaseModel):

    tilt: float

    displacement: float

    crack_width: float

    strain: float


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "system": "Mine Subsidence AI",
        "status": "running",
        "model": "XGBoost"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

# POST /predict accepts a JSON object containing sensor measurements,
# validates them with SensorData, passes them to predict_risk(),
# updates the latest_sensor_state for the dashboard, and returns the prediction.
@app.post("/predict")
def predict(data: SensorData):
    global latest_sensor_state

    input_data = data.model_dump()

    result = predict_risk(
        input_data
    )

    latest_sensor_state = {
        "tilt": data.tilt,
        "displacement": data.displacement,
        "crack_width": data.crack_width,
        "strain": data.strain,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk": result.get("risk"),
        "risk_level": result.get("risk_level"),
        "native_risk": result.get("native_risk"),
        "confidence": result.get("confidence"),
        "probabilities": result.get("probabilities"),
        "action": result.get("action"),
        "action_signal": result.get("action_signal"),
        "led_signal": result.get("led_signal"),
        "message": result.get("message"),
    }

    return result


# GET /predict/latest -> src/inference/predict.py -> raw InSAR CSV -> model
# probabilities/risk response -> esp32/main.py -> Wokwi LEDs and buzzer.
@app.get("/predict/latest")
def predict_latest():
    return predict_latest_from_data()


# ============================================================
# SENSOR STATE ENDPOINT (FOR DASHBOARD)
# ============================================================

@app.get("/sensor/latest")
def get_latest_sensor():
    if latest_sensor_state is None:
        return {
            "status": "no_data",
            "data": None
        }
    return {
        "status": "available",
        "data": latest_sensor_state
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "XGBoost"
    }