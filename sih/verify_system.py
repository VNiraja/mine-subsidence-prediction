"""
End-to-End System Verification for SIH26025
Tests FastAPI, ML inference pipeline, state management, dataset integrity, and dashboard data.
"""

from fastapi.testclient import TestClient
import pandas as pd
from backend.main import app
from src.inference.predict import predict_risk, predict_latest

def run_tests():
    print("==================================================")
    print("SIH26025 — END-TO-END SYSTEM VERIFICATION")
    print("==================================================")

    # 1. Test InSAR Features Dataset
    print("\n--- 1. Testing InSAR Feature Dataset ---")
    df = pd.read_csv("data/features/features.csv")
    print(f"Shape: {df.shape}")
    print(f"Unique Points: {df['point_id'].nunique()}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Risk Distribution:\n{df['risk_class'].value_counts().to_dict()}")
    assert len(df) == 1200, f"Expected 1200 records, got {len(df)}"
    assert df['point_id'].nunique() == 25, f"Expected 25 points, got {df['point_id'].nunique()}"
    print(">> InSAR Dataset: PASSED")

    # 2. Test FastAPI Endpoints
    print("\n--- 2. Testing FastAPI Endpoints ---")
    client = TestClient(app)

    # Root
    r_root = client.get("/")
    assert r_root.status_code == 200, f"Root failed: {r_root.status_code}"
    print(f"GET /: {r_root.json()}")

    # Health
    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Health failed: {r_health.status_code}"
    print(f"GET /health: {r_health.json()}")

    # Sensor Latest (Initial)
    r_init_sensor = client.get("/sensor/latest")
    assert r_init_sensor.status_code == 200
    print(f"GET /sensor/latest (initial): {r_init_sensor.json()}")

    # 3. Test POST /predict with Low Risk Sensor Values
    print("\n--- 3. Testing POST /predict (LOW Risk) ---")
    low_payload = {"tilt": 0.5, "displacement": 2.0, "crack_width": 0.2, "strain": 0.0005}
    r_low = client.post("/predict", json=low_payload)
    assert r_low.status_code == 200
    low_data = r_low.json()
    print("LOW Response:", low_data)
    assert "risk" in low_data
    assert "confidence" in low_data
    assert "action" in low_data
    assert "led_signal" in low_data
    assert low_data["led_signal"] == "GREEN"

    # Verify sensor state updated
    r_sensor_after = client.get("/sensor/latest")
    assert r_sensor_after.status_code == 200
    assert r_sensor_after.json()["status"] == "available"
    assert r_sensor_after.json()["data"]["displacement"] == 2.0
    print(">> State update verified: PASSED")

    # 4. Test POST /predict with Higher Displacement Values
    print("\n--- 4. Testing POST /predict (High Displacement) ---")
    high_payload = {"tilt": 8.5, "displacement": 75.0, "crack_width": 12.0, "strain": 0.0080}
    r_high = client.post("/predict", json=high_payload)
    assert r_high.status_code == 200
    high_data = r_high.json()
    print("HIGH Response:", high_data)

    # 5. Test GET /predict/latest (Historical latest InSAR pipeline)
    print("\n--- 5. Testing GET /predict/latest ---")
    r_latest = client.get("/predict/latest")
    assert r_latest.status_code == 200
    latest_data = r_latest.json()
    print(f"GET /predict/latest Risk: {latest_data['risk']}, Confidence: {latest_data['confidence']}")
    print(f"Models evaluated: {list(latest_data.get('models', {}).keys())}")

    # 6. Test Seismic Dataset
    print("\n--- 6. Testing Seismic Dataset ---")
    df_seis = pd.read_csv("data/processed/seismic_cleaned.csv")
    print(f"Seismic records: {len(df_seis)}")
    print(f"Seismic columns: {df_seis.columns.tolist()[:8]}...")
    assert len(df_seis) > 0

    print("\n==================================================")
    print("ALL 6 TEST SUITES PASSED WITH 100% SUCCESS!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
