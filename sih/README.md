# MINE SUBSIDENCE AI — SIH26025
### AI-Enabled Low-Cost Real-Time Mine Subsidence Monitoring, Prediction and Early Warning System for Underground Coal Mines in India

---

## 1. Project Overview

Underground coal mining poses continuous geomechanical subsidence hazards that threaten mine worker safety, surface infrastructure, and ecological stability. **SIH26025** provides an end-to-end multi-tiered monitoring and early warning solution that combines:

1. **Spaceborne InSAR Satellite Geodesy:** Multi-temporal surface deformation tracking (2020–2023, Sentinel-1 InSAR dataset in Talcher Coalfield basin, Odisha, India) across 25 ground monitoring points.
2. **Ground-Level IoT Edge Sensor Node:** Low-cost ESP32 microcontroller paired with real-time inclinometer (MPU6050 tilt), ultrasonic displacement sensor (HC-SR04), extensometer (crack width potentiometer), and strain gauge.
3. **Multi-Model AI Prediction Engine:** Ensemble combining XGBoost, Random Forest, and Attention/Transformer neural networks trained on 18 geotechnical and meteorological features.
4. **FastAPI Gateway:** Microservice layer managing real-time sensor ingestion, model inference, alert state persistence, and hardware actuation signals.
5. **Industrial Streamlit GIS Dashboard:** Real-time spatial monitoring dashboard featuring interactive Folium maps, temporal deformation curves, live telemetry polling, seismic bump analysis, and safety directives.

---

## 2. System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 DATA LAYER                                        |
|  [Real InSAR Geodesy: 1,200 Epochs]     [Seismic Hazard DB: 170 Records]          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        MACHINE LEARNING ENGINE (src/)                             |
|  - Feature Engineering (18 Features: cumulative disp, velocity, slope, rain, etc.) |
|  - XGBoost Classifier (Primary deployed model)                                    |
|  - Random Forest Classifier                                                       |
|  - Temporal Attention Transformer                                                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              FASTAPI BACKEND                                      |
|  POST /predict            -> Evaluates real-time sensor payload & updates state   |
|  GET  /predict/latest     -> Evaluates latest satellite InSAR observation        |
|  GET  /sensor/latest      -> Exposes current telemetry state to dashboard         |
|  GET  /health             -> Health & model availability probe                    |
+-----------------------------------------------------------------------------------+
         ^                                                          |
         | (Telemetry HTTP POST)                                    | (State Polling)
         |                                                          v
+------------------------------------+             +--------------------------------+
|      GROUND IOT EDGE PROTOTYPE     |             |     STREAMLIT INDUSTRIAL GIS   |
|  - ESP32 Microcontroller (Wokwi)   |             |           DASHBOARD            |
|  - MPU6050 (Tilt angle)            |             |  - Interactive Folium GIS Map  |
|  - HC-SR04 (Displacement mm)       |             |  - Multi-Station Time Series   |
|  - Potentiometers (Crack / Strain) |             |  - Live Sensor Telemetry Panel |
|  - Actuators: Green/Yellow/Red LED |             |  - Risk Distribution & Metrics |
|    + High Risk Audible Buzzer      |             |  - Seismic Hazard Layer        |
+------------------------------------+             +--------------------------------+
```

---

## 3. Data Sources

### A. Sentinel-1 InSAR Satellite Dataset (`data/features/features.csv`)
* **Location:** Talcher Coalfield, Angul District, Odisha, India (Lat: 20.919°–20.993° N, Lon: 85.171°–85.263° E).
* **Volume:** 1,200 observations across 25 permanent monitoring stations (`PT_000` to `PT_024`) spanning 48 chronological epochs (2020-01-01 to 2023-11-11).
* **18 Model Features:**
  `latitude`, `longitude`, `cumulative_displacement_mm`, `displacement_velocity_mm_month`, `displacement_acceleration`, `los_displacement_mm`, `coherence`, `elevation_m`, `slope_deg`, `aspect_deg`, `rainfall_mm`, `cumulative_rainfall_mm`, `mining_factor`, `distance_to_centre_deg`, `rolling_mean_displacement_3m`, `rolling_velocity_3m`, `deformation_trend`, `rainfall_anomaly`.
* **Target Classes:** `LOW`, `MODERATE`, `HIGH`, `CRITICAL`.

### B. Deep Seismic Bump Dataset (`data/processed/seismic_cleaned.csv`)
* **Volume:** 170 geomechanical seismic hazard records.
* **Attributes:** `energy`, `maxenergy`, `genergy`, `gpuls`, `gdenergy`, `gdpuls`, `ghazard`, `nbumps` (bump frequency count).
* **Role:** Complementary geomechanical monitoring layer for acoustic hazard awareness.

### C. Ground Edge Prototype Telemetry
* Real-time 4-channel physical sensor payload: `tilt` (degrees), `displacement` (mm), `crack_width` (mm), `strain` (µε).

---

## 4. Machine Learning Pipeline

| Model | Architecture | Best Accuracy | Macro F1 | Role |
|---|---|---|---|---|
| **XGBoost** | Gradient Boosted Decision Trees | **58.2%** | **0.516** | **Primary Deployed Classifier** |
| **Random Forest** | Bagged Decision Tree Ensemble | 56.0% | 0.500 | Comparative Baseline |
| **Attention Transformer** | 2-Layer Transformer Encoder | 38.0% | 0.298 | Multi-Epoch Temporal Sequence Evaluator |

### Scientific Grounding & Feature Mapping
* The InSAR ML models are trained on satellite interferometric geodesy and environmental terrain features.
* Ground sensor inputs (`tilt`, `crack_width`, `strain`, `displacement`) are dynamically converted to model feature representations anchored by real empirical baseline medians in `src/inference/predict.py`. Unrelated physical variables are never artificially fabricated.

---

## 5. API Endpoints Reference

Base URL: `http://127.0.0.1:8000`

### 1. `GET /`
Returns system metadata and active model.
```json
{
  "system": "Mine Subsidence AI",
  "status": "running",
  "model": "XGBoost"
}
```

### 2. `GET /health`
Liveness probe.
```json
{
  "status": "healthy",
  "model": "XGBoost"
}
```

### 3. `POST /predict`
Evaluates real-time sensor data and updates the latest sensor state.
* **Request:**
  ```json
  {
    "tilt": 2.5,
    "displacement": 12.0,
    "crack_width": 1.2,
    "strain": 0.003
  }
  ```
* **Response:**
  ```json
  {
    "risk": "LOW",
    "risk_level": "LOW",
    "native_risk": "LOW",
    "confidence": 0.987151,
    "probabilities": {
      "LOW": 0.987151,
      "MEDIUM": 0.011772,
      "HIGH": 0.001077
    },
    "action": "NORMAL",
    "action_signal": "GREEN",
    "led_signal": "GREEN",
    "message": "Ground conditions are stable."
  }
  ```

### 4. `GET /sensor/latest`
Retrieves the most recent telemetry state stored by the backend.
```json
{
  "status": "available",
  "data": {
    "tilt": 2.5,
    "displacement": 12.0,
    "crack_width": 1.2,
    "strain": 0.003,
    "timestamp": "2026-09-29T13:24:27.518406+00:00",
    "risk": "LOW",
    "confidence": 0.987151,
    "action_signal": "GREEN",
    "message": "Ground conditions are stable."
  }
}
```

### 5. `GET /predict/latest`
Runs multi-model inference on the latest available historical InSAR observation epoch.

---

## 6. Hardware & Wokwi Simulation

### ESP32 Pin Mapping:
* **Green LED (Safe / Normal):** GPIO 12
* **Yellow LED (Moderate / Warning):** GPIO 14
* **Red LED (High Risk Alert):** GPIO 27
* **Audible Buzzer:** GPIO 26
* **Potentiometer 1 (Crack Width mm):** ADC GPIO 34
* **Potentiometer 2 (Strain µε):** ADC GPIO 35
* **HC-SR04 (Displacement mm):** Trig GPIO 5, Echo GPIO 18

### Wokwi Configuration Steps:
1. Open a MicroPython ESP32 project on [Wokwi](https://wokwi.com).
2. Paste the code from `esp32/main.py`.
3. Set `SERVER_URL` to your active localtunnel HTTPS endpoint (e.g. `https://your-tunnel.loca.lt/predict`).
4. Start simulation — the node connects to `Wokwi-GUEST`, samples inputs, submits `POST /predict`, and drives LEDs/buzzer dynamically.

---

## 7. Streamlit Monitoring Dashboard

The dashboard provides an industrial-grade interface built with Streamlit, Folium, and Plotly:
* **Interactive InSAR GIS Map:** Color-coded stations (`LOW` green, `MODERATE` yellow, `HIGH` red, `CRITICAL` deep red) on Satellite/Dark basemaps with complete popup telemetry.
* **Station Inspector:** Deep drill-down for any selected monitoring station with historical sparklines.
* **Live IoT Telemetry Panel:** Real-time sensor metrics, countdown age, and integrated simulation console for manual testing.
* **Temporal Analytics:** Multi-station cumulative displacement curves, velocity dynamics, and rainfall cross-correlation.
* **Risk & Spatial Correlation Analytics:** Risk distribution charts, mining factor sensitivity scatter plots.
* **Seismic Layer:** Supplementary deep geomechanical acoustic bump tracking.

---

## 8. Installation & Quickstart

### Prerequisites
* Python 3.9+
* Recommended: Virtual environment

### Installation
```bash
cd sih
pip install -r requirements.txt
```

### Start FastAPI Backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open browser at: `http://localhost:8501`

### Run End-to-End Test Suite
```bash
python verify_system.py
```

---

## 9. Current Limitations & Future Work

* **Physical GPS Integration:** Current IoT prototype nodes operate at the mine face without dedicated GPS receivers; spatial coordinates on the GIS map correspond to verified Sentinel-1 InSAR satellite stations. Future revisions will embed GNSS RTK modules into edge hardware.
* **Combined Deep Fusion:** While InSAR and seismic data are currently analyzed in complementary layers, future work will train a unified spatio-temporal Graph Neural Network (GNN) combining satellite interferometry, subsurface seismic pulses, and continuous edge tiltmeter telemetry.