"""
==============================================================================
MINE SUBSIDENCE AI — SIH26025
AI-Enabled Low-Cost Real-Time Mine Subsidence Monitoring and Early Warning System
Streamlit Industrial Monitoring Dashboard
==============================================================================
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Dict, Any, Optional

import folium
from folium.plugins import Fullscreen, MeasureControl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

# ==============================================================================
# 1. APPLICATION CONFIGURATION & INDUSTRIAL THEME
# ==============================================================================

st.set_page_config(
    page_title="Mine Subsidence AI | SIH26025",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial Monitoring Grade UI
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, .stCode, .metric-value, .status-badge {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Industrial Header Card */
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0px;
        font-weight: 400;
    }
    .header-tag {
        display: inline-block;
        background: #1e3a8a;
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        letter-spacing: 0.05em;
        border: 1px solid #3b82f6;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #3b82f6;
        transform: translateY(-1px);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Risk Badges */
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-critical {
        background: rgba(220, 38, 38, 0.25);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.85rem;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }

    /* Section Cards */
    .section-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: -6px;
        margin-bottom: 14px;
    }

    /* Alert Banner */
    .alert-banner {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .alert-banner-low {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .alert-banner-medium {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        color: #fbbf24;
    }
    .alert-banner-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
    }

    /* System Status Pills */
    .sys-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .sys-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .sys-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .sys-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. PATHS & CACHED DATA LOADERS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV_PATH = PROJECT_ROOT / "data/features/features.csv"
CLEANED_INSAR_PATH = PROJECT_ROOT / "data/processed/insar_cleaned.csv"
SEISMIC_CSV_PATH = PROJECT_ROOT / "data/processed/seismic_cleaned.csv"
XGB_MODEL_PATH = PROJECT_ROOT / "models/xgboost/xgboost.pkl"

RISK_COLOR_MAP = {
    "LOW": "#10b981",       # Emerald
    "MODERATE": "#f59e0b",  # Amber
    "MEDIUM": "#f59e0b",    # Amber
    "HIGH": "#ef4444",      # Red
    "CRITICAL": "#dc2626"   # Deep Crimson
}

RISK_SORT_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3
}


@st.cache_data(show_spinner=False)
def load_insar_feature_dataset() -> pd.DataFrame:
    """Loads the preprocessed 18-feature InSAR dataset with zero fabrication."""
    if not FEATURES_CSV_PATH.exists():
        if CLEANED_INSAR_PATH.exists():
            df = pd.read_csv(CLEANED_INSAR_PATH)
        else:
            raise FileNotFoundError(f"InSAR features file not found at: {FEATURES_CSV_PATH}")
    else:
        df = pd.read_csv(FEATURES_CSV_PATH)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "point_id"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_seismic_dataset() -> pd.DataFrame:
    """Loads the cleaned seismic bumps dataset as an additional geomechanical layer."""
    if SEISMIC_CSV_PATH.exists():
        df = pd.read_csv(SEISMIC_CSV_PATH)
        return df
    return pd.DataFrame()


def check_fastapi_health(api_url: str) -> Dict[str, Any]:
    """Checks live connectivity to the FastAPI backend without blocking."""
    base_url = api_url.rstrip("/").replace("/predict", "")
    try:
        resp = requests.get(f"{base_url}/health", timeout=1.5)
        if resp.status_code == 200:
            return {"connected": True, "details": resp.json()}
    except Exception:
        pass
    return {"connected": False, "details": None}


def fetch_latest_sensor_data(api_url: str) -> Dict[str, Any]:
    """Queries GET /sensor/latest from the FastAPI backend."""
    base_url = api_url.rstrip("/").replace("/predict", "")
    try:
        resp = requests.get(f"{base_url}/sensor/latest", timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "error", "data": None}


# ==============================================================================
# 3. LOAD DATA & SYSTEM STATUS CHECKS
# ==============================================================================

try:
    df_insar = load_insar_feature_dataset()
    insar_loaded = True
except Exception as e:
    df_insar = pd.DataFrame()
    insar_loaded = False
    insar_error = str(e)

df_seismic = load_seismic_dataset()
seismic_loaded = not df_seismic.empty
model_loaded = XGB_MODEL_PATH.exists()


# ==============================================================================
# 4. SIDEBAR CONFIGURATION & FILTERS
# ==============================================================================

with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    api_url = st.text_input(
        "FastAPI Endpoint",
        value="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI server"
    )

    auto_refresh = st.checkbox("Auto-refresh Sensor Node (3s)", value=False)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

    st.markdown("---")
    st.markdown("### 🗺️ GIS Map Filters")

    if insar_loaded and not df_insar.empty:
        # Date Slider Filter
        all_timestamps = sorted(df_insar["timestamp"].unique())
        min_date = all_timestamps[0].date()
        max_date = all_timestamps[-1].date()
        
        date_mode = st.radio(
            "Temporal View",
            ["Latest Observation Epoch", "Historical Date Range"],
            index=0
        )
        
        if date_mode == "Historical Date Range":
            selected_date_range = st.slider(
                "Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )
        else:
            selected_date_range = (max_date, max_date)

        # Risk Class Filter
        available_risks = sorted(df_insar["risk_class"].unique(), key=lambda r: RISK_SORT_ORDER.get(r, 99))
        selected_risks = st.multiselect(
            "Risk Class",
            options=available_risks,
            default=available_risks
        )

        # Point ID Filter
        all_point_ids = ["ALL POINTS"] + sorted(df_insar["point_id"].unique().tolist())
        selected_point_id = st.selectbox(
            "Filter Specific Point",
            options=all_point_ids,
            index=0
        )

        # Displacement Range Filter
        min_disp = float(df_insar["cumulative_displacement_mm"].min())
        max_disp = float(df_insar["cumulative_displacement_mm"].max())
        disp_range = st.slider(
            "Displacement Range (mm)",
            min_value=round(min_disp, 1),
            max_value=round(max_disp, 1),
            value=(round(min_disp, 1), round(max_disp, 1)),
            step=0.5
        )

        # Base Map Tile
        tile_layer = st.selectbox(
            "GIS Map Basemap",
            ["Esri World Imagery (Satellite)", "CartoDB Dark_Matter", "CartoDB Positron", "OpenStreetMap"],
            index=0
        )
    else:
        selected_risks = []
        selected_point_id = "ALL POINTS"

    st.markdown("---")
    st.markdown("### ℹ️ SIH26025 Scope")
    st.caption(
        "**AI-Enabled Low-Cost Real-Time Mine Subsidence Monitoring and Early Warning System**\n\n"
        "• **Layer A:** Sentinel-1 InSAR Satellite Geodesy\n"
        "• **Layer B:** Ground Edge Sensor Prototype\n"
        "• **Layer C:** AI Predictive Early Warning Engine"
    )

# Check API health
api_status = check_fastapi_health(api_url)
sensor_poll = fetch_latest_sensor_data(api_url) if api_status["connected"] else {"status": "no_connection", "data": None}


# ==============================================================================
# 5. HEADER SECTION & LIVE SYSTEM PILLS
# ==============================================================================

fastapi_badge = (
    '<span class="sys-pill sys-online">● FASTAPI: CONNECTED</span>'
    if api_status["connected"]
    else '<span class="sys-pill sys-offline">○ FASTAPI: DISCONNECTED</span>'
)

insar_badge = (
    f'<span class="sys-pill sys-online">● INSAR DB: {len(df_insar):,} EPOCHS</span>'
    if insar_loaded
    else '<span class="sys-pill sys-offline">○ INSAR DB: ERROR</span>'
)

model_badge = (
    '<span class="sys-pill sys-online">● AI MODEL: READY (XGBOOST)</span>'
    if model_loaded
    else '<span class="sys-pill sys-warn">○ AI MODEL: UNAVAILABLE</span>'
)

sensor_data_obj = sensor_poll.get("data")
if sensor_data_obj is not None and "timestamp" in sensor_data_obj:
    sensor_badge = '<span class="sys-pill sys-online">● SENSOR NODE: ONLINE</span>'
else:
    sensor_badge = '<span class="sys-pill sys-warn">○ SENSOR NODE: NO RECENT DATA</span>'

st.markdown(f"""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span class="header-tag">SIH26025</span>
                <span style="color: #64748b; font-size: 0.8rem;">Underground Coal Mine Safety Monitoring</span>
            </div>
            <h1 class="header-title">MINE SUBSIDENCE AI</h1>
            <p class="header-subtitle">AI-Enabled Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System</p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
            {insar_badge}
            {model_badge}
            {fastapi_badge}
            {sensor_badge}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. TOP KPI SECTION
# ==============================================================================

# Calculate real KPI metrics from actual InSAR dataset
if insar_loaded and not df_insar.empty:
    latest_timestamp = df_insar["timestamp"].max()
    df_latest_epoch = df_insar[df_insar["timestamp"] == latest_timestamp]
    
    total_monitored_points = df_insar["point_id"].nunique()
    max_displacement = float(df_insar["cumulative_displacement_mm"].max())
    avg_displacement = float(df_insar["cumulative_displacement_mm"].mean())
    latest_avg_displacement = float(df_latest_epoch["cumulative_displacement_mm"].mean())
    
    # Active alerts in latest epoch
    high_critical_count = len(df_latest_epoch[df_latest_epoch["risk_class"].isin(["HIGH", "CRITICAL"])])
    
    # Current risk state (from latest sensor payload if active, otherwise from InSAR latest epoch)
    if sensor_data_obj and "risk" in sensor_data_obj:
        current_risk_str = str(sensor_data_obj["risk"]).upper()
        risk_source = "IoT Sensor Node (Live)"
    else:
        # Determine basin dominant risk in latest epoch
        risk_counts = df_latest_epoch["risk_class"].value_counts()
        if not risk_counts.empty:
            current_risk_str = risk_counts.index[0]
        else:
            current_risk_str = "LOW"
        risk_source = f"InSAR Epoch ({latest_timestamp.strftime('%Y-%m-%d')})"
else:
    total_monitored_points = 0
    max_displacement = 0.0
    avg_displacement = 0.0
    latest_avg_displacement = 0.0
    high_critical_count = 0
    current_risk_str = "N/A"
    risk_source = "Unavailable"

# Risk Badge Class
if current_risk_str in ["HIGH", "CRITICAL"]:
    risk_badge_html = f'<span class="badge-critical">{current_risk_str}</span>'
elif current_risk_str in ["MEDIUM", "MODERATE"]:
    risk_badge_html = f'<span class="badge-medium">{current_risk_str}</span>'
else:
    risk_badge_html = f'<span class="badge-low">{current_risk_str}</span>'

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Current Risk Level</div>
        <div style="margin: 6px 0;">{risk_badge_html}</div>
        <div class="kpi-sub">{risk_source}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Latest Mean Disp.</div>
        <div class="kpi-value">{latest_avg_displacement:.2f} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
        <div class="kpi-sub">Latest Observation</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Max Recorded Disp.</div>
        <div class="kpi-value" style="color: #ef4444;">{max_displacement:.2f} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
        <div class="kpi-sub">Basin Peak Value</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Basin Disp.</div>
        <div class="kpi-value">{avg_displacement:.2f} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
        <div class="kpi-sub">Cumulative Historical</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    alert_color = "#ef4444" if high_critical_count > 0 else "#10b981"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Active High Risk Points</div>
        <div class="kpi-value" style="color: {alert_color};">{high_critical_count}</div>
        <div class="kpi-sub">High/Critical Alerts</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Monitored Stations</div>
        <div class="kpi-value">{total_monitored_points}</div>
        <div class="kpi-sub">InSAR Points + 1 IoT Node</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ==============================================================================
# 7. FILTER IN-MEMORY DATASET FOR VISUALIZATIONS
# ==============================================================================

if insar_loaded and not df_insar.empty:
    df_filtered = df_insar.copy()

    # Apply date filter
    if date_mode == "Latest Observation Epoch":
        df_filtered = df_filtered[df_filtered["timestamp"] == df_filtered["timestamp"].max()]
    else:
        start_ts = pd.to_datetime(selected_date_range[0])
        end_ts = pd.to_datetime(selected_date_range[1]) + pd.Timedelta(days=1)
        df_filtered = df_filtered[(df_filtered["timestamp"] >= start_ts) & (df_filtered["timestamp"] <= end_ts)]

    # Apply risk filter
    if selected_risks:
        df_filtered = df_filtered[df_filtered["risk_class"].isin(selected_risks)]

    # Apply point filter
    if selected_point_id != "ALL POINTS":
        df_filtered = df_filtered[df_filtered["point_id"] == selected_point_id]

    # Apply displacement range filter
    df_filtered = df_filtered[
        (df_filtered["cumulative_displacement_mm"] >= disp_range[0]) &
        (df_filtered["cumulative_displacement_mm"] <= disp_range[1])
    ]
else:
    df_filtered = pd.DataFrame()


# ==============================================================================
# 8. GIS INTERACTIVE MAP (LARGE SATELLITE INSAR VIEW)
# ==============================================================================

st.markdown("""
<div class="section-card">
    <div class="section-header">
        <span>🛰️ GIS MONITORING MAP — REAL InSAR SATELLITE STATIONS</span>
    </div>
    <div class="section-desc">
        Interactive geospatial map plotting real InSAR observations in the Talcher Coalfields mining basin (Odisha, India).
        Click any station circle to inspect precise displacement, velocity, coherence, slope, and elevation data.
    </div>
</div>
""", unsafe_allow_html=True)

# Determine map center coordinates from real dataset
if not df_filtered.empty:
    center_lat = float(df_filtered["latitude"].mean())
    center_lon = float(df_filtered["longitude"].mean())
elif insar_loaded and not df_insar.empty:
    center_lat = float(df_insar["latitude"].mean())
    center_lon = float(df_insar["longitude"].mean())
else:
    center_lat, center_lon = 20.956, 85.217

# Basemap provider mapping
tile_providers = {
    "Esri World Imagery (Satellite)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri & GIS User Community"
    },
    "CartoDB Dark_Matter": {
        "tiles": "CartoDB dark_matter",
        "attr": "CartoDB"
    },
    "CartoDB Positron": {
        "tiles": "CartoDB positron",
        "attr": "CartoDB"
    },
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": "OpenStreetMap contributors"
    }
}

selected_tile_info = tile_providers.get(tile_layer, tile_providers["Esri World Imagery (Satellite)"])

# Initialize Folium Map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles=selected_tile_info["tiles"],
    attr=selected_tile_info["attr"],
    control_scale=True
)

Fullscreen(position="topright").add_to(m)
MeasureControl(position="topleft").add_to(m)

# Deduplicate points for map display (taking latest observation per point within filtered slice)
if not df_filtered.empty:
    df_map_points = df_filtered.sort_values("timestamp").groupby("point_id").last().reset_index()
    
    for _, row in df_map_points.iterrows():
        p_id = str(row["point_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        disp = float(row["cumulative_displacement_mm"])
        vel = float(row.get("displacement_velocity_mm_month", 0.0))
        acc = float(row.get("displacement_acceleration", 0.0))
        los = float(row.get("los_displacement_mm", 0.0))
        risk = str(row["risk_class"]).upper()
        coh = float(row.get("coherence", 0.0))
        elev = float(row.get("elevation_m", 0.0))
        slope = float(row.get("slope_deg", 0.0))
        rain = float(row.get("rainfall_mm", 0.0))
        ts_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
        
        color = RISK_COLOR_MAP.get(risk, "#3b82f6")
        
        # HTML Popup Content
        popup_html = f"""
        <div style="font-family: 'Inter', sans-serif; min-width: 240px; color: #1e293b; font-size: 12px;">
            <div style="background: #0f172a; color: #f8fafc; padding: 8px 12px; border-radius: 6px 6px 0 0; font-weight: 700; font-size: 13px; display: flex; justify-content: space-between;">
                <span>{p_id}</span>
                <span style="color: {color};">{risk}</span>
            </div>
            <div style="padding: 10px 12px; background: #ffffff; border: 1px solid #e2e8f0; border-top: none; border-radius: 0 0 6px 6px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="color: #64748b; padding: 3px 0;">Observation:</td><td style="font-weight: 600; text-align: right;">{ts_str}</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="color: #64748b; padding: 3px 0;">Cumulative Disp.:</td><td style="font-weight: 700; color: {color}; text-align: right;">{disp:.2f} mm</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="color: #64748b; padding: 3px 0;">Velocity:</td><td style="font-weight: 600; text-align: right;">{vel:.2f} mm/mo</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="color: #64748b; padding: 3px 0;">LOS Disp.:</td><td style="font-weight: 600; text-align: right;">{los:.2f} mm</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="color: #64748b; padding: 3px 0;">Coherence:</td><td style="font-weight: 600; text-align: right;">{coh:.3f}</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="color: #64748b; padding: 3px 0;">Elevation / Slope:</td><td style="font-weight: 600; text-align: right;">{elev:.1f}m / {slope:.1f}°</td></tr>
                    <tr><td style="color: #64748b; padding: 3px 0;">Coordinates:</td><td style="font-family: monospace; font-size: 11px; text-align: right;">{lat:.4f}, {lon:.4f}</td></tr>
                </table>
            </div>
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{p_id} | {risk} | {disp:.2f} mm"
        ).add_to(m)

# Render Folium Map in Streamlit
map_col, info_col = st.columns([3, 1])

with map_col:
    map_output = st_folium(m, width="100%", height=520, returned_objects=["last_object_clicked"])

with info_col:
    st.markdown("#### 📍 Station Inspector")
    
    if not df_filtered.empty:
        # Check if user selected point from dropdown or clicked marker
        inspect_point = selected_point_id if selected_point_id != "ALL POINTS" else "PT_000"
        
        # Point detailed profile
        point_records = df_insar[df_insar["point_id"] == inspect_point].sort_values("timestamp")
        
        if not point_records.empty:
            latest_p_row = point_records.iloc[-1]
            p_risk = str(latest_p_row["risk_class"]).upper()
            p_color = RISK_COLOR_MAP.get(p_risk, "#3b82f6")
            
            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 800; font-size: 1.1rem; color: #f8fafc;">{inspect_point}</span>
                    <span style="background: {p_color}22; color: {p_color}; border: 1px solid {p_color}; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{p_risk}</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: #94a3b8;">
                    <p style="margin: 3px 0;"><b>Latitude:</b> <code>{latest_p_row['latitude']:.6f}</code></p>
                    <p style="margin: 3px 0;"><b>Longitude:</b> <code>{latest_p_row['longitude']:.6f}</code></p>
                    <p style="margin: 3px 0;"><b>Displacement:</b> <span style="color: #f8fafc; font-weight: 700;">{latest_p_row['cumulative_displacement_mm']:.2f} mm</span></p>
                    <p style="margin: 3px 0;"><b>Velocity:</b> {latest_p_row.get('displacement_velocity_mm_month', 0.0):.2f} mm/mo</p>
                    <p style="margin: 3px 0;"><b>Elevation:</b> {latest_p_row.get('elevation_m', 0.0):.1f} m</p>
                    <p style="margin: 3px 0;"><b>Slope:</b> {latest_p_row.get('slope_deg', 0.0):.1f}°</p>
                    <p style="margin: 3px 0;"><b>Coherence:</b> {latest_p_row.get('coherence', 0.0):.3f}</p>
                    <p style="margin: 3px 0;"><b>Mining Factor:</b> {latest_p_row.get('mining_factor', 0.0):.3f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Mini sparkline for this station
            fig_spark = px.line(
                point_records,
                x="timestamp",
                y="cumulative_displacement_mm",
                title=f"{inspect_point} Displacement History",
                template="plotly_dark"
            )
            fig_spark.update_layout(
                height=160,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="",
                yaxis_title="Disp. (mm)",
                font=dict(size=9)
            )
            st.plotly_chart(fig_spark, use_container_width=True)
    else:
        st.info("Adjust sidebar filters to view station details.")


# ==============================================================================
# 9. REAL-TIME PROTOTYPE SENSOR MONITORING PANEL (IoT NODE)
# ==============================================================================

st.markdown("""
<div class="section-card">
    <div class="section-header">
        <span>⚡ REAL-TIME PROTOTYPE SENSOR MONITORING (IoT Node)</span>
    </div>
    <div class="section-desc">
        Ground-level Edge Sensor Node telemetry transmitted from Wokwi ESP32 simulation to FastAPI.
        <b>Note:</b> Prototype sensors monitor real-time ground tilt, strain, crack width, and displacement at the mine face without GPS spoofing.
    </div>
</div>
""", unsafe_allow_html=True)

sensor_left, sensor_right = st.columns([3, 2])

with sensor_left:
    st.markdown("##### 📡 Live Telemetry Stream")
    
    if sensor_data_obj is not None:
        t_val = sensor_data_obj.get("tilt", 0.0)
        d_val = sensor_data_obj.get("displacement", 0.0)
        c_val = sensor_data_obj.get("crack_width", 0.0)
        s_val = sensor_data_obj.get("strain", 0.0)
        ts_raw = sensor_data_obj.get("timestamp", "N/A")
        
        # Risk output from backend
        b_risk = str(sensor_data_obj.get("risk", "LOW")).upper()
        b_conf = sensor_data_obj.get("confidence", 0.0)
        b_action = sensor_data_obj.get("action", "NORMAL")
        b_led = sensor_data_obj.get("led_signal", "GREEN")
        b_msg = sensor_data_obj.get("message", "Ground conditions are stable.")
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Tilt (MPU6050)", f"{t_val:.2f}°", help="Inclinometer angular tilt")
        with sc2:
            st.metric("Displacement (Ultrasonic)", f"{d_val:.2f} mm", help="HC-SR04 surface displacement")
        with sc3:
            st.metric("Crack Width", f"{c_val:.2f} mm", help="Extensometer joint aperture")
        with sc4:
            st.metric("Strain", f"{s_val:.4f} µε", help="Ground strain gauge")

        # Alert Banner
        if b_risk == "HIGH":
            banner_cls = "alert-banner-high"
            icon = "🚨"
            status_text = "HIGH RISK — IMMEDIATE ATTENTION REQUIRED"
        elif b_risk in ["MEDIUM", "MODERATE"]:
            banner_cls = "alert-banner-medium"
            icon = "⚠️"
            status_text = "MODERATE RISK — INCREASE MONITORING FREQUENCY"
        else:
            banner_cls = "alert-banner-low"
            icon = "✅"
            status_text = "NORMAL — GROUND CONDITIONS STABLE"

        st.markdown(f"""
        <div class="alert-banner {banner_cls}">
            <span style="font-size: 1.4rem;">{icon}</span>
            <div style="flex: 1;">
                <div style="font-size: 0.95rem; font-weight: 700;">{status_text}</div>
                <div style="font-size: 0.8rem; font-weight: 400; opacity: 0.9;">Directive: {b_msg} | Action Signal: <b>{b_action}</b> | Hardware LED: <b>{b_led}</b></div>
            </div>
            <div style="text-align: right; font-size: 0.75rem;">
                <div>Confidence: <b>{b_conf*100:.1f}%</b></div>
                <div style="font-family: monospace; color: #94a3b8;">{ts_raw[:19] if len(ts_raw) >= 19 else ts_raw}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No sensor payload received yet from FastAPI. Use the test console on the right to transmit sample sensor values or start the Wokwi simulation.")

with sensor_right:
    st.markdown("##### 🧪 Sensor Transmission & Simulation Console")
    with st.expander("Test Custom Sensor Payload (POST /predict)", expanded=True):
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_tilt = st.number_input("Tilt (°)", min_value=-90.0, max_value=90.0, value=2.5, step=0.5)
            sim_crack = st.number_input("Crack Width (mm)", min_value=0.0, max_value=50.0, value=1.2, step=0.1)
        with sim_c2:
            sim_disp = st.number_input("Displacement (mm)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
            sim_strain = st.number_input("Strain (µε)", min_value=0.0, max_value=1.0, value=0.0035, step=0.0005, format="%.4f")

        if st.button("🚀 Transmit to FastAPI /predict", use_container_width=True):
            payload = {
                "tilt": float(sim_tilt),
                "displacement": float(sim_disp),
                "crack_width": float(sim_crack),
                "strain": float(sim_strain)
            }
            try:
                post_url = f"{api_url.rstrip('/')}/predict"
                resp = requests.post(post_url, json=payload, timeout=3.0)
                if resp.status_code == 200:
                    st.success("✅ Payload accepted! Risk evaluated and latest state updated.")
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {resp.status_code} - {resp.text}")
            except Exception as ex:
                st.error(f"❌ Connection failed: {ex}")


# ==============================================================================
# 10. DEFORMATION TIME SERIES & TEMPORAL ANALYTICS
# ==============================================================================

st.markdown("""
<div class="section-card">
    <div class="section-header">
        <span>📈 DEFORMATION TIME-SERIES & TREND ANALYSIS</span>
    </div>
    <div class="section-desc">
        Multi-temporal InSAR displacement progression (2020–2023) across monitoring points with velocity and rainfall cross-correlation.
    </div>
</div>
""", unsafe_allow_html=True)

if insar_loaded and not df_insar.empty:
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Time-series plot of cumulative displacement
        if selected_point_id != "ALL POINTS":
            ts_data = df_insar[df_insar["point_id"] == selected_point_id].sort_values("timestamp")
            chart_title = f"Cumulative Displacement vs Time — {selected_point_id}"
        else:
            ts_data = df_insar.sort_values(["timestamp", "point_id"])
            chart_title = "Cumulative Displacement vs Time — All Stations"

        fig_ts = px.line(
            ts_data,
            x="timestamp",
            y="cumulative_displacement_mm",
            color="point_id" if selected_point_id == "ALL POINTS" else None,
            title=chart_title,
            labels={"cumulative_displacement_mm": "Cumulative Displacement (mm)", "timestamp": "Observation Epoch"},
            template="plotly_dark"
        )
        fig_ts.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if selected_point_id == "ALL POINTS" else None
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with chart_col2:
        # Velocity and Rainfall dynamics
        if selected_point_id != "ALL POINTS":
            pt_dyn = df_insar[df_insar["point_id"] == selected_point_id].sort_values("timestamp")
        else:
            pt_dyn = df_insar.groupby("timestamp").agg({
                "displacement_velocity_mm_month": "mean",
                "rainfall_mm": "mean"
            }).reset_index()

        fig_dual = go.Figure()
        
        # Velocity line
        fig_dual.add_trace(go.Scatter(
            x=pt_dyn["timestamp"],
            y=pt_dyn["displacement_velocity_mm_month"],
            name="Velocity (mm/mo)",
            line=dict(color="#38bdf8", width=2.5),
            yaxis="y1"
        ))
        
        # Rainfall bar
        fig_dual.add_trace(go.Bar(
            x=pt_dyn["timestamp"],
            y=pt_dyn["rainfall_mm"],
            name="Rainfall (mm)",
            marker_color="rgba(168, 85, 247, 0.4)",
            yaxis="y2"
        ))

        fig_dual.update_layout(
            title="Displacement Velocity & Rainfall Correlation",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            yaxis=dict(title=dict(text="Velocity (mm/month)", font=dict(color="#38bdf8")), tickfont=dict(color="#38bdf8")),
            yaxis2=dict(title=dict(text="Rainfall (mm)", font=dict(color="#a855f7")), tickfont=dict(color="#a855f7"), overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dual, use_container_width=True)


# ==============================================================================
# 11. RISK DISTRIBUTION & SPATIAL CORRELATIONS
# ==============================================================================

stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>📊 RISK CLASS DISTRIBUTION (InSAR DATASET)</span>
        </div>
        <div class="section-desc">Proportion of ground stations across safety risk tiers.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if insar_loaded and not df_insar.empty:
        risk_counts = df_insar["risk_class"].value_counts().reset_index()
        risk_counts.columns = ["risk_class", "count"]
        
        fig_pie = px.pie(
            risk_counts,
            names="risk_class",
            values="count",
            color="risk_class",
            color_discrete_map=RISK_COLOR_MAP,
            hole=0.45,
            template="plotly_dark"
        )
        fig_pie.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#0f172a"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with stat_col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>⛏️ DISPLACEMENT VS MINING FACTOR</span>
        </div>
        <div class="section-desc">Ground deformation sensitivity relative to active mining intensity.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if insar_loaded and not df_insar.empty:
        fig_scatter = px.scatter(
            df_insar,
            x="mining_factor",
            y="cumulative_displacement_mm",
            color="risk_class",
            color_discrete_map=RISK_COLOR_MAP,
            labels={"mining_factor": "Mining Extraction Factor", "cumulative_displacement_mm": "Cumulative Disp. (mm)"},
            template="plotly_dark"
        )
        fig_scatter.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# 12. SEISMIC MONITORING LAYER (ADDITIONAL GEOMECHANICAL DATA)
# ==============================================================================

with st.expander("🔬 Seismic Activity — Additional Geomechanical Monitoring Layer", expanded=False):
    st.markdown("""
    <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">
        <b>Note:</b> Real seismic dataset (<code>seismic_cleaned.csv</code>, 170 records) measuring seismic energy pulses, seismic bumps (<code>nbumps</code>), and acoustic hazard levels (<code>ghazard</code>).
        This layer provides complementary deep geomechanical strain context alongside satellite InSAR.
    </div>
    """, unsafe_allow_html=True)
    
    if seismic_loaded:
        seis_c1, seis_c2, seis_c3 = st.columns(3)
        with seis_c1:
            st.metric("Total Seismic Records", f"{len(df_seismic)}")
        with seis_c2:
            st.metric("Max Seismic Energy", f"{df_seismic['energy'].max():,} J")
        with seis_c3:
            hazard_counts = df_seismic["ghazard"].value_counts().to_dict() if "ghazard" in df_seismic.columns else {}
            st.metric("Hazard Level 'a' (Low)", f"{hazard_counts.get('a', 0)}")

        # Energy vs Bump Count Plot
        if "energy" in df_seismic.columns and "nbumps" in df_seismic.columns:
            fig_seis = px.bar(
                df_seismic.head(40),
                x=df_seismic.head(40).index,
                y="energy",
                color="ghazard" if "ghazard" in df_seismic.columns else None,
                title="Seismic Pulse Energy (Joules) & Hazard Classification",
                labels={"energy": "Energy (J)", "index": "Record ID"},
                template="plotly_dark"
            )
            fig_seis.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a"
            )
            st.plotly_chart(fig_seis, use_container_width=True)
    else:
        st.warning("Seismic dataset not found in data/processed/seismic_cleaned.csv.")


# ==============================================================================
# 13. DATA QUALITY & AUDIT TRAIL
# ==============================================================================

with st.expander("📋 Data Quality, Provenance & Scientific Audit", expanded=False):
    if insar_loaded and not df_insar.empty:
        dq1, dq2, dq3, dq4 = st.columns(4)
        with dq1:
            st.markdown(f"**Total Records:** `{len(df_insar):,}`")
            st.markdown(f"**Unique Ground Points:** `{df_insar['point_id'].nunique()}`")
        with dq2:
            st.markdown(f"**Observation Start:** `{df_insar['timestamp'].min().strftime('%Y-%m-%d')}`")
            st.markdown(f"**Observation End:** `{df_insar['timestamp'].max().strftime('%Y-%m-%d')}`")
        with dq3:
            st.markdown(f"**Latitude Range:** `[{df_insar['latitude'].min():.4f}, {df_insar['latitude'].max():.4f}]`")
            st.markdown(f"**Longitude Range:** `[{df_insar['longitude'].min():.4f}, {df_insar['longitude'].max():.4f}]`")
        with dq4:
            st.markdown(f"**Missing Values:** `{int(df_insar.isna().sum().sum())}`")
            st.markdown(f"**Model Features:** `18 Engineered Features`")

        st.markdown("---")
        st.markdown("##### Sample InSAR Observations")
        st.dataframe(df_insar.head(10), use_container_width=True)


# ==============================================================================
# 14. FOOTER
# ==============================================================================

st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.78rem; margin-top: 30px; padding: 12px; border-top: 1px solid #1e293b;">
    <b>SIH26025</b> — AI-Enabled Low-Cost Real-Time Mine Subsidence Monitoring, Prediction and Early Warning System<br>
    Pairing Spaceborne Sentinel-1 InSAR Geodesy with Low-Cost Ground IoT Edge Sensor Nodes for Coal Mine Safety in India.
</div>
""", unsafe_allow_html=True)