import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="GIS Spatial Observatory | Pan-India Groundwater",
    layout="wide"
)

# Academic Styling
st.markdown("""
<style>
    .title-text {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .subtitle-text {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 20px;
        margin-bottom: 12px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-text'>GIS Spatial Observatory</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Pan-India interactive mapping of 4,465 monitoring wells. Select any timeline month to observe the spatial distribution of water table depth, accumulated rainfall, ambient temperature, humidity, solar radiation, and atmospheric pressure.</div>", unsafe_allow_html=True)

@st.cache_data
def load_monthly_data():
    df = pd.read_parquet('dashboard_data/monthly_station_features.parquet')
    # Filter valid coordinates in India mainland
    df = df[(df['latitude'] >= 6.0) & (df['latitude'] <= 36.0) & (df['longitude'] >= 68.0) & (df['longitude'] <= 98.0)].copy()
    
    # Pre-round all numerical feature columns to exactly 2 decimal places
    num_cols = ['gwl_m', 'rainfall_idw_mm', 'temp_c', 'humidity_pct', 'solar_wm2', 'pressure_mb', 'elevation_msl']
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].round(2)
            
    return df

df_monthly = load_monthly_data()
all_months = sorted(df_monthly['year_month'].unique().tolist())

month_labels = {
    m: f"{m} ({pd.to_datetime(m).strftime('%B %Y')})" for m in all_months
}

# CONTROLS
col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    selected_month = st.selectbox(
        "Select Timeline Month to Inspect Across India:",
        all_months,
        index=all_months.index("2024-08") if "2024-08" in all_months else len(all_months)-5,
        format_func=lambda x: month_labels.get(x, x)
    )

with col_m2:
    color_variable = st.selectbox(
        "Select Feature Layer for Spatial Map:",
        [
            "Groundwater Depth (m bgl)",
            "Monthly Rainfall Flux (mm)",
            "Ambient Temperature (°C)",
            "Relative Humidity (%)",
            "Solar Radiation (W/m²)",
            "Atmospheric Pressure (mb)"
        ]
    )

# Filter dataset for selected month
df_slice = df_monthly[df_monthly['year_month'] == selected_month].copy()

# Variable mapping configuration
var_config = {
    "Groundwater Depth (m bgl)": {
        "col": "gwl_m", "title": "Water Table Depth (m bgl)", "scale": "Turbo_r",
        "range": [-80.0, 0.0]
    },
    "Monthly Rainfall Flux (mm)": {
        "col": "rainfall_idw_mm", "title": "Cumulative Monthly Rainfall (mm)", "scale": "Viridis",
        "range": [0, 450]
    },
    "Ambient Temperature (°C)": {
        "col": "temp_c", "title": "Mean Ambient Temperature (°C)", "scale": "Thermal",
        "range": [10, 42]
    },
    "Relative Humidity (%)": {
        "col": "humidity_pct", "title": "Mean Relative Humidity (%)", "scale": "Teal",
        "range": [20, 95]
    },
    "Solar Radiation (W/m²)": {
        "col": "solar_wm2", "title": "Mean Solar Radiation (W/m²)", "scale": "Cividis",
        "range": [100, 350]
    },
    "Atmospheric Pressure (mb)": {
        "col": "pressure_mb", "title": "Atmospheric Pressure (mb)", "scale": "Blues",
        "range": [920, 1015]
    }
}

cfg = var_config[color_variable]

st.markdown(f"<div class='section-header'>National Telemetry Map: {cfg['title']} ({selected_month}) — Showing {len(df_slice):,} Wells</div>", unsafe_allow_html=True)

# Hover data formatted explicitly to 2 decimal places
hover_dict = {
    "state": True,
    "district": True,
    "elevation_msl": ":.2f",
    "gwl_m": ":.2f",
    "rainfall_idw_mm": ":.2f",
    "temp_c": ":.2f",
    "humidity_pct": ":.2f",
    "solar_wm2": ":.2f",
    "pressure_mb": ":.2f",
    "latitude": False,
    "longitude": False
}

# Full Mainland India bounds fitting (Supporting Plotly v6+ scatter_map & legacy scatter_mapbox)
map_kwargs = dict(
    lat="latitude",
    lon="longitude",
    color=cfg["col"],
    color_continuous_scale=cfg["scale"],
    range_color=cfg["range"],
    hover_name="station_name",
    hover_data=hover_dict,
    zoom=3.7,
    center=dict(lat=21.5, lon=82.0),
    labels={
        "gwl_m": "Water Table Depth (m bgl)",
        "rainfall_idw_mm": "Monthly Rain (mm)",
        "temp_c": "Temperature (°C)",
        "humidity_pct": "Humidity (%)",
        "solar_wm2": "Solar Rad (W/m²)",
        "pressure_mb": "Pressure (mb)",
        "elevation_msl": "Altitude (m MSL)"
    }
)

if hasattr(px, "scatter_map"):
    fig_map = px.scatter_map(
        df_slice,
        map_style="open-street-map",
        **map_kwargs
    )
else:
    fig_map = px.scatter_mapbox(
        df_slice,
        mapbox_style="open-street-map",
        **map_kwargs
    )

# Customizing marker size and colorbar to ensure full India visibility
fig_map.update_traces(marker=dict(size=5.5, opacity=0.85))

fig_map.update_layout(
    height=780,
    margin=dict(l=0, r=0, t=10, b=0),
    coloraxis_colorbar=dict(
        title=cfg["title"],
        thickness=18,
        len=0.70,
        yanchor="middle",
        y=0.5
    )
)

st.plotly_chart(fig_map, width='stretch')

st.markdown("---")
st.caption("Central Ground Water Board (CGWB) & Central Water Commission (CWC) Telemetric Telemetry Store | IIT Guwahati Department of Civil Engineering (CE4198P)")
