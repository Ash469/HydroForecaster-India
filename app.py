import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="HydroForecaster India | Pan-India Groundwater Intelligence",
    layout="wide"
)

# Professional Academic Design
st.markdown("""
<style>
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 24px;
        line-height: 1.5;
    }
    .metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-number {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-caption {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    .nav-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero-title'>HydroForecaster India</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Pan-India Groundwater Level Forecasting Platform | Department of Civil Engineering, IIT Guwahati (B.Tech Project CE4198P)</div>", unsafe_allow_html=True)

# Overview Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-title">Audited Telemetry</div>
        <div class="metric-number">52.09 Million</div>
        <div class="metric-caption">171 raw Parquet files (2021-2025)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-title">Active Monitoring Wells</div>
        <div class="metric-number">4,879 Wells</div>
        <div class="metric-caption">Pan-India Telemetric Network</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-title">Forecast Horizons</div>
        <div class="metric-number">14d / 30d / 60d</div>
        <div class="metric-caption">Multi-horizon strategic outlook</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-title">National 30-Day RMSE</div>
        <div class="metric-number">3.98 Meters</div>
        <div class="metric-caption">Sub-4.0m error barrier achieved</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Operational Platform Structure")
st.markdown("""
Please use the **navigation sidebar on the left** to explore the four specialized analytical modules:

1. **1_Data_Observatory:** Complete audit tracing the reduction of 52.09M raw telemetry pings to 4.34M daily medians and 2.78M ML feature rows, with spatial inverse distance weighting (IDW) and state-by-state station counts.
2. **2_Spatial_EDA_GIS:** High-density spatial mapping of all 4,465 monitoring wells across India, depth distributions, and empirical vadose zone infiltration delay curves.
3. **3_Model_Leaderboard:** Multi-horizon comparative tournament evaluating Tier 1 (Linear Baselines), Tier 2 (Decision Trees & Random Forests), Tier 3 (Gradient Boosting), and Tier 4 (Regularized Optimization) with both academic and layman-translated metrics.
4. **4_Interactive_Forecaster:** Location-based prediction engine allowing interactive selection of any well in India with real-time multi-model forecasting and practical operational tubewell advisories.
""")

st.markdown("---")
st.caption("Central Ground Water Board (CGWB) & Central Water Commission (CWC) Telemetric Telemetry Store | IIT Guwahati Department of Civil Engineering (CE4198P)")
