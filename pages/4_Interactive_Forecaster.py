import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import lightgbm as lgb
import pyarrow.parquet as pq

st.set_page_config(page_title="Well Forecaster | HydroForecaster India", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 22px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #2563EB;">
    <h2 style="color: #FFFFFF; margin: 0; font-size: 1.6rem; letter-spacing: -0.5px;">Screen 4: Interactive Monitoring Well Forecaster & Inference Engine</h2>
    <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 0.95rem;">
        Station-Level Real-Time Hydrogeological Inference Across India's Telemetric Observation Network (t+14, t+30, t+60 Days)
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_forecast_metadata():
    stations_df = pd.read_parquet('dashboard_data/stations_directory.parquet')
    return stations_df

@st.cache_data
def load_station_recent(station_name):
    table = pq.read_table('dashboard_data/recent_telemetry_2024_2025.parquet', filters=[('station_name', '=', station_name)])
    df_st = table.to_pandas().sort_values('timestamp')
    return df_st

stations_df = load_forecast_metadata()

# Selection Panel
st.markdown("#### Station & Forecast Horizon Selection")
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.6, 1.2])

with col1:
    states = sorted(stations_df['state'].dropna().unique().tolist())
    # Default to Assam if available, or prominent state
    default_state = "Assam" if "Assam" in states else ("Uttar Pradesh" if "Uttar Pradesh" in states else states[0])
    selected_state = st.selectbox("1. State / Union Territory", states, index=states.index(default_state))

with col2:
    districts = sorted(stations_df[stations_df['state'] == selected_state]['district'].dropna().unique().tolist())
    selected_district = st.selectbox("2. District", districts)

with col3:
    stations = sorted(stations_df[(stations_df['state'] == selected_state) & (stations_df['district'] == selected_district)]['station_name'].dropna().unique().tolist())
    selected_station = st.selectbox("3. DWLR Monitoring Well", stations)

with col4:
    horizon_map = {
        "14 Days (t+14)": "t_14",
        "30 Days (t+30)": "t_30",
        "60 Days (t+60)": "t_60"
    }
    selected_horizon_label = st.selectbox("4. Planning Horizon", list(horizon_map.keys()), index=1)
    horizon_code = horizon_map[selected_horizon_label]
    horizon_days = int(horizon_code.split('_')[1])

# Regional Telemetry Note & Academic Audit for Assam & North-East
with st.expander("Regional Data Coverage Note: Assam, North-East & Uttar Pradesh Telemetry Network"):
    st.markdown("""
    **Hydrological Telemetry Inventory:**
    * **Assam Telemetric Network:** Assam is fully integrated with **73 telemetric surface sensing stations** (40 CWC rainfall stations, 10 relative humidity, 10 solar pyranometers, 10 pressure, 3 river hydrometric gauges) and automated DWLR monitoring wells (Goalpara, Marigaon, Tinsukia) providing ground truth piezometer readings.
    * **Uttar Pradesh Telemetry Network:** Uttar Pradesh is now fully integrated with **170 automated DWLR monitoring wells across 38 districts** along the central and upper Gangetic plains.
    * **North-Eastern Network:** Active DWLR piezometers are fully connected across Assam, Meghalaya (15 stations), Arunachal Pradesh (4 stations), Tripura (6 stations), and Nagaland (1 station).
    """)

st.markdown("---")

# Load station time series
df_station = load_station_recent(selected_station)

if df_station.empty or len(df_station) < 30:
    st.error("Insufficient continuous telemetry records for the selected well. Please select another station from the directory.")
else:
    latest_row = df_station.iloc[-1]
    curr_depth = float(latest_row['gwl_m'])
    rain_90d = float(latest_row['rainfall_roll90_sum']) if 'rainfall_roll90_sum' in latest_row else 0.0
    temp_curr = float(latest_row['temp_c']) if 'temp_c' in latest_row else 25.0
    latest_date_str = pd.to_datetime(latest_row['timestamp']).strftime('%Y-%m-%d')

    # Operational Telemetry Cards
    st.markdown("#### Baseline Telemetry Metrics (Latest Observation Date: " + latest_date_str + ")")
    c_t1, c_t2, c_t3, c_t4 = st.columns(4)
    c_t1.metric("Current Water Table Depth", f"{curr_depth:.2f} m bgl", "Meters below ground level")
    c_t2.metric("Cumulative 90-Day Rainfall", f"{rain_90d:.2f} mm", "Deep percolation flux")
    c_t3.metric("Ambient Temperature", f"{temp_curr:.2f} °C", "Atmospheric demand")
    c_t4.metric("Forecast Planning Target", f"+{horizon_days} Days Ahead", selected_horizon_label)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Multi-Model Prospective Forecasts (Horizon: t+" + str(horizon_days) + " Days)")

    # Load pre-trained inference models
    feature_names = joblib.load('models/feature_names.joblib')
    
    sample_feat_dict = {}
    for col in feature_names:
        if col in df_station.columns:
            sample_feat_dict[col] = latest_row[col]
        else:
            sample_feat_dict[col] = 0.0
            
    X_pred = pd.DataFrame([sample_feat_dict])[feature_names].fillna(0.0).values.astype(np.float32)

    # 1. Persistence Baseline
    pred_persist = curr_depth

    # 2. Ridge L2 Linear
    ridge_model = joblib.load(f'models/ridge_{horizon_code}.joblib')
    pred_ridge = float(ridge_model.predict(X_pred)[0])

    # 3. Random Forest (Tier 2 Non-linear Ensemble)
    rf_model = joblib.load(f'models/rf_{horizon_code}.joblib')
    pred_rf = float(rf_model.predict(X_pred)[0])

    # 4. LightGBM (Tier 3 Gradient Boosted Trees)
    lgb_booster = lgb.Booster(model_file=f'models/lgb_{horizon_code}.txt')
    pred_lgb = float(lgb_booster.predict(X_pred)[0])

    # Display Predictions Grid
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border-radius: 6px; padding: 16px; border: 1px solid #E2E8F0;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">Persistence Baseline</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #1E293B; margin-top: 4px;">{pred_persist:.2f} m</div>
            <div style="font-size: 0.82rem; color: #64748B; margin-top: 4px;">Δ 0.00 m (Zero change assumption)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border-radius: 6px; padding: 16px; border: 1px solid #CBD5E1; border-top: 3px solid #2563EB;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.5px;">Ridge L2 Regularized</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #1D4ED8; margin-top: 4px;">{pred_ridge:.2f} m</div>
            <div style="font-size: 0.82rem; color: #2563EB; margin-top: 4px;">Δ {pred_ridge - curr_depth:+.2f} m linear trajectory</div>
        </div>
        """, unsafe_allow_html=True)

    with col_p3:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border-radius: 6px; padding: 16px; border: 1px solid #CBD5E1; border-top: 3px solid #059669;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #065F46; text-transform: uppercase; letter-spacing: 0.5px;">Random Forest (Tier 2)</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #047857; margin-top: 4px;">{pred_rf:.2f} m</div>
            <div style="font-size: 0.82rem; color: #059669; margin-top: 4px;">Δ {pred_rf - curr_depth:+.2f} m ensemble mean</div>
        </div>
        """, unsafe_allow_html=True)

    with col_p4:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border-radius: 6px; padding: 16px; border: 1px solid #CBD5E1; border-top: 3px solid #D97706;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #92400E; text-transform: uppercase; letter-spacing: 0.5px;">LightGBM (Tier 3 Primary)</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #B45309; margin-top: 4px;">{pred_lgb:.2f} m</div>
            <div style="font-size: 0.82rem; color: #D97706; margin-top: 4px;">Δ {pred_lgb - curr_depth:+.2f} m gradient boosted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Practical Ground Water Advisory
    st.markdown("#### Hydrological Advisory & Tubewell Operations Analysis")
    
    water_change = pred_lgb - curr_depth
    forecast_depth = pred_lgb

    if forecast_depth > -10.0:
        st.markdown(f"""
        <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 5px solid #16A34A; padding: 16px; border-radius: 6px;">
            <div style="font-weight: 700; color: #166534; font-size: 1rem;">STATUS: SAFE OPERATIONAL EQUILIBRIUM</div>
            <div style="color: #14532D; font-size: 0.92rem; margin-top: 6px; line-height: 1.5;">
                • <b>Aquifer State:</b> The predicted water table depth at t+{horizon_days} days is <b>{forecast_depth:.2f} m bgl</b> (projected net variance of <b>{water_change:+.2f} m</b>). The aquifer remains within optimal recharge recovery parameters.<br>
                • <b>Tubewell Operation:</b> Standard centrifugal and surface suction pumps will operate efficiently without cavitation risks.<br>
                • <b>Water Management:</b> Safe for standard crop rotation and planned irrigation discharge without artificial replenishment constraints.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif forecast_depth > -30.0:
        st.markdown(f"""
        <div style="background-color: #FEFCE8; border: 1px solid #FEF08A; border-left: 5px solid #CA8A04; padding: 16px; border-radius: 6px;">
            <div style="font-weight: 700; color: #854D0E; font-size: 1rem;">STATUS: MODERATE SEASONAL DRAWDOWN</div>
            <div style="color: #713F12; font-size: 0.92rem; margin-top: 6px; line-height: 1.5;">
                • <b>Aquifer State:</b> The predicted water table depth at t+{horizon_days} days is <b>{forecast_depth:.2f} m bgl</b> (projected net variance of <b>{water_change:+.2f} m</b>). Reflects active seasonal pumping drawdown and unreplenished evaporative loss.<br>
                • <b>Tubewell Operation:</b> Deep-well submersible pumps must be deployed. Monitor pump intake filters for sand entry caused by drawdown cone depression.<br>
                • <b>Water Management:</b> Shift irrigation schedules to low-evaporative nighttime windows and encourage micro-irrigation (drip/sprinkler) methods.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-left: 5px solid #DC2626; padding: 16px; border-radius: 6px;">
            <div style="font-weight: 700; color: #991B1B; font-size: 1rem;">STATUS: CRITICAL AQUIFER OVER-DRAFT ALERT</div>
            <div style="color: #7F1D1D; font-size: 0.92rem; margin-top: 6px; line-height: 1.5;">
                • <b>Aquifer State:</b> The predicted water table depth at t+{horizon_days} days is <b>{forecast_depth:.2f} m bgl</b> (projected net variance of <b>{water_change:+.2f} m</b>). The monitoring zone is operating under severe hydrogeological depletion.<br>
                • <b>Tubewell Operation:</b> Severe danger of suction break, pump motor burn-out, and localized aquifer compaction.<br>
                • <b>Water Management:</b> Enforce moratorium on new extraction borewells and mandate immediate artificial aquifer recharge via percolation tanks and injection wells.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Station Hydrograph
    st.markdown("#### Spatiotemporal Hydrograph & Prospective Trajectory")
    
    fig_ts = go.Figure()
    
    # Historical actual water table
    fig_ts.add_trace(go.Scatter(
        x=df_station['timestamp'],
        y=df_station['gwl_m'],
        mode='lines',
        name='Observed DWLR Telemetry (m bgl)',
        line=dict(color='#0F172A', width=2)
    ))

    # Prospective prediction dates
    target_date = latest_row['timestamp'] + pd.Timedelta(days=horizon_days)
    
    # Tier 3 (LightGBM) Forecast
    fig_ts.add_trace(go.Scatter(
        x=[latest_row['timestamp'], target_date],
        y=[curr_depth, pred_lgb],
        mode='lines+markers',
        name=f'LightGBM Tier 3 Forecast ({pred_lgb:.2f} m)',
        line=dict(color='#D97706', width=2.5, dash='dash'),
        marker=dict(size=[5, 9], color=['#D97706', '#D97706'], symbol=['circle', 'diamond'])
    ))

    # Tier 2 (Random Forest) Forecast
    fig_ts.add_trace(go.Scatter(
        x=[latest_row['timestamp'], target_date],
        y=[curr_depth, pred_rf],
        mode='lines+markers',
        name=f'Random Forest Tier 2 Forecast ({pred_rf:.2f} m)',
        line=dict(color='#059669', width=2, dash='dot'),
        marker=dict(size=[4, 8], color='#059669', symbol=['circle', 'square'])
    ))

    # Tier 1 (Ridge) Forecast
    fig_ts.add_trace(go.Scatter(
        x=[latest_row['timestamp'], target_date],
        y=[curr_depth, pred_ridge],
        mode='lines+markers',
        name=f'Ridge L2 Tier 1 Forecast ({pred_ridge:.2f} m)',
        line=dict(color='#2563EB', width=1.5, dash='dashdot'),
        marker=dict(size=[3, 7], color='#2563EB', symbol=['circle', 'circle'])
    ))

    fig_ts.update_layout(
        title=f"Hydrograph Trajectory: {selected_station} | {selected_district}, {selected_state}",
        xaxis_title="Telemetry Timestamp",
        yaxis_title="Water Table Depth (m bgl, negative values indicate depth below surface)",
        height=480,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9')
    )
    st.plotly_chart(fig_ts, width='stretch')
