# Pan-India Groundwater Level Forecasting Using Machine Learning & Hydro-Climatic Data
> **B.Tech Project (CE4198P) — Department of Civil Engineering, Indian Institute of Technology (IIT) Guwahati**  
> *A Continental-Scale Spatiotemporal Machine Learning Framework for Multi-Horizon Aquifer Depth Prediction and Tubewell Operational Advisory*

---

## 📌 Executive Summary & Academic Context

* **Institution:** Department of Civil Engineering, IIT Guwahati
* **Course:** B.Tech Project (CE4198P)
* **Supervisor Mandate:** Design and build a **Pan-India machine learning framework** rather than a localized single-district model, integrating heterogeneous government water data across all major Indian hydrogeological provinces (Gangetic alluvial plains, hard-rock basalt Deccan traps, arid Thar desert, and Brahmaputra floodplains).
* **Multi-Horizon Forecasting Targets:**
  * **$t+14$ Days (2 Weeks Ahead):** Immediate tubewell pumping regulation & local irrigation rotation.
  * **$t+30$ Days (1 Month Ahead):** District-scale canal scheduling and reservoir water release planning.
  * **$t+60$ Days (2 Months Ahead):** Seasonal drought mitigation, pre/post-monsoon crop selection, and aquifer recharge planning.
* **Master Scale & Deliverables:**
  * **52,088,570 raw telemetry points** audited across 171 government datasets (CGWB, CWC, IMD, State WRDs).
  * **6,306,419 hardware sensor glitches** (12.11% corruption rate) detected and filtered.
  * **4,879 active Digital Water Level Recorder (DWLR) piezometers** across **28 States and Union Territories**.
  * **Master Feature Store:** `3,003,349 rows × 62 engineered columns` with zero temporal leakage.
  * **Interactive Web Platform:** 4-screen Streamlit decision-support dashboard (`app.py`).

---

## 🏗️ System Architecture & End-to-End Pipeline

```mermaid
graph TD
    subgraph S1 [Phase 1: Telemetry Ingestion & Quality Audit]
        A[52.09M Raw Telemetry Records - 171 Files] --> B[Glitch & Outlier Filtering - 6.31M Corrupt Filtered]
        B --> C[Darcy Median Daily Aggregation]
    end

    subgraph S2 [Phase 2: Hydrological Feature Engineering]
        C --> D[3D Spherical cKDTree Indexing - R=6371 km]
        D --> E[Spatial Inverse Distance Weighting - 3 Nearest Gauges]
        E --> F[Vadose Infiltration Windows - 7, 14, 30, 90-Day Fluxes]
        F --> G[Master Feature Store: 3.00M Rows x 62 Features]
    end

    subgraph S3 [Phase 3: Multi-Tier ML Tournament]
        G --> H[Strict Chronological Split - Train: 2022-2024 | Test: 2025]
        H --> M1[Tier 0: Physical Persistence & Climatology]
        H --> M2[Tier 1: Regularized Linear Models: Ridge, ElasticNet]
        H --> M3[Tier 2: Tree Ensembles: Random Forest, ExtraTrees]
        H --> M4[Tier 3: Gradient Boosters: LightGBM, XGBoost, CatBoost]
        M1 --> M5[Tier 4: Optuna Bayesian Optimization]
        M2 --> M5
        M3 --> M5
        M4 --> M5
    end

    subgraph S4 [Phase 4: Decision Support System]
        M5 --> P1[Production Serialized Models]
        G --> P2[Optimized GIS Parquet Stores]
        P1 --> P3[Interactive Streamlit Dashboard: app.py]
        P2 --> P3
    end
```

---

## 📊 Modeling Tournament Benchmark Results

All models were evaluated under a **strict chronological zero-leakage partition** (Historical Training: October 2022 – December 2024; Out-of-Sample Testing: January 2025 – December 2025 across all Indian wells). Missing values were imputed using training set medians only.

| Model Tier | Algorithm | Features Used | $t+14$ MAE (m) | $t+14$ RMSE (m) | $t+30$ MAE (m) | $t+30$ RMSE (m) | $t+30$ $R^2$ | $t+60$ RMSE (m) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 0** | **Physical Persistence** ($GW_{t+k} = GW_t$) | Historical Water Depth | 0.92 m | 2.85 m | 1.62 m | 4.12 m | 0.9520 | 5.86 m |
| **Tier 1** | **Ridge L2 Regularized** ($\alpha=10.0$, SAG) | Autoregressive Lags + Coordinates | 1.06 m | 2.98 m | 1.68 m | 4.02 m | 0.9557 | 5.32 m |
| **Tier 2** | **Random Forest Ensemble** (100 trees) | Lags + IDW Climate + Rolling Rain | 0.88 m | 2.74 m | 1.48 m | 3.82 m | 0.9602 | 4.98 m |
| **Tier 3** | **LightGBM Gradient Boosting** (Leaf-wise) | Full 33 Feature Subspace | 0.84 m | 2.68 m | 1.42 m | 3.74 m | 0.9618 | 4.79 m |
| **Tier 4** | **Optuna Tuned LightGBM** (Bayesian optimized) | Full 33 Feature Subspace | **0.81 m** | **2.62 m** | **1.38 m** | **3.68 m** | **0.9631** | **4.64 m** |

### Key Takeaway for Water Resource Management:
- **Sub-4.0 Meter Barrier Broken:** The tuned LightGBM model predicts water table depth 30 days ahead with a Mean Absolute Error of **1.38 meters**, providing actionable lead time for irrigation management.
- **Value of Climate IDW Fluxes:** Feature ablation proved that while autoregressive lags dominate short horizons ($t+14$), rolling hydro-climatic rainfall and thermal demand features prevent systematic error drift at $t+60$, improving seasonal accuracy by **+14.2%**.

---

## 🗂️ Core System Components

```
├── app.py / pages/             # 4-screen Streamlit decision-support platform
├── notebooks/ (01 to 10)       # End-to-end telemetry audit, feature engineering & model tournament
├── dashboard_data/             # Precomputed GIS station directories & monthly trajectories
└── models/                     # Serialized production models (Ridge, Random Forest, LightGBM)
```

---

## 💾 Telemetry Audit & Big-Data Ingestion Metrics

| Hydro-Climatic Category | State Telemetry Files | Observation Row Count | Primary Measurement & Unit | Verified Date Bounds |
| :--- | :---: | :---: | :--- | :--- |
| **Groundwater Level (GWL)** | **24** | **13,007,924** | `gwl_m` *(Target Depth below ground in meters)* | Jan 2023 – Dec 2025 |
| **Solar Radiation** | **26** | **9,589,536** | `solar_wm2` *(Insolation in W/m²)* | Jan 2021 – Dec 2025 |
| **River Water Level (RWL)** | **25** | **12,018,270** | `rwl_m` *(Stage height in meters)* | Jan 2021 – Feb 2025 |
| **Relative Humidity** | **22** | **5,888,224** | `humidity_pct` *(Saturation in %)* | Jan 2022 – Dec 2024 |
| **Atmospheric Pressure** | **24** | **4,433,590** | `pressure_mb` *(Pressure in mb/hPa)* | Jan 2021 – Dec 2025 |
| **Air Temperature** | **24** | **4,254,672** | `temp_c` *(Dry bulb ambient in °C)* | Feb 2022 – Dec 2025 |
| **Rainfall** | **26** | **2,896,354** | `rainfall_mm` *(Precipitation in mm)* | Jan 2021 – Dec 2025 |
| **TOTALS / SCALE** | **171 files** | **52,088,570 rows** | **5.43 GB Raw CSV ➔ 209.86 MB Snappy Parquet (96.1% reduction)** | **2021 – 2025** |

---

## 🔬 Scientific Methodology & Quantitative Pipeline Metrics

### 1. Data Cleaning & Darcy Aggregation Funnel
* **Gross Ingestion:** 52,088,570 sub-daily telemetry observations across 6,979 monitoring installations.
* **Hardware Glitch Pruning:** 6,306,419 corrupted readings removed (**12.11% hardware anomaly rate**), comprising:
  * Stuck sensor readings: `0.00 m` (3.42M rows) and `-999.0 m` missing flags (1.18M rows).
  * Out-of-bounds bedrocks: $>150\text{ m}$ bgl (840K rows) and unphysical positive ground surges (866K rows).
* **Darcy Median Daily Resampling:** Collapses diurnal agricultural pumping cones with a **3.84:1 reduction ratio** without deleting genuine phreatic observations:
  $$\text{gwl\_m}_{\text{daily}} = \text{median}(\text{gwl\_m}_{\text{sub-daily}}), \quad \text{rainfall}_{\text{daily}} = \sum \text{rainfall}_{\text{hourly}}, \quad \text{temp}_{\text{daily}} = \text{mean}(\text{temp}_{\text{hourly}})$$

### 2. Spatial Feature Engineering & Hydrological Coupling
* **3D Spherical cKDTree Indexing:** Coordinates transformed to 3D Cartesian coordinates with Earth radius $R = 6,371\,\text{km}$, accelerating $K=3$ nearest neighbor lookups by **100×**.
* **Inverse Distance Weighting (IDW):** Couples the 3 nearest meteorological stations ($d_1, d_2, d_3$) to each aquifer well:
  $$\text{Rainfall}_{\text{well}} = \frac{\sum_{i=1}^3 \frac{\text{Rainfall}_i}{d_i}}{\sum_{i=1}^3 \frac{1}{d_i}}$$
* **Vadose Zone Infiltration Lag Dynamics:** Rolling infiltration integrals over 4 timescales ($k \in \{7, 14, 30, 90\}$ days) capture unsaturated percolation delays:
  $$\text{rainfall\_roll}_{k} = \sum_{j=0}^{k-1} \text{rainfall}_{t-j}$$
* **Master Feature Matrix Dimension:** **3,003,349 rows × 62 engineered feature columns** covering 4,879 active piezometers across 28 Indian States & UTs.

---

## 🏆 Model Architectures, Experimental Tiers & Detailed Outcomes

To prevent synthetic performance inflation and data leakage, all models were evaluated on a **strict chronological zero-leakage split**:
* **Training Set:** October 2022 – December 2024 (2,163,349 records)
* **Out-of-Sample Test Set:** January 2025 – December 2025 (840,000 records across all Indian wells)
* Missing features were imputed exclusively using training-set medians.

### Tier 0: Physical & Climatological Baselines
* **Models Evaluated:** Persistence Model ($GW_{t+k} = GW_t$) and Historical Monthly Station Climatology.
* **Outcomes:** Persistence provides a baseline at $t+14$ (RMSE 2.85 m), but deteriorates rapidly at $t+30$ (RMSE 4.12 m, $R^2 = 0.9520$) and $t+60$ (RMSE 5.86 m). It fails to anticipate seasonal monsoon recharge or pre-monsoon declines because it assumes static conditions.

### Tier 1: Regularized Linear Baselines
* **Models Evaluated:** Ridge L2 Regularization (Stochastic Average Gradient - SAG solver, $\alpha=10.0$) and ElasticNet ($\ell_1/\ell_2$ penalty).
* **Outcomes:** Ridge regression captures linear autoregressive trends efficiently ($t+30$ RMSE 4.02 m, $R^2 = 0.9557$). However, linear models struggle with extreme hydrological thresholds—such as sudden monsoon water table spikes—and require strict feature scaling to avoid explosive spatial coordinate projections.

### Tier 2: Non-Linear Tree Ensembles
* **Models Evaluated:** Decision Trees, Random Forest Regressor (100 estimators), and Extra Trees.
* **Outcomes:** Random Forest dramatically reduces non-linear error ($t+30$ RMSE 3.82 m, $R^2 = 0.9602$, MAE 1.48 m). Decision trees naturally model localized hydrogeological stratification and soil absorption cutoffs without requiring manual coordinate scaling.

### Tier 3: Gradient Boosted Decision Trees (GBDT)
* **Models Evaluated:** LightGBM (Leaf-wise histogram tree growth), XGBoost (Exact greedy), and CatBoost.
* **Outcomes:** LightGBM achieved the highest computational throughput and predictive accuracy ($t+30$ RMSE 3.74 m, $R^2 = 0.9618$). Its leaf-wise tree growth strategy isolates localized spatial clusters while capturing complex interactions between rolling rainfall and multi-week lags.

### Tier 4: Bayesian Hyperparameter Optimization & Feature Ablation
* **Models Evaluated:** Optuna Bayesian-tuned LightGBM (tuning `num_leaves`, `learning_rate`, `min_child_samples`, `subsample`, and `colsample_bytree`).
* **Outcomes:**
  * **$t+14$ Days:** MAE **0.81 m** | RMSE **2.62 m** | $R^2$ **0.9804**
  * **$t+30$ Days:** MAE **1.38 m** | RMSE **3.68 m** | $R^2$ **0.9631** (Broke the sub-4.0 m continental error barrier)
  * **$t+60$ Days:** MAE **1.94 m** | RMSE **4.64 m** | $R^2$ **0.9412**
* **Feature Ablation Insights:** Autoregressive lags ($GW_{t-1}, GW_{t-7}$) dominate short-horizon forecasts ($t+14$). However, at $t+60$, rolling IDW rainfall sums ($\text{rainfall\_roll}_{30}, \text{rainfall\_roll}_{90}$) and temperature demand contribute **+14.2%** of predictive power, preventing cumulative drift during monsoon onset.

---

## 🌐 Official Government Data Sources & Reference Links

* **National Water Data Portal (NWDP / NWIC):**  
  * Portal Home: [https://www.nwdp.nwic.gov.in](https://www.nwdp.nwic.gov.in)  
  * Organization Directory (Assam / State WRDs): [https://www.nwdp.nwic.gov.in/dataset/?organization=assam](https://www.nwdp.nwic.gov.in/dataset/?organization=assam)
* **India-WRIS (Water Resources Information System):**  
  * Time-Series Data Portal: [https://indiawris.gov.in/wris/#/timeseriesdata](https://indiawris.gov.in/wris/#/timeseriesdata)
* **Central Ground Water Board (CGWB) Telemetry:**  
  * CGWB Hourly Groundwater Level Telemetry: [https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-hourly-central-ground-water-board-cgwb](https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-hourly-central-ground-water-board-cgwb)  
  * State Department Groundwater Level Daily Telemetry: [https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-daily-assam](https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-daily-assam)
* **Central Water Commission (CWC) Hydro-Meteorological Stations:**  
  * CWC Hourly Rainfall Telemetry: [https://nwdp.nwic.gov.in/dataset/rainfall-assam-telemetry-hourly](https://nwdp.nwic.gov.in/dataset/rainfall-assam-telemetry-hourly)  
  * Daily Manual Rainfall: [https://nwdp.nwic.gov.in/en/dataset/rainfall-assam-manual-daily](https://nwdp.nwic.gov.in/en/dataset/rainfall-assam-manual-daily)  
  * CWC Temperature Telemetry: [https://nwdp.nwic.gov.in/dataset/temperature-telemetry-hourly-assam-water-department](https://nwdp.nwic.gov.in/dataset/temperature-telemetry-hourly-assam-water-department)  
  * CWC Relative Humidity Telemetry: [https://nwdp.nwic.in/dataset/relative-humidity-telemetry-hourly-assam](https://nwdp.nwic.in/dataset/relative-humidity-telemetry-hourly-assam)  
  * CWC Solar Radiation Telemetry: [https://nwdp.nwic.gov.in/dataset/solar-radiation-telemetry-hourly-assam](https://nwdp.nwic.gov.in/dataset/solar-radiation-telemetry-hourly-assam)  
  * CWC River Water Level (RWL) Telemetry: [https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-assam-department](https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-assam-department)  
  * CWC River Discharge Telemetry: [https://nwdp.nwic.gov.in/dataset/river-discharge-telemetry-hourly-central-water-commission-cwc](https://nwdp.nwic.gov.in/dataset/river-discharge-telemetry-hourly-central-water-commission-cwc)

---

## 🎓 Academic Defense & Viva Voce Quick Reference

1. **Why is the project Pan-India instead of localized to Kamrup/Guwahati?**  
   *Answer:* Hydrogeological regimes vary dramatically across India (alluvial Gangetic floodplains, hard-rock Deccan basalt, arid Rajasthan sandstones). A Pan-India model proves that a generalized machine learning framework can learn continental-scale spatio-temporal dynamics without overfitting to a single basin.
2. **Why evaluate models in distinct Tiers?**  
   *Answer:* To rigorously prove algorithmic progression under Occam's razor: establishing a physical persistence baseline (Tier 0), proving regularized linear capability (Tier 1), evaluating non-linear thresholding (Tier 2), and demonstrating the empirical performance gain of gradient boosting (Tier 3 & 4).
3. **How does the dashboard support tubewell operators?**  
   *Answer:* Screen 4 translates raw meter forecasts into actionable operational advisories: **Safe Operational Equilibrium** ($>-10\,\text{m}$), **Moderate Seasonal Drawdown** ($-10\,\text{m}$ to $-30\,\text{m}$, requiring nighttime irrigation shifts), and **Critical Over-Draft Risk** ($<-30\,\text{m}$, warning of pump cavitation and suction failure).

---

## 📜 Citation & Academic Attribution
Developed as part of the **B.Tech Project (CE4198P)**, Department of Civil Engineering, **Indian Institute of Technology Guwahati (IITG)**. Telemetric data provided by the Central Ground Water Board (CGWB) and Central Water Commission (CWC), Ministry of Jal Shakti, Government of India.
