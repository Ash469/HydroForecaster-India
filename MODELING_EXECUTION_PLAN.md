# 🌊 Pan-India Groundwater Level Forecasting Using Machine Learning & Hydro-Climatic Data
> **B.Tech Project (CE4198P) — Department of Civil Engineering, IIT Guwahati**  
> **Master Architecture, Modeling Blueprint & Pedagogical Implementation Plan**

---

## 1. Executive Project Summary & Context
* **Academic Setting:** 4th-Year Civil Engineering B.Tech Project at IIT Guwahati.
* **Supervisor Mandate:** A true **Pan-India machine learning framework** predicting depth to groundwater table below ground level (`gwl_m` / bgl) across India rather than a localized single-district model.
* **Dataset Scale:** 
  * Consolidated from **52,088,570 raw observation rows** across 171 government datasets (CGWB, CWC, NWDP, State WRDs).
  * Filtered **6,306,419 hardware sensor spikes** (12.11% corruption rate) and resampled to Daily Station Medians.
  * Cataloged **6,979 physical monitoring stations** across 31 States and Union Territories (4,719 GWL wells, 825 rain gauges, hundreds of weather sensors).
  * **Master Feature Store:** `dataset/feature_matrix/pan_india_gwl_feature_matrix.parquet` (**2,779,800 rows × 62 engineered columns**).

---

## 2. Multi-Horizon Forecasting Mandate
The framework must predict across three distinct operational timeframes:
1. **$t+14$ Days (2 Weeks Ahead):** Short-term operational management (localized irrigation rotation, tubewell regulation).
2. **$t+30$ Days (1 Month Ahead):** Medium-term planning window (district-level canal and reservoir release planning).
3. **$t+60$ Days (2 Months Ahead):** Seasonal strategic window (pre-monsoon vs. post-monsoon drought mitigation and crop selection).

---

## 3. Strict Zero-Leakage Temporal Validation Split
Due to strong temporal autocorrelation in hydrogeological systems, random K-Fold cross-validation causes severe data leakage. Every notebook must strictly partition data chronologically:
* **Historical Training Partition (68%):** October 30, 2022 to December 31, 2024 (~1,810,352 observations).
* **Held-Out Testing Partition (32%):** January 01, 2025 to December 31, 2025 (~836,027 observations).
* **Missing Value Imputation:** Imputed using **training partition medians only** (zero leakage into test data).

---

## 4. Standardized 8-Step Pedagogical Architecture for Every Tier Notebook
Every upcoming Tier Notebook must follow this exact 8-step structure so that live code, plots, comparisons, and defense notes are clearly presented:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Academic & Physical Formulation                                │
│   - Mathematical formulation of the specific Tier algorithms           │
│   - Civil engineering / hydrologic hypothesis being tested             │
├────────────────────────────────────────────────────────────────────────┤
│ STEP 2: Strict Temporal Train/Test Partitioning                        │
│   - Train: 2022-2024 (1.81M rows) | Test: 2025 (836K rows)             │
│   - Feature subspace isolation (which features this Tier accesses)     │
├────────────────────────────────────────────────────────────────────────┤
│ STEP 3: Systematic Model-by-Model Training (Live Logging)              │
│   - Real-time training progress, convergence curve, compute time       │
├────────────────────────────────────────────────────────────────────────┤
│ STEP 4: Comprehensive Multi-Horizon Evaluation Metrics                 │
│   - MAE (m), RMSE (m), R², NSE, and PBIAS across t+14, t+30, t+60      │
│   - Error reduction / degradation vs. previous Tier baselines          │
├────────────────────────────────────────────────────────────────────────┤
│ STEP 5: Residual Diagnostics & Error Distributions                     │
│   - Residual histogram & Q-Q plots (Normality check of errors)         │
│   - Heteroscedasticity check (Residuals vs. Actual Water Depth)        │
├────────────────────────────────────────────────────────────────────────┤
│ STEP 6: Real-World Hydrograph Case Studies                             │
│   - Actual vs. Predicted curves for 3 representative hydrogeologies:   │
│     * Deep Arid Aquifer (Rajasthan / Thar Basin)                       │
│     * Hard-rock Fractured Aquifer (Tamil Nadu / Deccan Peninsular)     │
│     * Shallow Alluvial Floodplain (Assam / Brahmaputra Basin)          │
├────────────────────────────────────────────────────────────────────────┤
│ STEP 7: Side-by-Side Model Comparison Tournament Table                 │
│   - Consolidated metrics table summarizing all models in this Tier     │
│   - Bar charts comparing MAE & RMSE across horizons                   │
├────────────────────────────────────────────────────────────────────────┤
│ STEP 8: Viva Voce & Technical Defense Cheatsheet                       │
│   - Detailed questions examiners will ask about this specific Tier     │
│   - Technical, domain-grounded defense answers                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Complete Notebook Roadmap (Notebooks 01 to 10)

### Phase 1 to 3: Completed Data Engineering & Spatial Feature Store
* ✅ **`notebooks/01_PanIndia_Spatial_EDA.ipynb`:** Spatial catalog of 6,979 stations across 31 States/UTs.
* ✅ **`notebooks/02_Full_dataInspection.ipynb`:** Deep audit of 52+ Million raw telemetry rows and 6.3M sensor glitches.
* ✅ **`notebooks/03_PanIndia_Data_Cleaning_Pipeline.ipynb`:** Sensor glitch truncation and daily station median aggregation.
* ✅ **`notebooks/04_PanIndia_Feature_Engineering.ipynb`:** Multi-station KD-Tree IDW rainfall & climate couplings.
* ✅ **`notebooks/05_PanIndia_Feature_Store_Report_and_ML_Blueprint.ipynb`:** Master Feature Store verification (62 features, vadose zone infiltration dynamics).

---

### Phase 4: Tier-by-Tier Machine Learning Modeling Roadmap

#### 📓 Notebook 06: `Tier_1_Physical_and_Linear_Baselines.ipynb`
* **Focus:** Physical zero-knowledge and regularized linear benchmarks across $t+14, t+30, t+60$.
* **Models:**
  1. **Persistence Benchmark:** $GW_{t+k} = GW_t$ (evaluates baseline aquifer inertia).
  2. **Ridge Regression (L2 penalty):** Shrinks correlated lag coefficients ($GW_{t-1} \dots GW_{t-90}$).
  3. **ElasticNet Regression (L1 + L2 penalty):** Automatic feature sparsity to show which climate features collapse to zero.
  4. **Polynomial / Spline Non-Linear Regression:** Tests if simple curvilinear terms capture vadose zone infiltration without decision trees.
* **Outputs:** Coefficient magnitude plots, multi-horizon error curve, actual vs. predicted hydrographs in Assam vs. Rajasthan, residual Q-Q plots.

#### 📓 Notebook 07: `Tier_2_Decision_Trees_and_Bagging_Ensembles.ipynb`
* **Focus:** Non-linear axis-aligned partitioning and variance reduction via bagging.
* **Models:**
  1. **Single Decision Tree Regressors:** Shallow (`max_depth=5`) vs. Deep (`max_depth=20`) to visualize underfitting vs. overfitting.
  2. **Random Forest Regressor (Bagging Ensemble):** Bootstrap feature sampling with Out-of-Bag (OOB) error tracking ($N=20 \rightarrow 100 \rightarrow 200$ trees).
  3. **ExtraTrees Regressor:** Extremely randomized thresholds to reduce variance on dense sensor data.
* **Outputs:** Depth vs. Overfitting curve, Feature Importance (MDI), hydrographs for $t+14, t+30, t+60$.

#### 📓 Notebook 08: `Tier_3_Advanced_Gradient_Boosting_Machines.ipynb`
* **Focus:** Sequential residual error minimization across localized hydrogeological regimes.
* **Models:**
  1. **XGBoost Regressor (`hist` method):** Second-order gradient boosting for multi-million row datasets.
  2. **LightGBM Regressor (Leaf-wise GBDT):** High-speed histogram binning and Exclusive Feature Bundling.
  3. **CatBoost Regressor:** Oblivious trees with categorical geographic handling (`state`, `basin`, `river`).
* **Outputs:** Loss reduction curves over boosting rounds, learning rate ablation, multi-horizon comparison, state-by-state residual error maps.

#### 📓 Notebook 09: `Tier_4_Hyperparameter_Optimization_and_Ablation.ipynb`
* **Focus:** Systematic tuning and feature ablation experiments.
* **Experiments:**
  1. **Systematic Hyperparameter Search:** Tuning tree depth, learning rate, `min_child_weight`, subsampling ratio.
  2. **Feature Ablation Studies:**
     * Run A: GW Lags Only.
     * Run B: GW Lags + Rainfall IDW.
     * Run C: GW Lags + Rainfall IDW + Temperature & Solar.
     * Run D: Full 62 Features (Spatial coordinates + Monsoon seasonality).
     * Quantifying the exact marginal metric gain of each hydro-climatic feature group.

#### 📓 Notebook 10: `Tier_5_Stacking_Ensemble_and_Hydrological_Synthesis.ipynb`
* **Focus:** Final production meta-model, Grand Tournament summary, and BTP defense conclusions.
* **Models & Synthesis:**
  1. **Weighted Averaging Blend:** Inverse-variance weighted combination of Ridge + Random Forest + XGBoost + LightGBM.
  2. **Stacking Meta-Regressor:** Out-of-fold meta-learner learning optimal regional model weights.
  3. **Grand Tournament Summary:** Final comparison table ranking all models from Tier 1 to Tier 5 across $t+14, t+30, t+60$.
  4. **SHAP Interpretability & Hydrological Validation:** Empirical proof of delayed vadose zone infiltration.

---

## 6. How Any Agent / Developer Should Resume Work
When starting a new session:
1. Open this document: `MODELING_EXECUTION_PLAN.md`.
2. Inspect the verified feature matrix at `dataset/feature_matrix/pan_india_gwl_feature_matrix.parquet`.
3. Verify that Notebooks 01 to 05 in `notebooks/` are intact and finalized.
4. Begin by implementing **Notebook 06 (`Tier_1_Physical_and_Linear_Baselines.ipynb`)** following the 8-Step Pedagogical Pattern for $t+14, t+30, t+60$ days.

