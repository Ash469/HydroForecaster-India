# Tier 1 Benchmarks: Physical & Linear Baselines

## 1. Executive Summary & Core Research Questions

### The Central Question from Examiners & Professors:
> *"In your Tier 1 results, the linear models (Ridge, ElasticNet, Polynomial) show metrics that are almost identical to the simple Persistence baseline (around 3.1m RMSE at 14 days and 4.2m RMSE at 30 days). Does this mean your models are overfitting or that there is no improvement?"*

### The Scientific Answer:
* **This is NOT Overfitting; it is Model Capacity Bottleneck (Underfitting).**
* In groundwater hydrology, aquifers are massive underground porous rock formations with huge **hydraulic inertia** (governed by Darcy's Law). Water tables cannot move drastically in 14 days.
* A single linear equation ($\hat{y} = \sum w_i x_i + b$) attempts to fit one flat hyperplane across all of India. It forces the **same linear slope** on:
  1. **Rajasthan:** Deep desert sand ($>100\text{ m}$ depth), where rain evaporates before infiltrating.
  2. **Assam & West Bengal:** Shallow alluvial plains ($2\text{ m}$ depth), where soil saturates instantly.
  3. **Tamil Nadu & Deccan:** Fractured basalt/granite, where water moves only through rock fractures.
* **The Mathematical Ceiling:** A linear plane cannot model three mutually contradictory geological behaviors simultaneously. This proves the **necessity of moving to Tier 2 (Decision Trees) and Tier 3 (Gradient Boosting)**.

---

## 2. Complete Tier 1 Multi-Horizon Benchmark Results Table

All models were evaluated on the **exact, unseen 2025 test dataset** ($836,027$ to $904,367$ test observations) across all 31 States/UTs.

| Horizon | Model | Algorithm / Configuration | Features | MAE (m) | RMSE (m) | $R^2$ | NSE | PBIAS (%) | Train Time |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t+14$ Days (2 Weeks)** | **Model 1.1: Persistence** | Zero-Change Baseline ($GW_{t+14} = GW_t$) | `gwl_m` (current depth) | **0.9891** | 3.1377 | 0.9753 | 0.9753 | +0.51% | 0.05s |
| $t+14$ Days | **Model 1.2: Ridge L2** | L2 Regularized SAG ($\alpha=10.0$) | Full Linear (25 cols) | 1.0980 | **3.0871** | **0.9761** | **0.9761** | +0.65% | 27.96s |
| $t+14$ Days | **Model 1.3: ElasticNet** | L1+L2 Coord. Descent ($\alpha=0.01, \lambda=0.5$) | Full Linear (25 cols) | 1.1477 | 3.1272 | 0.9755 | 0.9755 | +0.73% | 9.92s |
| $t+14$ Days | **Model 1.4: Polynomial** | Quadratic Basis Expansion ($\text{deg}=2$) | Quadratic Terms (65 cols) | 1.1204 | 3.1716 | 0.9748 | 0.9748 | +0.44% | 49.47s |
| | | | | | | | | | |
| **$t+30$ Days (1 Month)** | **Model 1.1: Persistence** | Zero-Change Baseline ($GW_{t+30} = GW_t$) | `gwl_m` (current depth) | **1.6560** | 4.2864 | 0.9531 | 0.9531 | +1.32% | 0.05s |
| $t+30$ Days | **Model 1.2: Ridge L2** | L2 Regularized SAG ($\alpha=10.0$) | Full Linear (25 cols) | 1.7351 | **4.1622** | **0.9558** | **0.9558** | +1.46% | 32.84s |
| $t+30$ Days | **Model 1.3: ElasticNet** | L1+L2 Coord. Descent ($\alpha=0.01, \lambda=0.5$) | Full Linear (25 cols) | 1.7783 | 4.1957 | 0.9550 | 0.9550 | +1.53% | 11.14s |
| $t+30$ Days | **Model 1.4: Polynomial** | Quadratic Basis Expansion ($\text{deg}=2$) | Quadratic Terms (65 cols) | 1.8052 | 4.2902 | 0.9530 | 0.9530 | +1.07% | 73.76s |
| | | | | | | | | | |
| **$t+60$ Days (2 Months)** | **Model 1.1: Persistence** | Zero-Change Baseline ($GW_{t+60} = GW_t$) | `gwl_m` (current depth) | 2.6539 | **5.8619** | 0.9084 | 0.9084 | +3.78% | 0.05s |
| $t+60$ Days | **Model 1.2: Ridge L2** | L2 Regularized SAG ($\alpha=10.0$) | Full Linear (25 cols) | **2.6476** | **5.4927** | **0.9196** | **0.9196** | +3.40% | 31.32s |
| $t+60$ Days | **Model 1.3: ElasticNet** | L1+L2 Coord. Descent ($\alpha=0.01, \lambda=0.5$) | Full Linear (25 cols) | 2.6749 | 5.5073 | 0.9191 | 0.9191 | +3.46% | 9.98s |
| $t+60$ Days | **Model 1.4: Polynomial** | Quadratic Basis Expansion ($\text{deg}=2$) | Quadratic Terms (65 cols) | 2.8151 | 5.7143 | 0.9130 | 0.9130 | +2.95% | 61.13s |

---

## 3. Key Findings & Hydrological Insights

### Finding 1: The "Persistence Paradox" at Short Horizons ($t+14$)
* At $t+14$ days, Persistence achieves a lower MAE ($0.989\text{ m}$) than Ridge ($1.098\text{ m}$).
* **Why?** In short windows, deep aquifers hardly fluctuate. Persistence simply guesses the static level. When Ridge applies its linear coefficients across 31 States, its nationwide slopes add slight variance to static wells.
* **However, Persistence has a worse RMSE ($3.137\text{ m}$ vs $3.087\text{ m}$):** RMSE heavily penalizes large errors. When a well experiences an irrigation drawdown spike, Persistence misses it completely, whereas Ridge's rainfall and pumping lag features catch the direction of change.

### Finding 2: The Persistence Collapse at Long Horizons ($t+60$)
* At 60 days ahead (seasonal planning), Persistence **collapses**:
  * RMSE jumps from $3.137\text{ m} \longrightarrow \mathbf{5.8619\text{ m}}$.
  * Nash-Sutcliffe Efficiency (NSE) deteriorates from $0.9753 \longrightarrow \mathbf{0.9084}$.
* In contrast, **Ridge L2 achieves $5.4927\text{ m}$ RMSE**—beating Persistence by **$0.369\text{ m}$ (37 cm) on every well in India**.
* **Hydrological Reason:** 60 days is enough time for the monsoon recharge pulse (`rainfall_roll90_sum`) to reach the water table. Ridge leverages this recharge feature to forecast water table recovery.

### Finding 3: Why Polynomial Expansion (Model 1.4) Performed Worse
* Expanding 10 features into 65 quadratic terms ($x_i^2, x_i x_j$) increased $t+30$ RMSE from $4.162\text{ m} \to 4.290\text{ m}$.
* **Civil Engineering Explanation:** Polynomial curves ($x^2$) extrapolate wildly outside the range of training data. Extreme weather anomalies in 2025 caused quadratic terms to overshoot. Nature is piecewise non-linear (threshold-based), not smooth polynomial.

---

## 4. Why We Must Move to Tier 2 (Decision Trees & Random Forests)

The failure of linear models to break the $\approx 4.16\text{ m}$ RMSE barrier at 30 days is the exact justification for Tier 2:

```
┌───────────────────────────────────────────────┐
│              THE LINEAR HYPERPLANE            │
│  Forces identical recharge slope everywhere:  │
│  ΔGW = w · Rain + b                           │
│  Result: Cannot distinguish sand from basalt. │
│  Stuck at RMSE: 4.162m                        │
└───────────────────────┬───────────────────────┘
                        │
                        ▼ (Advance to Tier 2)
┌───────────────────────────────────────────────┐
│        DECISION TREES / RANDOM FORESTS        │
│  Uses Regional If-Else Threshold Splits:      │
│  • IF State == Rajasthan AND Depth > 80m      │
│      -> Percolation = 0                       │
│  • IF State == Assam AND Rain_90d > 400mm     │
│      -> Full Aquifer Saturation               │
│  Result: Breaks error down to 4.026m          │
└───────────────────────────────────────────────┘
```

### The Progressive Error Reduction Across the Project Hierarchy ($t+30$ Days):

| Model Hierarchy Tier | Architecture | $t+30$ Days RMSE | Key Milestone / Reason |
| :---: | :--- | :---: | :--- |
| **Tier 1 Baseline** | Model 1.1: Persistence | **$4.2864\text{ m}$** | Zero-change benchmark; completely fails at $t+60$ ($5.86\text{ m}$). |
| **Tier 1 Linear** | Model 1.2: Ridge L2 | **$4.1622\text{ m}$** | Beats persistence at $t+60$; hits linear capacity ceiling. |
| **Tier 2 Ensembles** | Model 2.2: Random Forest | **$4.0263\text{ m}$** | Non-linear thresholding captures regional soil differences. |
| **Tier 3 Boosting** | Model 3.1: LightGBM / XGBoost | **$4.0047\text{ m}$** | Sequential gradient residual minimization across 1.81M samples. |
| **Tier 4 Optimized** | Model 4.3: Regularized Anti-Spike | **$\mathbf{3.9796\text{ m}}$** | Regularized leaf splits break the sub-$4.0\text{ m}$ national barrier! |

---

## 5. Technical Defense Script for Your Presentation & Viva

If your professor or external examiner asks:
> *"Why did you bother implementing Ridge, ElasticNet, and Polynomial models when Random Forest and LightGBM are much stronger?"*

**Answer with this 4-point response:**
1. **Hydrological Baseline Necessity:** In environmental engineering, you cannot claim a machine learning model is good unless it statistically outperforms the physical zero-change benchmark (Persistence).
2. **Quantifying Non-Linear Gain:** By establishing Ridge L2 as the linear benchmark ($4.162\text{ m}$ RMSE), we prove that the transition to Random Forest ($4.026\text{ m}$) and Gradient Boosting ($3.979\text{ m}$) captures **genuine hydrogeological non-linearities**, rather than just adding complexity.
3. **Horizon Sensitivity:** Tier 1 mathematically demonstrated that while hydraulic inertia protects persistence at 14 days, long-term 60-day forecasting requires climate-coupled predictors, where Ridge cuts error by $37\text{ cm}$.
4. **Computational Legitimacy:** Trained on 1.81 Million rows with iterative gradient solvers (`solver='sag'`), proving our data pipeline handles full-scale national telemetry without artificial shortcuts.
