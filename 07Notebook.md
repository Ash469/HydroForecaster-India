# Tier 2 Benchmarks: Decision Trees & Bagging Ensembles
### **Department of Civil Engineering, IIT Guwahati | B.Tech Project (CE4198P)**
**Pan-India Groundwater Level Forecasting Using Machine Learning & Hydro-Climatic Data**

---

## 1. Executive Summary: The Non-Linear Breakthrough

In **Tier 1**, linear models (Ridge L2, ElasticNet, Polynomial) demonstrated that a single flat hyperplane cannot simultaneously model heterogeneous hydrogeology across India (Rajasthan desert vs. West Bengal alluvium vs. Tamil Nadu hard rock). Linear models hit an insurmountable error floor of **$4.1622\text{ m}$ RMSE** at the 30-day horizon.

**Tier 2 introduces Non-Linear Greedy Splitting and Bootstrap Aggregation (Bagging):**
* **Model 2.1: Single Decision Tree (CART):** Introduces non-linear thresholding, but suffers from high variance and overfitting ($4.5503\text{ m}$ RMSE).
* **Model 2.2: Random Forest (Bagging Ensemble):** De-correlates 50 trees with random feature subsampling ($m = \sqrt{p}$), slashing the 30-day RMSE down to **$4.0263\text{ m}$**—an immediate **$13.6\text{ cm}$ improvement over Tier 1**.
* **Model 2.3: ExtraTrees (Extremely Randomized Trees):** Tests randomized split thresholds ($4.1905\text{ m}$ RMSE), confirming that greedy CART thresholds are essential for hydrogeological cutoff boundaries.

---

## 2. Complete Tier 2 Multi-Horizon Benchmark Tournament Table

Evaluated against the exact held-out operational 2025 prospective test dataset ($712,340$ to $904,367$ unseen samples).

| Horizon | Model | Algorithm / Configuration | Features | MAE (m) | RMSE (m) | $R^2$ | NSE | PBIAS (%) | Train Time |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t+14$ Days (2 Weeks)** | **Model 2.1: Single Decision Tree** | CART (`max_depth=16`, `leaf=20`) | 24 Features | 1.1778 | 3.2655 | 0.9733 | 0.9733 | +0.45% | 21.88s |
| $t+14$ Days | **Model 2.2: Random Forest** | Bagging Ensemble ($N=50$, $\sqrt{p}$) | 24 Features | **1.0243** | **3.0194** | **0.9772** | **0.9772** | +0.52% | 16.45s |
| $t+14$ Days | **Model 2.3: ExtraTrees** | Extremely Randomized ($N=50$) | 24 Features | 1.2671 | 3.1971 | 0.9744 | 0.9744 | +0.68% | 6.40s |
| | | | | | | | | | |
| **$t+30$ Days (1 Month)** | **Model 2.1: Single Decision Tree** | CART (`max_depth=16`, `leaf=20`) | 24 Features | 1.8908 | 4.5503 | 0.9471 | 0.9471 | +1.12% | 21.31s |
| $t+30$ Days | **Model 2.2: Random Forest** | Bagging Ensemble ($N=50$, $\sqrt{p}$) | 24 Features | **1.6141** | **4.0263** | **0.9586** | **0.9586** | +1.28% | 15.95s |
| $t+30$ Days | **Model 2.3: ExtraTrees** | Extremely Randomized ($N=50$) | 24 Features | 1.8403 | 4.1905 | 0.9551 | 0.9551 | +1.55% | 7.06s |
| | | | | | | | | | |
| **$t+60$ Days (2 Months)** | **Model 2.1: Single Decision Tree** | CART (`max_depth=16`, `leaf=20`) | 24 Features | 2.8144 | 6.3002 | 0.8942 | 0.8942 | +2.67% | 20.78s |
| $t+60$ Days | **Model 2.2: Random Forest** | Bagging Ensemble ($N=50$, $\sqrt{p}$) | 24 Features | **2.3821** | **5.1919** | **0.9281** | **0.9281** | +3.15% | 17.35s |
| $t+60$ Days | **Model 2.3: ExtraTrees** | Extremely Randomized ($N=50$) | 24 Features | 2.5774 | 5.3325 | 0.9242 | 0.9242 | +3.33% | 7.16s |

---

## 3. Direct Head-to-Head Comparison: Tier 1 vs. Tier 2 Progression

The table below contrasts the best model of Tier 1 (Ridge L2) with the best model of Tier 2 (Random Forest) across all operational horizons:

```
                  TIER 1 (LINEAR) vs TIER 2 (BAGGING) PROGRESSION
┌──────────────────────┬──────────────────────┬──────────────────────┬───────────────────────┐
│ Metric & Horizon     │ Tier 1: Ridge L2     │ Tier 2: Random Forest│ Net Real Improvement  │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────────┤
│ t+14 Days MAE        │ 1.0980 m             │ 1.0243 m             │ 7.37 cm Slashed       │
│ t+14 Days RMSE       │ 3.0871 m             │ 3.0194 m             │ 6.77 cm Slashed       │
│ t+14 Days NSE / R²   │ 0.9761               │ 0.9772               │ +0.0011 Gain          │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────────┤
│ t+30 Days MAE        │ 1.7351 m             │ 1.6141 m             │ 12.10 cm Slashed      │
│ t+30 Days RMSE       │ 4.1622 m             │ 4.0263 m             │ 13.59 cm Slashed      │
│ t+30 Days NSE / R²   │ 0.9558               │ 0.9586               │ +0.0028 Gain          │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────────┤
│ t+60 Days MAE        │ 2.6476 m             │ 2.3821 m             │ 26.55 cm Slashed      │
│ t+60 Days RMSE       │ 5.4927 m             │ 5.1919 m             │ 30.08 cm Slashed      │
│ t+60 Days NSE / R²   │ 0.9196               │ 0.9281               │ +0.0085 Gain          │
└──────────────────────┴──────────────────────┴──────────────────────┴───────────────────────┘
```

---

## 4. Key Hydrological & Machine Learning Discoveries

### Discovery 1: Why Random Forest Shattered the Linear Ceiling
* **Hydrological Reason:** India has vastly different groundwater storage coefficients ($S$). In crystalline fractured rock (Tamil Nadu), a $50\text{ mm}$ rain event causes an immediate $2\text{ m}$ water table rise. In alluvial sands (Indo-Gangetic Basin), the same rain event is absorbed with barely a $0.2\text{ m}$ rise.
* **Algorithmic Reason:** A linear model is forced to assign one global coefficient to `rainfall_roll90_sum`. Random Forest creates conditional splits:
  $$\text{IF } \text{latitude} < 13.0 \text{ AND } \text{gwl\_lag\_1} < -25\text{m} \longrightarrow \text{Branch to Hard-Rock Recharge Node}$$
  $$\text{IF } \text{latitude} > 25.0 \text{ AND } \text{longitude} < 75.0 \longrightarrow \text{Branch to Deep Arid Aquifer Node}$$
  This localized specialization reduced national RMSE by **$13.6\text{ cm}$ at 30 days and $30.1\text{ cm}$ at 60 days**.

### Discovery 2: The Failure of the Single Decision Tree (CART)
* Single Tree achieved an RMSE of **$4.5503\text{ m}$ at $t+30$** (substantially worse than Ridge's $4.1622\text{ m}$).
* **Explanation:** A single greedy tree is a high-variance estimator. It makes deterministic threshold cuts on noisy 2024 training telemetry, creating "step-function" artifacts. When prospective 2025 weather deviates slightly, predictions jump across sharp step boundaries.

### Discovery 3: The Power of Bootstrap Aggregation (Bagging)
* Random Forest averages 50 de-correlated trees:
  $$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B=50} T_b(x)$$
* Because each tree is trained on a bootstrap sample of the data and selects split candidates from a random subset ($m = \sqrt{24} \approx 5$ features), the step-function noise cancels out, yielding a smooth, continuous hydrograph.

### Discovery 4: ExtraTrees vs. Random Forest
* ExtraTrees (randomized thresholds) trained faster ($7\text{s}$ vs $16\text{s}$), but achieved higher RMSE ($4.1905\text{ m}$ vs $4.0263\text{ m}$).
* **Explanation:** In hydrogeology, physical boundaries (e.g., ground level at $0\text{ m}$, bed rock at $-40\text{ m}$, monsoon rainfall threshold $>150\text{ mm}$) are sharp and meaningful. Choosing random split points blurs these real geological interfaces.

---

## 5. Why We Must Advance to Tier 3 (Gradient Boosting: LightGBM / XGBoost / CatBoost)

While Random Forest made a massive leap, it possesses one structural limitation:
1. **Independent Parallel Trees:** Each tree in Random Forest is built independently. A mistake made by Tree 1 is ignored by Tree 2.
2. **Residual Focusing in Tier 3:** Gradient Boosting trains trees **sequentially**, where each new tree directly targets and minimizes the pseudo-residuals (unexplained hydrogeological error) of the previous trees:
   $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x), \quad r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]$$
3. This sequential correction enables Tier 3 to learn hard-to-predict geographical anomalies (such as over-pumped deep aquifers in Punjab and flash recharge in Assam), driving the RMSE down toward the sub-$4.0\text{ m}$ mark (**$3.9796\text{ m}$ in Tier 4**).

---

## 6. Ready-to-Use Presentation / Thesis Defense Script

If asked during your thesis examination:
> *"What is the main takeaway from Tier 2, and how does it justify your project methodology?"*

**Examiner Defense Script:**
> *"Sir, Tier 2 provides empirical proof that groundwater forecasting across a continental scale cannot be treated as a linear process.  
> While a single decision tree overfits with a high RMSE of $4.55\text{ m}$, Random Forest bagging slashes the 30-day forecast RMSE to $4.026\text{ m}$—cutting error by $13.6\text{ cm}$ over Tier 1, and by over $30.1\text{ cm}$ at the 60-day horizon.  
> This improvement occurs because random feature subsampling and hierarchical splits allow the model to autonomously compartmentalize India's 31 States into localized hydrogeological regimes. However, because bagging trees operate in parallel without communicating their errors, we advance to Tier 3 Gradient Boosting to sequentially correct remaining residual spikes."*

