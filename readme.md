# BTP PROJECT CONTEXT — PAN-INDIA GROUNDWATER LEVEL PREDICTION

## 1. Student / Academic Context

The project is a B.Tech Project (BTP), course code **CE4198P**, at **IIT Guwahati**, under the Civil Engineering department.

The student is a 4th-year Civil Engineering undergraduate and has a strong software-development background but comparatively less experience in advanced machine learning and hydrology.

The BTP is being developed during the placement season, so the project needs to be:

* technically meaningful,
* implementable within the available academic timeline,
* sufficiently substantial for a full-fledged BTP,
* suitable for demonstration to a faculty supervisor and examiners,
* and preferably supported by a working web-based system/dashboard.

The supervisor has specifically instructed that the project should be made **Pan-India**, rather than being restricted to a single district or state.

---

# 2. Final Project Direction

## Proposed Title

**Pan-India Groundwater Level Prediction Using Machine Learning and Hydro-Climatic Data**

Alternative acceptable title:

**Pan-India Groundwater Level Forecasting Using Machine Learning and Hydro-Climatic Variables**

The exact title can be refined later, but the core project must remain:

> A Pan-India machine-learning framework for predicting groundwater levels using historical groundwater observations and relevant hydro-climatic variables.

---

# 3. Original Project Idea

The initial project proposal was:

**“Groundwater Level Prediction Using Machine Learning and Climate Data.”**

The original motivation was that:

* groundwater is an important source of water for agriculture, domestic use, and other activities;
* groundwater levels vary significantly over time and space;
* groundwater depletion is an important water-resource concern;
* government agencies collect groundwater observations, but these observations can potentially be used more effectively for predictive modelling;
* machine learning can be used to learn relationships between historical groundwater behaviour and hydro-climatic variables.

The initial idea considered predicting groundwater levels for a particular district/location, potentially 1–3 months into the future.

However, after discussion with the supervisor, the project scope was expanded to **Pan-India**.

---

# 4. Important Supervisor Decision

The supervisor explicitly instructed that the project should be made **Pan-India** so that it becomes a full-fledged project.

The supervisor indicated that the major additional effort would be in:

* collecting data across India,
* organizing the data,
* handling spatial/temporal differences,
* and building a generalized framework.

The fundamental ML training process would remain conceptually similar.

Therefore:

**DO NOT design the project as:**

> Build a model for Kamrup/one district first → later expand to India.

The preferred approach is:

> Design the data pipeline and modelling framework as Pan-India from the beginning.

A particular state/district can still be used for debugging, exploratory analysis, or case studies, but the architecture and dataset design should support the complete Pan-India system.

---

# 5. Main Research Problem

The core problem is:

> How can historical groundwater-level observations and hydro-climatic variables be integrated into a scalable machine-learning framework capable of predicting future groundwater levels across different regions of India?

The project therefore has both:

### Data-engineering component

Build a consistent dataset from heterogeneous government datasets distributed across different:

* locations,
* stations/wells,
* temporal frequencies,
* observation periods,
* formats,
* and data-quality levels.

### Machine-learning component

Use the resulting dataset to learn groundwater-level dynamics and predict future groundwater levels.

### Application/deployment component

Provide predictions and historical trends through a web dashboard.

---

# 6. Main Objectives

The project should aim to:

1. Develop a **Pan-India groundwater-level prediction framework** using machine learning.

2. Collect and integrate groundwater-level observations with selected hydro-climatic and hydrological variables.

3. Perform:

   * data cleaning,
   * quality control,
   * missing-value handling,
   * spatial alignment,
   * temporal alignment,
   * and feature engineering.

4. Develop suitable machine-learning models for short-term groundwater-level forecasting.

5. Evaluate model performance across different geographical regions and groundwater monitoring locations.

6. Investigate which hydro-climatic variables and temporal features contribute to groundwater-level prediction.

7. Develop a web-based dashboard for:

   * viewing groundwater conditions,
   * historical groundwater trends,
   * hydro-climatic information,
   * and model predictions.

---

# 7. Selected Data Variables

At the current stage, ONLY the following eight dataset categories have been selected.

## 1. Groundwater Level

This is the **target variable**.

Possible sources include:

* Assam Water Department groundwater observations,
* Central Ground Water Board (CGWB) groundwater observations,
* and eventually equivalent official groundwater observations from other regions of India.

Groundwater level may be represented using the measurement convention provided by the source dataset. The exact target definition must be verified from metadata before modelling.

Groundwater history will also be used to construct temporal features such as:

* groundwater level at t-1,
* groundwater level at t-2,
* groundwater level at t-3,
* potentially longer lag values,
* rolling statistics.

---

## 2. Rainfall — Telemetry

Rainfall is expected to be one of the most important predictors.

Possible transformations include:

* daily rainfall,
* monthly rainfall,
* previous-month rainfall,
* 3-month cumulative rainfall,
* 6-month cumulative rainfall,
* rolling rainfall averages/sums,
* rainfall anomalies if justified.

The exact aggregation must depend on the final modelling frequency.

---

## 3. Rainfall — Manual

Manual rainfall observations are also being collected.

The reason for retaining both telemetry and manual rainfall datasets is to:

* increase spatial coverage,
* compare data sources,
* investigate consistency,
* and potentially improve rainfall availability.

Do not automatically merge them.

First determine:

* station overlap,
* temporal coverage,
* measurement differences,
* missingness,
* and data quality.

Then decide whether they should be:

* treated as separate sources,
* combined after quality control,
* or used selectively.

---

## 4. Temperature

Temperature may be aggregated from higher-frequency observations to the modelling frequency.

Potential features:

* mean temperature,
* maximum temperature,
* minimum temperature,
* lagged temperature,
* rolling temperature statistics.

Temperature is relevant because it can influence evapotranspiration and water demand.

---

## 5. Relative Humidity

Relative humidity is being collected as an atmospheric predictor.

Potential role:

* represent atmospheric moisture conditions,
* indirectly provide information related to evapotranspiration,
* potentially improve prediction when combined with temperature and solar radiation.

It should not automatically be assumed to be useful.

Feature importance/ablation analysis should determine whether it contributes meaningfully.

---

## 6. Solar Radiation

Solar radiation is being collected as another atmospheric/hydro-climatic variable.

Potential role:

* represent available energy,
* provide information related to evapotranspiration,
* complement temperature and humidity.

Again, it should be evaluated rather than automatically included in the final model.

---

## 7. River Water Level

River water level is being collected to investigate possible interactions between:

* surface water,
* river systems,
* and groundwater.

Its usefulness will depend strongly on spatial proximity and hydrological connectivity.

A river station should not simply be matched to every groundwater station in the same district.

Spatial relevance must be considered.

---

## 8. River Water Discharge

River discharge is another hydrological predictor.

Potential role:

* characterize surface-water conditions,
* investigate possible groundwater–surface-water interactions,
* potentially improve prediction near relevant river systems.

Again, spatial and temporal relevance must be established before using it as a model feature.

---

# 8. Variables NOT Currently Selected

Do NOT automatically add these variables unless explicitly requested later:

* Atmospheric pressure
* Wind speed
* Wind direction
* Snowfall
* Suspended sediment
* ENSO
* IOD
* PET
* Other climate indices

These may be considered in future extensions, but they are **not part of the current selected dataset scope**.

---

# 9. Known Data Sources

The project is primarily looking at official Indian government water/hydro-climatic datasets.

Important sources identified so far include:

### National Water Data Portal / NWDP

General Assam organization page:

https://www.nwdp.nwic.gov.in/dataset/?organization=assam

### India-WRIS

Time-series data portal:

https://indiawris.gov.in/wris/#/timeseriesdata

### Groundwater — Assam

Assam Department groundwater telemetry:

https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-daily-assam

### Groundwater — CGWB

Central Ground Water Board groundwater telemetry:

https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-hourly-central-ground-water-board-cgwb

### Rainfall — Telemetry

https://nwdp.nwic.gov.in/dataset/rainfall-assam-telemetry-hourly

### Rainfall — Manual

https://nwdp.nwic.gov.in/en/dataset/rainfall-assam-manual-daily

### Temperature

https://nwdp.nwic.gov.in/dataset/temperature-telemetry-hourly-assam-water-department

### Relative Humidity

https://nwdp.nwic.in/dataset/relative-humidity-telemetry-hourly-assam

### Solar Radiation

https://nwdp.nwic.gov.in/dataset/solar-radiation-telemetry-hourly-assam

### River Water Level

https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-assam-department

### River Water Discharge

https://nwdp.nwic.gov.in/dataset/river-discharge-telemetry-hourly-central-water-commission-cwc

IMPORTANT:

These URLs are starting points identified during the project. Before using any dataset in final analysis, verify:

* current availability,
* metadata,
* spatial coverage,
* temporal coverage,
* units,
* station/well identifiers,
* and data quality.

Do not assume that every dataset has nationwide coverage simply because the final project is Pan-India.

---

# 10. Most Important Technical Challenge: Data Integration

The biggest challenge in this project is expected to be **data engineering rather than simply model selection**.

Different datasets may have:

* hourly observations,
* daily observations,
* six-hourly observations,
* monthly observations,
* irregular groundwater measurements,
* different station locations,
* different station identifiers,
* different coordinate systems,
* missing observations,
* different units,
* different observation periods.

Therefore, the first major research/engineering task is to construct a reliable modelling dataset.

The general pipeline is:

**Raw Government Data**

↓

**Data Validation & Metadata Extraction**

↓

**Quality Control**

↓

**Spatial Matching**

↓

**Temporal Alignment**

↓

**Common Modelling Frequency**

↓

**Feature Engineering**

↓

**Master ML Dataset**

↓

**Model Training**

↓

**Forecasting**

↓

**API**

↓

**Web Dashboard**

---

# 11. Temporal Resolution

The final modelling frequency has NOT yet been permanently fixed.

Monthly modelling is currently the most natural candidate because:

* groundwater observations may not have consistent high-frequency measurements everywhere,
* hydro-climatic variables can be aggregated to monthly resolution,
* short-term forecasting can then be expressed as 1-, 2-, or 3-month-ahead prediction.

However:

**Do not artificially convert sparse groundwater observations into monthly data merely to satisfy the project proposal.**

First investigate:

* groundwater observation frequency,
* station history,
* missingness,
* and temporal coverage.

Then choose a modelling frequency supported by the actual data.

Potential modelling frequency:

**Monthly**

if the groundwater data support it.

---

# 12. Forecasting Horizon

The original project considered:

* 1-month ahead
* 2-month ahead
* 3-month ahead

forecasting.

The final horizon should depend on the temporal resolution and availability of groundwater observations.

If monthly modelling is adopted:

Input:

`t-6, t-5, ..., t-1`

↓

Predict:

`t+1`

or

`t+1, t+2, t+3`

The exact forecasting strategy still needs to be decided.

Possible approaches:

### Direct forecasting

Separate model for each horizon:

* Model 1 → t+1
* Model 2 → t+2
* Model 3 → t+3

### Recursive forecasting

Predict t+1, then use that prediction to predict t+2, etc.

### Multi-output forecasting

One model predicts:

`[t+1, t+2, t+3]`

The best approach should be selected after understanding the dataset and experimenting with baselines.

---

# 13. Feature Engineering

Potential groundwater temporal features:

```text
GW(t-1)
GW(t-2)
GW(t-3)
GW(t-6)
GW rolling mean
GW rolling std
GW trend
```

Rainfall features:

```text
Rainfall(t)
Rainfall(t-1)
Rainfall(t-2)
3-month cumulative rainfall
6-month cumulative rainfall
3-month rainfall mean
rainfall anomaly
```

Temperature:

```text
mean temperature
max temperature
min temperature
temperature lag
rolling temperature
```

Relative humidity:

```text
mean humidity
humidity lag
rolling humidity
```

Solar radiation:

```text
mean solar radiation
solar radiation lag
rolling solar radiation
```

River variables:

```text
river water level
river discharge
river level lag
river discharge lag
rolling statistics
```

Not every feature should automatically be used.

Feature selection should be driven by:

* domain relevance,
* missingness,
* correlation,
* feature importance,
* model performance,
* and leakage considerations.

---

# 14. Critical Data-Leakage Concern

The project must avoid temporal leakage.

For prediction at time `t+1`, only information available at or before time `t` should be used.

For example:

Valid:

```text
GW(t-1)
Rainfall(t)
Rainfall(t-1)
Temperature(t)
```

Potentially invalid:

```text
GW(t+1)
Rainfall(t+1)
Temperature(t+1)
```

unless the project explicitly assumes future weather observations are available.

This distinction is especially important when creating rolling features.

---

# 15. Spatial Matching

This is one of the most important aspects of the Pan-India project.

A groundwater well/station cannot simply be matched to the nearest climate station without considering:

* geographic distance,
* terrain,
* hydrological conditions,
* station density,
* river proximity,
* and potentially aquifer characteristics.

A basic initial strategy can be:

**Groundwater station**

→ find nearby rainfall/climate stations

→ calculate distance

→ assign observations using a defined spatial threshold or nearest-neighbour method.

However, the matching methodology should be documented clearly and tested.

For river variables, proximity alone may not be sufficient.

A nearby river station may still be hydrologically unrelated to a groundwater well.

---

# 16. Pan-India Modelling Strategy

One of the major design questions is how to build a model that works across India.

The project should NOT assume that groundwater behaviour is identical everywhere.

Potential strategies include:

### Strategy A — Global Pan-India Model

Train one model using data from multiple regions.

Include location information such as:

```text
latitude
longitude
state
district
possibly elevation
possibly aquifer information
```

Advantages:

* one unified framework,
* scalable,
* easier deployment,
* can learn patterns across regions.

### Strategy B — Region/State-Specific Models

Train separate models for different regions.

Advantages:

* can capture local behaviour.

Disadvantages:

* less data per model,
* difficult to maintain,
* difficult for deployment at national scale.

### Strategy C — Global Model + Regional Models

Build a global baseline first and investigate whether regional models provide improvements.

This may become a useful experimental component.

The final strategy should be decided based on:

* data availability,
* station count,
* geographical heterogeneity,
* model performance,
* and computational cost.

---

# 17. Location Features

For a Pan-India model, location information is potentially important.

Possible features:

```text
latitude
longitude
state
district
elevation
well depth
aquifer type
```

However, only variables actually available and reliable across the dataset should be used.

Categorical variables such as state/district may require:

* one-hot encoding,
* target encoding,
* embeddings,
* or another appropriate representation.

Latitude/longitude can provide continuous spatial information.

---

# 18. Candidate Machine Learning Models

The model selection is NOT finalized.

The project can begin with simple baselines and progressively evaluate stronger models.

### Baseline models

Examples:

* persistence model
* historical mean
* linear regression

A persistence baseline is particularly important:

> Predict the next groundwater level as the current groundwater level.

This provides a realistic benchmark for whether ML actually provides improvement.

### Tree-based models

Potential candidates:

* Random Forest
* XGBoost
* LightGBM
* Gradient Boosting

These are likely to be strong candidates because the dataset will consist largely of structured/tabular features.

### Deep learning

Potential candidates:

* LSTM
* GRU
* Temporal CNN
* Transformer-based time-series models

However, deep learning should NOT be used merely to make the project look more advanced.

First establish strong tabular baselines.

If the dataset supports it, an LSTM/GRU experiment can be added as a comparison.

---

# 19. Recommended Experimental Progression

A sensible experimental sequence is:

### Experiment 1

Persistence baseline.

### Experiment 2

Historical groundwater-only model.

Example:

```text
GW(t-1)
GW(t-2)
GW(t-3)
...
```

### Experiment 3

Groundwater + rainfall.

### Experiment 4

Groundwater + all selected hydro-climatic variables.

### Experiment 5

Tree-based models:

```text
Random Forest
XGBoost / LightGBM
```

### Experiment 6

Optional deep-learning model:

```text
LSTM / GRU
```

### Experiment 7

Compare:

* global model,
* regional/state models if data permits.

This creates a clear research narrative rather than simply training one model.

---

# 20. Evaluation Metrics

Potential regression metrics:

### MAE

Mean Absolute Error.

Useful because it gives the average absolute prediction error in groundwater-level units.

### RMSE

Root Mean Squared Error.

Penalizes larger errors more heavily.

### R²

Coefficient of determination.

Useful for understanding explained variance but should not be used alone.

Potentially also:

* MAPE, if mathematically appropriate for the target representation;
* normalized RMSE;
* station-wise error;
* region-wise error.

The evaluation should be performed using **time-based splits**, not random train/test splitting.

---

# 21. Train/Validation/Test Strategy

Because this is a forecasting problem, random splitting is generally inappropriate.

A possible structure is:

```text
Historical period
        |
        +---- Training
        |
        +---- Validation
        |
        +---- Test
```

For example:

```text
2018–2023 → Training
2024 → Validation
2025 → Test
```

The actual years must depend on the final dataset.

The key principle:

> The model must be evaluated on future observations relative to the training period.

Potentially use:

* expanding-window validation,
* rolling-origin evaluation,
* walk-forward validation.

---

# 22. Missing Data

Missing observations are expected to be a major issue.

Potential strategies include:

### Small gaps

Interpolation may be considered.

### Large gaps

Potentially:

* leave missing,
* exclude the station,
* or use a model-specific missing-data strategy.

Do NOT blindly interpolate long periods of missing groundwater data.

For each station, calculate:

```text
total observations
missing percentage
continuous history length
maximum missing gap
```

Then define station inclusion criteria.

For example:

> Only stations with sufficient temporal coverage and acceptable missingness are included in the primary modelling dataset.

The exact threshold should be decided after inspecting the actual data.

---

# 23. Data Quality Control

The project should explicitly include QC.

Potential checks:

* impossible values,
* duplicate timestamps,
* duplicate stations,
* inconsistent units,
* sudden unrealistic jumps,
* timestamp errors,
* station relocation,
* missing coordinates,
* inconsistent station IDs,
* sensor failures.

Outlier detection should distinguish between:

* genuine hydrological events,
* and measurement errors.

Do not automatically delete extreme groundwater values.

---

# 24. Master Dataset Concept

Eventually the raw datasets should be transformed into a structured modelling table.

Conceptually:

```text
station_id
latitude
longitude
state
district
date

groundwater_level

rainfall
rainfall_lag_1
rainfall_3m_sum
rainfall_6m_sum

temperature_mean
temperature_max
temperature_min

humidity_mean

solar_radiation_mean

river_level
river_discharge

gw_lag_1
gw_lag_2
gw_lag_3
gw_lag_6

target_gw_t_plus_1
target_gw_t_plus_2
target_gw_t_plus_3
```

The exact columns will depend on the final modelling frequency and available data.

---

# 25. Software Architecture

The intended final system can be structured as:

```text
                GOVERNMENT DATA SOURCES
                         |
                         v
                DATA COLLECTION LAYER
                         |
                         v
              DATA CLEANING / QC LAYER
                         |
                         v
            SPATIAL + TEMPORAL ALIGNMENT
                         |
                         v
                FEATURE ENGINEERING
                         |
                         v
                 MASTER DATASET
                         |
                         v
                 ML MODEL PIPELINE
                         |
                         v
                 TRAINED MODEL
                         |
                         v
                    BACKEND API
                         |
                         v
                  WEB DASHBOARD
```

---

# 26. Backend

The student's existing technical background includes:

* Node.js
* Express
* MongoDB
* React
* Next.js
* Flutter
* AWS
* Docker
* Nginx
* PM2

Therefore, a possible architecture is:

```text
React / Next.js Dashboard
          |
          v
      REST API
          |
          v
Python ML Service
          |
          v
Trained ML Model
          |
          v
Data Storage
```

Python is preferred for:

* data processing,
* scikit-learn,
* XGBoost/LightGBM,
* PyTorch/TensorFlow if deep learning is used.

Node.js can optionally serve as the main application backend/API gateway.

However, architecture should remain as simple as possible for the BTP.

---

# 27. Dashboard Concept

The dashboard should demonstrate the practical application of the project.

Possible user flow:

```text
India Map
   ↓
Select State
   ↓
Select District
   ↓
Select Groundwater Station/Well
   ↓
View Current Groundwater Level
   ↓
View Historical Trend
   ↓
View Rainfall / Climate Context
   ↓
View ML Forecast
```

Potential dashboard components:

### India map

Display:

* groundwater monitoring locations,
* potentially groundwater-level status.

### Location page

Show:

```text
Station Name
District
State
Coordinates
Current Groundwater Level
```

### Historical chart

Example:

```text
Groundwater level vs Time
```

### Hydro-climatic charts

Examples:

```text
Rainfall vs Time
Temperature vs Time
River Level vs Time
```

### Forecast chart

Example:

```text
Historical GW -----> Forecast
                       |
                       + t+1
                       + t+2
                       + t+3
```

Include confidence/uncertainty intervals only if a defensible method is implemented.

---

# 28. Dashboard Is Not the Research Contribution by Itself

The dashboard is primarily the **application/deployment layer**.

The core technical/research contribution should be:

1. Pan-India data integration.
2. Spatial-temporal alignment.
3. Feature engineering.
4. Groundwater forecasting.
5. Model evaluation across heterogeneous regions.
6. Analysis of hydro-climatic predictors.
7. A scalable prediction framework.

The dashboard demonstrates the usefulness of the resulting system.

---

# 29. Important Research Questions

Potential research questions include:

### RQ1

How accurately can groundwater levels be predicted using historical groundwater observations alone?

### RQ2

How much does incorporating rainfall and other hydro-climatic variables improve prediction performance?

### RQ3

Which hydro-climatic variables contribute most to groundwater-level prediction?

### RQ4

How does model performance vary across different geographical regions of India?

### RQ5

Can a generalized Pan-India model perform adequately across heterogeneous groundwater regimes?

### RQ6

How does a global model compare with region/state-specific models where sufficient data are available?

These questions can later be refined into formal BTP research objectives.

---

# 30. Potential Research Contribution

The project should NOT be presented as merely:

> “We trained XGBoost to predict groundwater.”

A stronger framing is:

> “A scalable Pan-India framework was developed for integrating heterogeneous groundwater and hydro-climatic observations and performing short-term groundwater-level forecasting across geographically diverse monitoring locations.”

The contribution consists of the combination of:

**Pan-India data engineering + spatiotemporal feature construction + ML forecasting + regional evaluation + deployment.**

---

# 31. Current Project Status

The project is currently in the **early data-collection and project-definition stage**.

The supervisor has approved/required the Pan-India direction.

The immediate priority is:

### Phase 1 — Data acquisition

Collect the selected datasets.

### Phase 2 — Understand metadata

For every dataset determine:

```text
What?
Where?
When?
How frequently?
Which units?
Which station ID?
Which coordinates?
How much missing data?
```

### Phase 3 — Build data inventory

Create a table:

```text
Dataset
Source
State/Region
Station count
Start date
End date
Frequency
Variables
Units
Missingness
Format
```

### Phase 4 — Construct initial Pan-India master dataset

### Phase 5 — EDA

### Phase 6 — Baseline ML

### Phase 7 — Advanced models

### Phase 8 — Evaluation

### Phase 9 — Backend

### Phase 10 — Dashboard

### Phase 11 — Report / poster / presentation

---

# 32. Immediate Priority

DO NOT start by building the dashboard.

DO NOT start with LSTM.

DO NOT spend excessive time tuning models before understanding the data.

The immediate priority should be:

> **Understand and build the Pan-India dataset.**

The project will succeed or fail largely based on whether a reliable modelling dataset can be constructed.

Start with one or a few datasets to understand the structure, but the final architecture must remain Pan-India.

---

# 33. Important Questions That Are Still Open

The following decisions are intentionally NOT finalized.

### Data questions

* What exact groundwater datasets will provide nationwide coverage?
* What is the actual temporal resolution of groundwater observations?
* How many stations/wells are available?
* Which stations have sufficient historical records?
* What years should be used?
* What is the spatial density?
* How should different groundwater sources be reconciled?

### Temporal questions

* Monthly or another modelling frequency?
* 1-, 2-, and 3-month forecasting?
* Direct vs recursive vs multi-output forecasting?

### Spatial questions

* Nearest climate station?
* Radius-based matching?
* Interpolation?
* Gridded climate data?
* How should river stations be spatially associated with groundwater stations?

### Model questions

* Random Forest vs XGBoost vs LightGBM?
* Is LSTM/GRU justified?
* Global model vs regional models?
* How should location be represented?

### Evaluation questions

* What time period should be used for testing?
* What station-level metrics should be reported?
* How should regional performance be compared?

These should be resolved using the actual data rather than assumed in advance.

---

# 34. Rules for Any AI Assistant Helping With This Project

When assisting with this BTP:

1. Treat the project as **Pan-India from the beginning**.

2. Do not revert to a single-district project unless explicitly requested.

3. Do not introduce additional variables such as pressure, wind, ENSO, IOD, PET, etc. unless explicitly requested.

4. Do not assume data availability. Verify dataset coverage and metadata.

5. Do not fabricate station counts, temporal coverage, or dataset statistics.

6. Distinguish clearly between:

   * confirmed project decisions,
   * proposed approaches,
   * experimental ideas,
   * and assumptions.

7. Prioritize data quality and data engineering before sophisticated ML.

8. Avoid data leakage.

9. Use time-based validation for forecasting.

10. Establish simple baselines before complex models.

11. Do not recommend deep learning merely because it sounds more advanced.

12. Consider spatial heterogeneity when designing the Pan-India model.

13. Consider groundwater-station-specific history and hydrogeological differences.

14. Do not blindly interpolate missing groundwater observations.

15. For river variables, consider actual hydrological relevance rather than only geographical proximity.

16. Keep the final system practical enough to complete as a BTP.

17. When suggesting architecture, prefer simple, maintainable components over unnecessary microservices.

18. When writing academic material, use technically defensible claims rather than exaggerated claims such as “AI will solve groundwater depletion.”

19. When discussing current government datasets, verify information from official sources before presenting it as fact.

20. When suggesting ML experiments, explain:

* what question the experiment answers,
* what data it needs,
* what baseline it should be compared against,
* and what metric should be used.

---

# 35. Preferred Working Style

The student prefers explanations that are:

* practical,
* technically precise,
* step-by-step,
* implementation-oriented,
* and directly connected to the actual BTP.

When discussing ML, explain the reasoning behind the modelling choices rather than simply giving code.

When providing code:

* explain the expected input schema,
* explain the output,
* make it reproducible,
* include validation/error handling where appropriate,
* and keep the implementation compatible with the eventual Pan-India pipeline.

When discussing research decisions, explicitly distinguish:

**Confirmed decision**

from

**Recommendation**

from

**Experiment to test.**

---

# 36. One-Line Project Summary

> **A Pan-India machine-learning framework that integrates groundwater observations with hydro-climatic and hydrological data to forecast future groundwater levels across diverse monitoring locations, supported by a web-based visualization and prediction dashboard.**

---

# 37. High-Level Final Architecture

```text
                    ┌──────────────────────────┐
                    │  Government Data Sources │
                    │                          │
                    │ Groundwater               │
                    │ Rainfall                 │
                    │ Temperature              │
                    │ Humidity                 │
                    │ Solar Radiation          │
                    │ River Level              │
                    │ River Discharge          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Data Acquisition          │
                    │ & Metadata Processing     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Data Quality Control      │
                    │ Missing Values            │
                    │ Outliers                  │
                    │ Units                     │
                    │ Duplicate Records         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Spatial + Temporal        │
                    │ Alignment                 │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Feature Engineering       │
                    │                           │
                    │ GW Lags                   │
                    │ Rainfall Aggregates       │
                    │ Climate Features          │
                    │ Hydrological Features     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Master Pan-India Dataset  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │         ML Model Experiments        │
              │                                     │
              │ Persistence → Linear → RF/XGB       │
              │                    ↓                │
              │                 LSTM/GRU             │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Time-Based Evaluation     │
                    │ MAE / RMSE / R²           │
                    │ Station-wise              │
                    │ Region-wise               │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Trained Prediction Model  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Backend / Prediction API  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Web Dashboard             │
                    │                          │
                    │ India → State → District │
                    │ → Station → Forecast     │
                    └──────────────────────────┘
```

---

# 38. Phase 1 Completed — Data Engineering & Feature Store Metrics

> **Status: 100% Completed and Verified (2026-09-30)**

### Verified Dataset Metrics

* **Total Observation Rows Across India:** **52,088,570 rows** (52+ Million Data Points)
* **Total State Dataset Files:** **171 files** across 7 hydro-climatic categories
* **Raw CSV Storage Footprint:** **5,427.91 MB (5.43 GB)**
* **Compressed Parquet Feature Store Size:** **209.86 MB**
* **Overall Storage Size Reduction:** **96.1% (Lossless Snappy Compression)**
* **Pandas Load Speedup:** **10× to 50× faster loading** (from 10 seconds down to 0.1 seconds per file)

### Category Breakdown Table

| Category | State Files | Total Row Count | Primary Measurement Column | Date Bounds |
| :--- | :---: | :---: | :--- | :--- |
| **Groundwater Level (GWL)** | **24** | **13,007,924** | `gwl_m` *(Target Variable in meters)* | Jan 2023 – Dec 2025 |
| **Rainfall** | **26** | **2,896,354** | `rainfall_mm` *(Hourly rainfall in mm)* | Jan 2021 – Dec 2025 |
| **Temperature** | **24** | **4,254,672** | `temp_c` *(Air temperature in °C)* | Feb 2022 – Dec 2025 |
| **Relative Humidity** | **22** | **5,888,224** | `humidity_pct` *(Relative humidity in %)* | Jan 2022 – Dec 2024 |
| **Solar Radiation** | **26** | **9,589,536** | `solar_wm2` *(Solar radiation in W/m²)* | Jan 2021 – Dec 2025 |
| **River Water Level (RWL)** | **25** | **12,018,270** | `rwl_m` *(River level in meters)* | Jan 2021 – Feb 2025 |
| **Atmospheric Pressure** | **24** | **4,433,590** | `pressure_mb` *(Atmospheric pressure in mb/hPa)* | Jan 2021 – Dec 2025 |
| **PAN-INDIA TOTAL** | **171** | **52,088,570** | **—** | **2021 – 2025** |

### Standardized Schema Layout

`station_name` | `agency` | `state_lgd_code` | `state` | `district_lgd_code` | `district` | `basin` | `river` | `latitude` | `longitude` | `elevation_msl` | `timestamp` | `<measurement_col>`

### Automation Pipelines Built

* `unify_clean_csvs.py`: Schema unification, datetime standardization (`YYYY-MM-DD HH:MM:SS`), missing string normalization, and metadata cleaning.
* `convert_to_parquet.py`: Binary downcasting (`float32`, `datetime64[us]`) and Snappy compression.
* `verify_parquet.py`: Automated multi-file dataset auditor and integrity verification reporter.
* `build_station_catalog.py`: Pan-India station inventory index extractor (`stations_master.parquet` & `stations_master.csv`).

---

# 39. Phase 2 — Master Station Catalog & Spatial Inventory (Step 2.1 Completed)

> **Status: 100% Completed (2026-09-30)**

### Physical Sensor Locations vs. ML Training Observations

* **Total Physical Stations Cataloged:** **6,979 stations** across India (averaging 6 to 8 wells per district).
* **Total Time-Series Training Samples:** **52,088,570 observation rows** across all categories.

| Hydro-Climatic Category | Physical Stations in India | Avg Readings / Station | Total ML Time-Series Samples |
| :--- | :---: | :---: | :---: |
| **Groundwater Level (GWL)** | **4,719 wells** | 2,756 readings/well | **13,007,924 rows** |
| **Rainfall** | **825 rain gauges** | 3,510 readings/station | **2,896,354 rows** |
| **Temperature** | **199 weather stations** | 21,380 readings/station | **4,254,672 rows** |
| **Relative Humidity** | **239 sensors** | 24,636 readings/station | **5,888,224 rows** |
| **Solar Radiation** | **405 sensors** | 23,677 readings/station | **9,589,536 rows** |
| **River Water Level (RWL)** | **336 river stations** | 35,768 readings/station | **12,018,270 rows** |
| **Atmospheric Pressure** | **256 sensors** | 17,318 readings/station | **4,433,590 rows** |
| **PAN-INDIA TOTAL** | **6,979 Stations** | **—** | **52,088,570 ML Rows** |

### Output Master Catalog Files

* `dataset/stations_master.csv`
* `dataset/stations_master.parquet`

---

# END OF PROJECT CONTEXT


