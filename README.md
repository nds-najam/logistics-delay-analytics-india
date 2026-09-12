# DataQ -- Logistics Delay Intelligence

A client-facing proof-of-concept analytics + ML application built for a logistics
company whose core business pain point is **delayed packages**. The application
ingests shipment data, detects and quantifies delays, explains *why* they happen,
predicts *which* shipments are at risk, explains *why* a given shipment is at risk,
and recommends *what operations should do about it* -- including interactive
what-if scenario testing.

> Built entirely on **synthetic** Indian logistics data with realistic, designed-in
> causal relationships (warehouse congestion, traffic, weather/monsoon, driver
> availability, handoffs, documentation delays, strikes, COD, address quality, etc.).
> Runs **100% locally** -- no cloud services, no external APIs.

---

## 1. Project Objective

> "Identify why packages are delayed, quantify the major causes, predict which
> packages are at risk of delay, and provide actionable recommendations to
> reduce delays."

The application tells one continuous story:

```
DATA -> DATA QUALITY -> DELAY DETECTION -> ROOT CAUSE ANALYSIS ->
DELAY PREDICTION -> EXPLANATION -> RECOMMENDATION -> WHAT-IF ANALYSIS -> OPERATIONAL ACTION
```

## 2. Architecture

```
logistics_delay_analytics/
├── app.py                     # Streamlit application (all pages)
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_logistics_data.csv
├── src/
│   ├── data_generator.py      # Synthetic data generation + quality-issue injection
│   ├── data_processing.py     # Loading, quality assessment, cleaning
│   ├── delay_analysis.py      # KPIs, root-cause decomposition, insights, alerts
│   ├── model.py                # Feature engineering, training, evaluation, persistence
│   ├── explainability.py      # SHAP / fallback per-shipment & global explanations
│   └── recommendations.py     # Rule-based recommendation engine
├── models/
│   └── delay_risk_model.joblib
└── tests/
    └── test_pipeline.py
```

Each module is independently testable and importable; `app.py` is a thin
presentation layer over `src/`.

## 3. Dataset Description

~120,000 synthetic Indian logistics shipment records (before cleaning; ~120,000
after removing injected duplicates) across 47 columns, including:

- **Identifiers**: Shipment_ID, Order_ID, Customer_ID
- **Geography**: Origin/Destination City & Region (24 Indian cities across 6
  regions), Origin/Destination Warehouse (12 warehouses)
- **Shipment attributes**: Shipping_Mode, Carrier, Service_Type, Package_Type,
  Package_Weight, Package_Volume, Customer_Priority
- **Timeline**: Order_Date, Pickup_Date, Expected_Delivery_Date, Actual_Delivery_Date,
  Promised_Delivery_Days, Actual_Delivery_Days
- **Component times**: Warehouse_Processing_Time_Hours, Pickup_Delay_Hours,
  Transit_Time_Hours, Sorting_Time_Hours, Last_Mile_Time_Hours
- **Operational conditions**: Number_of_Handoffs, Weather_Condition, Traffic_Level,
  Vehicle/Driver_Availability, Warehouse_Capacity_Utilization, EWay_Bill_Delay_Hours,
  Strike_Disruption_Flag, COD_Flag, Monsoon_Disruption_Flag, Address_Quality
- **Flags**: Holiday_Flag, Weekend_Flag, Peak_Season_Flag
- **Target**: Delay_Flag, Delay_Hours, Delay_Category, Delay_Reason

## 4. How Synthetic Data Is Generated

`src/data_generator.py` builds each shipment from realistic building blocks rather
than pure randomness:

1. **Geography & distance**: 24 real Indian cities with lat/lon; distance computed
   via haversine formula x a mode-specific route factor.
2. **Component time generation**: each of the 5 stage-time columns (warehouse
   processing, pickup, transit, sorting, last-mile) is built as a *baseline
   variability* term (gamma-distributed "generic" noise) **plus named causal
   excess terms**, e.g.:
   - Warehouse capacity utilization > 80% -> added warehouse processing hours
   - Heavy traffic / severe weather / monsoon -> added transit & last-mile hours
   - Low driver/vehicle availability -> added pickup / last-mile hours
   - Poor address quality -> added last-mile hours
   - COD -> added last-mile hours (repeated delivery/cash-confirmation attempts)
   - E-way bill/documentation delay -> added warehouse processing hours
   - Regional strike/bandh windows -> added transit hours
   - More handoffs -> added sorting hours
   - Peak season (Oct-Dec) -> added warehouse processing hours & higher traffic
   - A carrier-specific reliability multiplier scales excess (not baseline) hours,
     so one carrier ("Carrier B") is deliberately weaker.
3. **Delay determination**: `Promised_Delivery_Days` is derived independently from
   mode/distance/service-type (as if quoted at booking time); `Delay_Flag` /
   `Delay_Hours` are the actual difference between `Actual_Delivery_Date` and
   `Expected_Delivery_Date`.
4. **Delay_Category attribution**: for each delayed shipment, the generator tracks
   the magnitude of all 14 named causal "excess" terms and assigns `Delay_Category`
   to whichever term was largest for that shipment (`argmax`) -- so root-cause
   labels are internally consistent with what actually drove the delay.
5. **Controlled noise**: every causal effect is a random draw from a distribution
   (not a fixed deterministic add), so the resulting ~30% delay rate is realistic
   and the ML model cannot trivially memorize rules.
6. **Data-quality issues** (`inject_data_quality_issues`): missing values (~1-2%
   in 8 columns), duplicate rows (~0.7%), outliers (extreme weight/distance/delay
   values), and inconsistent categorical labels (case/whitespace/alias variants
   of city and carrier names) are injected afterward to simulate real-world mess.

Regenerate the dataset at any time:
```
python src/data_generator.py
```

## 5. ML Methodology

- **Target**: `Delay_Flag` (binary classification).
- **Features**: only information realistically known/estimable **at booking or
  dispatch time** (region, warehouse, carrier, mode, service type, package
  attributes, distance, promised days, handoffs, current capacity utilization,
  weather/traffic conditions, driver/vehicle availability, address quality,
  COD/peak/weekend/holiday/strike/monsoon flags). Post-delivery/realized fields
  (component hours, actual delay, delay category/reason, actual delivery date)
  are explicitly excluded to avoid label leakage.
- **Models compared**: Logistic Regression, Random Forest, HistGradientBoosting
  (all with `class_weight="balanced"`), inside a shared `ColumnTransformer`
  (StandardScaler for numeric, OneHotEncoder for categorical) + `Pipeline`.
- **Model selection**: a weighted score `0.45*recall + 0.35*ROC-AUC + 0.20*F1` on
  the delayed class, favoring recall because **missing an at-risk shipment is
  operationally costlier than a false alarm** -- a false negative means no chance
  to intervene or proactively notify the customer, while a false positive only
  costs a bit of extra operational attention.
- **Typical result** (varies slightly by run/seed): HistGradientBoosting wins
  with ~0.92 recall, ~0.83 precision, ~0.87 F1, ~0.98 ROC-AUC.
- Retrain and persist the best model:
  ```
  python src/model.py
  ```

## 6. Root Cause Methodology

- **Delay by Cause**: shipment counts and total delay-hours per `Delay_Category`,
  both in absolute terms and as % share -- computed live from the (filtered) data.
- **Root Cause Decomposition**: pick a segment (region/carrier/warehouse/mode/
  service type) and compare its delay rate, top causes, and dominant associated
  operational driver against the network average.
- **Driver Analysis**: box plots, scatter plots, delay-rate-by-category bars, and
  a correlation matrix over operational variables -- always phrased as
  **"strongly associated with delay"**, never as proven causation.
- **Dynamic insights & alerts**: `generate_key_insights()` and `alerts()` compute
  natural-language findings (peak-season effect, dominant cause, worst route/
  warehouse/carrier, weather impact) directly from the data every time filters
  change -- nothing is hard-coded.

## 7. Recommendation Methodology

`src/recommendations.py` inspects the (filtered) data for a fixed set of known
operational failure patterns (warehouse congestion, low driver availability, high
traffic, excess handoffs, poor address quality, documentation delay, strike/bandh
disruption, COD, adverse weather/monsoon). For each pattern that clears a minimum
sample-size and severity threshold, it emits a structured recommendation:
`{issue, evidence, recommendation, expected_impact, priority}`, prioritized by how
far the segment's delay rate exceeds the network average (HIGH >= 1.5x, MEDIUM >=
1.15x, else LOW). All "expected impact" language is explicitly labeled as an
estimate -- no fabricated financial or operational savings figures.

## 8. Installation

Requires Python 3.10+.

```bash
cd logistics_delay_analytics
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> **Windows note**: on machines with Windows **Smart App Control** enabled,
> brand-new/uncommon compiled wheel versions (e.g. a freshly released
> scikit-learn or pyarrow) can be blocked on first load ("An Application
> Control policy has blocked this file") until Microsoft's reputation service
> evaluates them. If you hit this, install a slightly older, widely-used
> version explicitly, e.g. `pip install "scikit-learn==1.4.2" "pyarrow==16.1.0"`,
> which resolves it immediately. This project deliberately does **not** hard-pin
> these in `requirements.txt`, since an old pin instead breaks deployment on
> Streamlit Community Cloud (which may run a newer Python without prebuilt
> wheels for an old pinned version). A `runtime.txt` / `.python-version` set to
> 3.11 is included to keep cloud deployments on the same Python version this
> app was developed and tested against.

## 9. How to Run

1. Generate the dataset (already included in `data/`, regenerate if desired):
   ```
   python src/data_generator.py
   ```
2. Train and save the model (already included in `models/`, retrain if desired):
   ```
   python src/model.py
   ```
3. Launch the app:
   ```
   streamlit run app.py
   ```
4. Run tests:
   ```
   pytest tests/test_pipeline.py -v
   ```

## 10. Application Pages

- **Executive Overview** -- KPI cards, volume/delay trends, on-time split, delay
  distribution, dynamically generated key insights.
- **Root Cause Analysis** -- delay-by-cause charts, segment decomposition,
  driver analysis (box/scatter/correlation), all data-driven.
- **Delay Prediction** -- model comparison table, confusion matrix, ROC curves,
  global feature importance, recall-first rationale.
- **Shipment Risk** -- pick an existing shipment or enter one manually; get a
  delay probability, risk level (LOW/MEDIUM/HIGH/CRITICAL), estimated delay
  hours, top contributing factors (SHAP), and tailored recommended actions.
- **Recommendations** -- prioritized, evidence-backed recommendation cards.
- **What-If Analysis** -- adjust operational parameters for a base shipment and
  see the model-predicted delay-probability change (explicitly labeled as a
  model-based scenario, not guaranteed causality).
- **Route / Warehouse / Carrier Analytics** -- performance tables & rankings,
  with sample-size safeguards for fair comparison.
- **Business Impact** -- delay hours, estimated SLA breaches, estimated customers
  affected, and an assumption-labeled estimated cost calculation.
- **Operational Alerts** -- auto-detected congestion/route/carrier anomalies plus
  a model-scored high-risk shipment list.
- **Intelligent Insights** -- a concentrated view of the programmatic insight engine.
- **Data Quality** -- raw-data completeness, missing values, duplicates, outliers,
  and the exact cleaning steps applied.
- **Model Governance** -- model card: training date/records, features, metrics,
  limitations, and the standing disclaimer.

All filters (date range, origin/destination, carrier, mode, service type,
warehouse, delay category, customer priority, weather, region) apply live across
every analytics page.

## 11. Assumptions

- All data is synthetic; causal relationships are designed to be directionally
  realistic (e.g. high warehouse utilization -> longer processing), not calibrated
  against a specific real operator's historical data.
- Business-impact costs (e.g. "₹300 per delayed shipment") are illustrative,
  user-adjustable assumptions, clearly labeled as such -- not measured figures.
- Features used for prediction are limited to what would plausibly be known/
  estimable at booking/dispatch time in a real operation.

## 12. Limitations

- Model performance on this synthetic dataset will not directly transfer to a
  real carrier's data; it must be revalidated on real historical shipments
  before any production use.
- Root-cause/driver relationships are statistical **associations**, not proven
  causal effects -- the app deliberately avoids causal language.
- Small-sample segments (a rare route, a low-volume carrier) are flagged but not
  fully de-biased; interpret rankings with sample size in mind.
- SHAP explanations are computed against the trained pipeline's encoded feature
  space; the fallback (non-SHAP) explanation method is a coarser approximation.

## 13. Future Enhancements

- Real-time data ingestion connectors (WMS/TMS integration) in place of static CSV.
- Time-series delay forecasting (e.g., predicting network-wide delay-rate trends).
- Automated model retraining/monitoring pipeline with drift detection.
- Route optimization recommendations informed by a network-flow / VRP solver.
- Multi-language / regional-language support for the customer-facing layer.
