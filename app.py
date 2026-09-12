"""
DataQ -- Logistics Delay Intelligence
Streamlit proof-of-concept application.

Run with:  streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import data_processing as dp
import delay_analysis as da
import explainability as xp
import model as mdl
import recommendations as rec

# --------------------------------------------------------------------------
# Page config & light theming
# --------------------------------------------------------------------------
st.set_page_config(page_title="DataQ | Logistics Delay Intelligence", page_icon="\U0001F4E6", layout="wide")

PRIMARY = "#2454A6"
ACCENT = "#F2994A"
GOOD = "#1E8E5A"
BAD = "#D64545"
NEUTRAL = "#5B6B79"
PLOTLY_TEMPLATE = "plotly_white"
CATEGORY_COLORS = px.colors.qualitative.Safe

st.markdown(f"""
<style>
.kpi-card {{
    background: var(--background-color, #fff);
    border: 1px solid rgba(120,120,120,0.18);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.kpi-value {{ font-size: 1.6rem; font-weight: 700; color: {PRIMARY}; }}
.kpi-label {{ font-size: 0.78rem; color: {NEUTRAL}; text-transform: uppercase; letter-spacing: 0.03em; }}
.insight-card {{
    border-left: 4px solid {PRIMARY};
    background: rgba(36,84,166,0.06);
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.92rem;
}}
.alert-card {{
    border-left: 4px solid {BAD};
    background: rgba(214,69,69,0.07);
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 10px;
}}
.rec-card {{
    border: 1px solid rgba(120,120,120,0.18);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}}
.badge-HIGH {{ background:{BAD}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; }}
.badge-MEDIUM {{ background:{ACCENT}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; }}
.badge-LOW {{ background:{NEUTRAL}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; }}
.badge-CRITICAL {{ background:{BAD}; color:white; padding:3px 12px; border-radius:12px; font-weight:700; }}
.badge-RISKHIGH {{ background:#E08A3C; color:white; padding:3px 12px; border-radius:12px; font-weight:700; }}
.badge-RISKMEDIUM {{ background:#D8B84A; color:#3a2f00; padding:3px 12px; border-radius:12px; font-weight:700; }}
.badge-RISKLOW {{ background:{GOOD}; color:white; padding:3px 12px; border-radius:12px; font-weight:700; }}
.subtitle {{ color:{NEUTRAL}; font-size:1.0rem; margin-top:-10px; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data / model loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning shipment data...")
def get_clean_data():
    return dp.load_and_clean()


@st.cache_resource(show_spinner="Loading delay-risk model...")
def get_model():
    """Load the pre-trained model. Falls back to training a fresh model in-process
    if the saved artifact is missing or was pickled with an incompatible
    scikit-learn version (pickles are not guaranteed portable across sklearn
    versions, which can differ between a local dev environment and a cloud
    deployment's resolved dependencies)."""
    if os.path.exists(mdl.MODEL_PATH):
        try:
            return mdl.load_model()
        except Exception:
            pass  # fall through to retrain fresh with the currently installed sklearn version

    with st.spinner("No compatible saved model found -- training a fresh model now (one-time, ~1 minute)..."):
        train_df, _, _ = get_clean_data()
        training_output = mdl.train_and_evaluate(train_df)
        try:
            payload = mdl.save_best_model(training_output)
        except Exception:
            best_name = training_output["best_model_name"]
            best = training_output["results"][best_name]
            payload = {
                "pipeline": best["pipeline"], "model_name": best_name, "metrics": best["metrics"],
                "all_model_metrics": {k: v["metrics"] for k, v in training_output["results"].items()},
                "roc_curves": {k: v["roc_curve"] for k, v in training_output["results"].items()},
                "trained_at": training_output["trained_at"],
                "training_records": training_output["training_records"],
                "test_records": training_output["test_records"],
                "feature_list": training_output["feature_list"],
            }
    return payload


@st.cache_data(show_spinner=False)
def cached_kpis(df):
    return da.compute_kpis(df)


def risk_badge(level: str) -> str:
    cls = f"badge-RISK{level}" if level != "CRITICAL" else "badge-CRITICAL"
    return f'<span class="{cls}">{level}</span>'


def priority_badge(level: str) -> str:
    return f'<span class="badge-{level}">{level}</span>'


# --------------------------------------------------------------------------
# Sidebar: branding + filters
# --------------------------------------------------------------------------
clean_df, quality_report, cleaning_report = get_clean_data()

st.sidebar.markdown("## \U0001F4E6 DATAQ")
st.sidebar.markdown("**Logistics Delay Intelligence**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview", "Root Cause Analysis", "Delay Prediction", "Shipment Risk",
        "Recommendations", "What-If Analysis", "Route Analytics", "Warehouse Analytics",
        "Carrier Analytics", "Business Impact", "Operational Alerts", "Intelligent Insights",
        "Data Quality", "Model Governance",
    ],
    key="nav_page",
)

st.sidebar.divider()
st.sidebar.markdown("### Filters")

min_date = clean_df["Order_Date"].min().date()
max_date = clean_df["Order_Date"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

def multiselect_all(label, series):
    options = sorted(series.dropna().unique().tolist())
    return st.sidebar.multiselect(label, options, default=[])

origin_sel = multiselect_all("Origin city", clean_df["Origin_City"])
dest_sel = multiselect_all("Destination city", clean_df["Destination_City"])
region_sel = multiselect_all("Region (origin)", clean_df["Origin_Region"])
carrier_sel = multiselect_all("Carrier", clean_df["Carrier"])
mode_sel = multiselect_all("Shipping mode", clean_df["Shipping_Mode"])
service_sel = multiselect_all("Service type", clean_df["Service_Type"])
warehouse_sel = multiselect_all("Origin warehouse", clean_df["Origin_Warehouse"])
category_sel = multiselect_all("Delay category", clean_df["Delay_Category"])
priority_sel = multiselect_all("Customer priority", clean_df["Customer_Priority"])
weather_sel = multiselect_all("Weather", clean_df["Weather_Condition"])

if st.sidebar.button("Reset filters"):
    st.rerun()


def apply_filters(df):
    out = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        out = out[(out["Order_Date"].dt.date >= date_range[0]) & (out["Order_Date"].dt.date <= date_range[1])]
    for col, sel in [
        ("Origin_City", origin_sel), ("Destination_City", dest_sel), ("Origin_Region", region_sel),
        ("Carrier", carrier_sel), ("Shipping_Mode", mode_sel), ("Service_Type", service_sel),
        ("Origin_Warehouse", warehouse_sel), ("Delay_Category", category_sel),
        ("Customer_Priority", priority_sel), ("Weather_Condition", weather_sel),
    ]:
        if sel:
            out = out[out[col].isin(sel)]
    return out


df = apply_filters(clean_df)
st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(df):,}** of {len(clean_df):,} shipments")

DISCLAIMER = ("Predictions indicate risk based on historical patterns and should support, "
              "not replace, operational decision-making.")


# ==========================================================================
# PAGE: Executive Overview
# ==========================================================================
if page == "Executive Overview":
    st.title("DataQ")
    st.markdown('<p class="subtitle">Logistics Delay Intelligence -- Executive Overview</p>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    kpis = cached_kpis(df)
    cols = st.columns(4)
    kpi_defs = [
        ("Total Shipments", f"{kpis['total_shipments']:,}"),
        ("Delayed Shipments", f"{kpis['delayed_shipments']:,}"),
        ("Delay Rate", f"{kpis['delay_rate']}%"),
        ("On-Time Delivery %", f"{kpis['on_time_pct']}%"),
    ]
    for c, (label, val) in zip(cols, kpi_defs):
        c.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                    f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    cols2 = st.columns(4)
    kpi_defs2 = [
        ("Avg Delay (Delayed Shpts)", f"{kpis['avg_delay_hours']} h"),
        ("Median Delay (Delayed Shpts)", f"{kpis['median_delay_hours']} h"),
        ("Avg Delivery Time", f"{kpis['avg_delivery_days']} days"),
        ("Total Delay-Hours Impact", f"{kpis['estimated_delay_impact_hours']:,.0f} h"),
    ]
    for c, (label, val) in zip(cols2, kpi_defs2):
        c.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                    f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        vol = df.set_index("Order_Date").resample("W")["Shipment_ID"].count().rename("Shipments").reset_index()
        fig = px.line(vol, x="Order_Date", y="Shipments", title="Shipment Volume Over Time (weekly)",
                       template=PLOTLY_TEMPLATE)
        fig.update_traces(line_color=PRIMARY)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        wk = df.set_index("Order_Date").resample("W")["Delay_Flag"].mean().mul(100).rename("Delay Rate %").reset_index()
        fig = px.line(wk, x="Order_Date", y="Delay Rate %", title="Delay Rate Over Time (weekly)",
                       template=PLOTLY_TEMPLATE)
        fig.update_traces(line_color=BAD)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        counts = df["Delay_Flag"].map({0: "On Time", 1: "Delayed"}).value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(counts, names="Status", values="Count", title="On-Time vs Delayed Shipments",
                       color="Status", color_discrete_map={"On Time": GOOD, "Delayed": BAD}, hole=0.45,
                       template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        delayed = df.loc[df["Delay_Flag"] == 1]
        fig = px.histogram(delayed, x="Delay_Hours", nbins=40, title="Delay Distribution (hours)",
                             template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("\U0001F4A1 Key Insights")
    for ins in da.generate_key_insights(df):
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)


# ==========================================================================
# PAGE: Root Cause Analysis
# ==========================================================================
elif page == "Root Cause Analysis":
    st.title("Root Cause Analysis")
    st.markdown('<p class="subtitle">Why are packages getting delayed?</p>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    causes = da.delay_by_cause(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(causes, x="Shipments", y="Delay_Category", orientation="h",
                       title="Delay Category vs Number of Delayed Shipments",
                       template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(causes, x="Total_Delay_Hours", y="Delay_Category", orientation="h",
                       title="Delay Category vs Total Delay Hours",
                       template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Contributors")
    top_display = causes[["Delay_Category", "Shipments", "Share_of_Delayed_Pct", "Total_Delay_Hours", "Share_of_Hours_Pct"]]
    top_display.columns = ["Delay Category", "Delayed Shipments", "% of Delayed", "Total Delay Hours", "% of Delay Hours"]
    st.dataframe(top_display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Root Cause Decomposition by Segment")
    seg_col_label = st.selectbox("Decompose by", ["Origin_Region", "Carrier", "Origin_Warehouse", "Shipping_Mode", "Service_Type"])
    seg_values = sorted(df[seg_col_label].dropna().unique().tolist())
    seg_value = st.selectbox("Select value", seg_values)

    decomp = da.segment_decomposition(df, seg_col_label, seg_value)
    if decomp["n"] > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{seg_value} Delay Rate", f"{decomp['segment_rate']}%", f"{decomp['delta_pct_points']:+.1f} pp vs network")
        c2.metric("Network Average", f"{decomp['network_rate']}%")
        c3.metric("Relative Ratio", f"{decomp['relative_ratio']}x")

        seg_mask = df[seg_col_label] == seg_value
        driver = da.strongest_associated_driver(df, seg_mask)
        top_cause_row = decomp["top_causes"].iloc[0] if len(decomp["top_causes"]) else None

        st.markdown(f"""
        <div class="insight-card">
        <b>{seg_value}</b> has a <b>{decomp['segment_rate']}%</b> delay rate compared with a network average of
        <b>{decomp['network_rate']}%</b> ({decomp['relative_ratio']}x).<br>
        {"<b>" + str(top_cause_row['Delay_Category']) + "</b> contributes " + str(top_cause_row['Share_of_Hours_Pct']) + "% of " + str(seg_value) + "'s delay hours.<br>" if top_cause_row is not None else ""}
        The strongest associated operational factor for this segment is: <b>{driver}</b>
        (association only -- not a claim of causation).
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(decomp["top_causes"], use_container_width=True, hide_index=True)
    else:
        st.info("No shipments in this segment for the current filters.")

    st.divider()
    st.subheader("Driver Analysis")
    st.caption("Operational variables associated with delay. Wording reflects association, not proven causation.")

    driver_col1, driver_col2 = st.columns(2)
    with driver_col1:
        fig = px.box(df, x="Delay_Flag", y="Warehouse_Capacity_Utilization",
                       labels={"Delay_Flag": "Delayed (0=No, 1=Yes)"},
                       title="Warehouse Capacity Utilization vs Delay", template=PLOTLY_TEMPLATE,
                       color="Delay_Flag", color_discrete_map={0: GOOD, 1: BAD})
        st.plotly_chart(fig, use_container_width=True)
    with driver_col2:
        fig = px.box(df, x="Delay_Flag", y="Number_of_Handoffs",
                       labels={"Delay_Flag": "Delayed (0=No, 1=Yes)"},
                       title="Number of Handoffs vs Delay", template=PLOTLY_TEMPLATE,
                       color="Delay_Flag", color_discrete_map={0: GOOD, 1: BAD})
        st.plotly_chart(fig, use_container_width=True)

    driver_col3, driver_col4 = st.columns(2)
    with driver_col3:
        rate_by = (df.groupby("Traffic_Level")["Delay_Flag"].mean() * 100).reindex(
            ["Low", "Medium", "High", "Severe"]).reset_index()
        fig = px.bar(rate_by, x="Traffic_Level", y="Delay_Flag", title="Delay Rate by Traffic Level (%)",
                       template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)
    with driver_col4:
        rate_by = (df.groupby("Weather_Condition")["Delay_Flag"].mean() * 100).sort_values(ascending=False).reset_index()
        fig = px.bar(rate_by, x="Weather_Condition", y="Delay_Flag", title="Delay Rate by Weather Condition (%)",
                       template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, use_container_width=True)

    driver_col5, driver_col6 = st.columns(2)
    with driver_col5:
        fig = px.scatter(df.sample(min(4000, len(df)), random_state=1), x="Distance_KM", y="Transit_Time_Hours",
                           color=df.sample(min(4000, len(df)), random_state=1)["Delay_Flag"].map({0: "On Time", 1: "Delayed"}),
                           title="Distance vs Transit Time", template=PLOTLY_TEMPLATE,
                           color_discrete_map={"On Time": GOOD, "Delayed": BAD}, opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    with driver_col6:
        rate_by = (df.groupby("Address_Quality")["Delay_Flag"].mean() * 100).reindex(["Good", "Average", "Poor"]).reset_index()
        fig = px.bar(rate_by, x="Address_Quality", y="Delay_Flag", title="Delay Rate by Address Quality (%)",
                       template=PLOTLY_TEMPLATE, color_discrete_sequence=[NEUTRAL])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Correlation matrix (numeric operational variables)"):
        num_cols = ["Warehouse_Capacity_Utilization", "Number_of_Handoffs", "Distance_KM",
                    "Pickup_Delay_Hours", "EWay_Bill_Delay_Hours", "Delay_Hours"]
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, template=PLOTLY_TEMPLATE, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE: Delay Prediction
# ==========================================================================
elif page == "Delay Prediction":
    st.title("Delay Prediction")
    st.markdown('<p class="subtitle">Machine-learning model comparison and performance</p>', unsafe_allow_html=True)

    payload = get_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` to train and save a model.")
        st.stop()

    st.info(
        "**Why recall matters most here:** a missed delayed shipment (false negative) means operations "
        "loses the chance to intervene -- the customer is surprised by a late package with no proactive "
        "communication or corrective action. A false alarm (flagging an on-time shipment as at-risk) simply "
        "costs a bit of extra attention. Because the operational cost of missing a delay is materially higher "
        "than the cost of a false alarm, model selection is weighted toward **recall on the delayed class**, "
        "while still requiring strong overall discrimination (ROC-AUC) and balance (F1)."
    )

    all_metrics = payload["all_model_metrics"]
    comp_rows = []
    for name, m in all_metrics.items():
        comp_rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                           "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
                           "Selected": "✓" if name == payload["model_name"] else ""})
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.subheader(f"Selected Model: {payload['model_name']}")
    m = payload["metrics"]
    cols = st.columns(5)
    for c, (label, val) in zip(cols, [("Accuracy", m["accuracy"]), ("Precision", m["precision"]),
                                        ("Recall", m["recall"]), ("F1", m["f1"]), ("ROC-AUC", m["roc_auc"])]):
        c.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                    f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cm = np.array(m["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, x=["Pred: On Time", "Pred: Delayed"], y=["Actual: On Time", "Actual: Delayed"],
                          color_continuous_scale="Blues", title="Confusion Matrix", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for name, curve in payload["roc_curves"].items():
            fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(title="ROC Curves (all candidate models)", xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global Feature Importance")
    X_sample, _ = mdl.prepare_features(clean_df.sample(min(3000, len(clean_df)), random_state=42))
    imp = xp.global_feature_importance(payload, X_sample=X_sample, top_k=15)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", template=PLOTLY_TEMPLATE,
                   color_discrete_sequence=[PRIMARY], title="Top Features Driving Delay Risk (mean |SHAP value|)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.caption(DISCLAIMER)


# ==========================================================================
# PAGE: Shipment Risk
# ==========================================================================
elif page == "Shipment Risk":
    st.title("Shipment Risk")
    st.markdown('<p class="subtitle">Operational risk scoring for individual shipments</p>', unsafe_allow_html=True)

    payload = get_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` to train and save a model.")
        st.stop()

    mode_lookup = st.radio("Choose shipment", ["Select an existing shipment", "Enter a new shipment manually"], horizontal=True)

    if mode_lookup == "Select an existing shipment":
        sample_ids = clean_df["Shipment_ID"].sample(min(500, len(clean_df)), random_state=1).tolist()
        chosen_id = st.selectbox("Shipment ID", sample_ids)
        row = clean_df.loc[clean_df["Shipment_ID"] == chosen_id].iloc[0]
        input_row = row.to_dict()
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            origin_region = st.selectbox("Origin Region", sorted(clean_df["Origin_Region"].unique()))
            shipping_mode = st.selectbox("Shipping Mode", sorted(clean_df["Shipping_Mode"].unique()))
            service_type = st.selectbox("Service Type", sorted(clean_df["Service_Type"].unique()))
            package_type = st.selectbox("Package Type", sorted(clean_df["Package_Type"].unique()))
            carrier = st.selectbox("Carrier", sorted(clean_df["Carrier"].unique()))
        with c2:
            destination_region = st.selectbox("Destination Region", sorted(clean_df["Destination_Region"].unique()))
            origin_wh = st.selectbox("Origin Warehouse", sorted(clean_df["Origin_Warehouse"].unique()))
            dest_wh = st.selectbox("Destination Warehouse", sorted(clean_df["Destination_Warehouse"].unique()))
            customer_priority = st.selectbox("Customer Priority", sorted(clean_df["Customer_Priority"].unique()))
            weather = st.selectbox("Weather Condition", sorted(clean_df["Weather_Condition"].unique()))
        with c3:
            traffic = st.selectbox("Traffic Level", sorted(clean_df["Traffic_Level"].unique()))
            vehicle_avail = st.selectbox("Vehicle Availability", sorted(clean_df["Vehicle_Availability"].unique()))
            driver_avail = st.selectbox("Driver Availability", sorted(clean_df["Driver_Availability"].unique()))
            address_quality = st.selectbox("Address Quality", sorted(clean_df["Address_Quality"].unique()))

        c4, c5, c6 = st.columns(3)
        with c4:
            distance_km = st.number_input("Distance (KM)", 15.0, 6000.0, 500.0)
            package_weight = st.number_input("Package Weight (kg)", 0.1, 60.0, 2.0)
        with c5:
            promised_days = st.number_input("Promised Delivery Days", 1, 7, 3)
            handoffs = st.number_input("Number of Handoffs", 1, 6, 2)
        with c6:
            capacity_util = st.slider("Warehouse Capacity Utilization (%)", 30, 99, 75)
            cod = st.checkbox("COD Shipment")
        c7, c8, c9 = st.columns(3)
        with c7:
            peak_season = st.checkbox("Peak Season")
        with c8:
            weekend = st.checkbox("Weekend Dispatch")
        with c9:
            strike = st.checkbox("Strike/Bandh Region")
        holiday = False
        monsoon = st.checkbox("Monsoon Disruption Active")

        input_row = {
            "Origin_Region": origin_region, "Destination_Region": destination_region,
            "Origin_Warehouse": origin_wh, "Destination_Warehouse": dest_wh,
            "Shipping_Mode": shipping_mode, "Carrier": carrier, "Service_Type": service_type,
            "Package_Type": package_type, "Customer_Priority": customer_priority,
            "Weather_Condition": weather, "Traffic_Level": traffic,
            "Vehicle_Availability": vehicle_avail, "Driver_Availability": driver_avail,
            "Address_Quality": address_quality, "Distance_KM": distance_km,
            "Package_Weight": package_weight, "Package_Volume": package_weight * 0.012,
            "Promised_Delivery_Days": promised_days, "Number_of_Handoffs": handoffs,
            "Warehouse_Capacity_Utilization": capacity_util, "COD_Flag": int(cod),
            "Peak_Season_Flag": int(peak_season), "Weekend_Flag": int(weekend),
            "Strike_Disruption_Flag": int(strike), "Holiday_Flag": int(holiday),
            "Monsoon_Disruption_Flag": int(monsoon), "Shipment_ID": "MANUAL-ENTRY",
        }

    X_row = pd.DataFrame([input_row])[mdl.ALL_FEATURES]
    prob = mdl.predict_risk(payload, X_row)[0]
    level = mdl.risk_level(prob)
    est_delay_hours = round(prob * clean_df.loc[clean_df["Delay_Flag"] == 1, "Delay_Hours"].mean(), 1)

    st.divider()
    st.subheader(f"Shipment: {input_row.get('Shipment_ID', 'N/A')}")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-value">{prob*100:.0f}%</div>'
                 f'<div class="kpi-label">Delay Probability</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card">{risk_badge(level)}<div class="kpi-label" style="margin-top:6px;">Risk Level</div></div>',
                 unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-value">{est_delay_hours} h</div>'
                 f'<div class="kpi-label">Estimated Delay (if delayed)</div></div>', unsafe_allow_html=True)

    st.subheader("Why this risk score? (Top Contributing Factors)")
    factors = xp.explain_instance(payload, X_row, top_k=5)
    for i, f in enumerate(factors, 1):
        arrow = "▲" if f["impact"] > 0 else "▼"
        color = BAD if f["impact"] > 0 else GOOD
        st.markdown(f"**{i}. {f['feature']}** &nbsp; <span style='color:{color}'>{arrow} {f['direction']}</span>",
                     unsafe_allow_html=True)

    st.subheader("Recommended Actions for This Shipment")
    for r in rec.shipment_recommendations(factors, input_row):
        st.markdown(f"- {r}")

    st.caption(DISCLAIMER)


# ==========================================================================
# PAGE: Recommendations
# ==========================================================================
elif page == "Recommendations":
    st.title("Recommendations")
    st.markdown('<p class="subtitle">What should the logistics company do?</p>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    recs = rec.generate_recommendations(df, max_items=12)
    if not recs:
        st.success("No significant issues detected for the current filter selection.")
    for r in recs:
        st.markdown(f"""
        <div class="rec-card">
        {priority_badge(r['priority'])} &nbsp; <b>{r['issue']}</b><br><br>
        <b>Evidence:</b> {r['evidence']}<br>
        <b>Recommended Action:</b> {r['recommendation']}<br>
        <b>Expected Impact:</b> {r['expected_impact']}
        </div>
        """, unsafe_allow_html=True)

    st.caption("Impact statements are estimates derived from observed patterns, not guaranteed outcomes.")


# ==========================================================================
# PAGE: What-If Analysis
# ==========================================================================
elif page == "What-If Analysis":
    st.title("What-If Analysis")
    st.markdown('<p class="subtitle">Model-based scenario exploration</p>', unsafe_allow_html=True)
    st.warning("This is a **MODEL-BASED SCENARIO**, not a guarantee of causality or outcome.")

    payload = get_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` to train and save a model.")
        st.stop()

    sample_ids = clean_df["Shipment_ID"].sample(min(500, len(clean_df)), random_state=2).tolist()
    chosen_id = st.selectbox("Base shipment", sample_ids)
    base_row = clean_df.loc[clean_df["Shipment_ID"] == chosen_id].iloc[0].to_dict()
    base_X = pd.DataFrame([base_row])[mdl.ALL_FEATURES]
    base_prob = mdl.predict_risk(payload, base_X)[0]

    st.metric("Current Delay Probability", f"{base_prob*100:.0f}%")

    st.subheader("Adjust Operational Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        new_util = st.slider("Warehouse Capacity Utilization (%)", 30, 99, int(base_row["Warehouse_Capacity_Utilization"]))
        new_driver = st.selectbox("Driver Availability", ["Low", "Medium", "High"],
                                    index=["Low", "Medium", "High"].index(base_row["Driver_Availability"]))
    with c2:
        new_traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High", "Severe"],
                                     index=["Low", "Medium", "High", "Severe"].index(base_row["Traffic_Level"]))
        new_handoffs = st.slider("Number of Handoffs", 1, 6, int(base_row["Number_of_Handoffs"]))
    with c3:
        new_weather = st.selectbox("Weather Condition", sorted(clean_df["Weather_Condition"].unique()),
                                     index=sorted(clean_df["Weather_Condition"].unique()).index(base_row["Weather_Condition"]))
        new_vehicle = st.selectbox("Vehicle Availability", ["Low", "Medium", "High"],
                                     index=["Low", "Medium", "High"].index(base_row["Vehicle_Availability"]))

    scenario_row = dict(base_row)
    scenario_row.update({
        "Warehouse_Capacity_Utilization": new_util, "Driver_Availability": new_driver,
        "Traffic_Level": new_traffic, "Number_of_Handoffs": new_handoffs,
        "Weather_Condition": new_weather, "Vehicle_Availability": new_vehicle,
    })
    scenario_X = pd.DataFrame([scenario_row])[mdl.ALL_FEATURES]
    scenario_prob = mdl.predict_risk(payload, scenario_X)[0]

    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-value">{base_prob*100:.0f}%</div>'
                 f'<div class="kpi-label">Current Delay Probability</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-value">{scenario_prob*100:.0f}%</div>'
                 f'<div class="kpi-label">Predicted Delay Probability (What-If)</div></div>', unsafe_allow_html=True)

    delta = scenario_prob - base_prob
    if delta < 0:
        st.success(f"This scenario is predicted to **reduce** delay probability by {abs(delta)*100:.1f} percentage points.")
    elif delta > 0:
        st.error(f"This scenario is predicted to **increase** delay probability by {delta*100:.1f} percentage points.")
    else:
        st.info("No material change predicted.")

    fig = go.Figure(go.Bar(x=["Current", "What-If Scenario"], y=[base_prob*100, scenario_prob*100],
                             marker_color=[NEUTRAL, PRIMARY]))
    fig.update_layout(title="Delay Probability: Current vs Scenario", yaxis_title="Delay Probability (%)",
                        template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE: Route Analytics
# ==========================================================================
elif page == "Route Analytics":
    st.title("Route Analytics")
    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    routes = da.route_performance(df)
    min_vol = st.slider("Minimum shipment volume for ranking", 5, 500, 30)
    sig_routes = routes.loc[routes["Shipments"] >= min_vol]

    st.subheader("Worst-Performing Routes (High Delay Rate)")
    st.dataframe(sig_routes.sort_values("Delay_Rate_Pct", ascending=False).head(15)
                  [["Route", "Shipments", "Delay_Rate_Pct", "Avg_Delay_Hours", "On_Time_Pct", "Avg_Transit_Hours"]],
                  use_container_width=True, hide_index=True)

    st.subheader("Priority Areas: High Volume + High Delay Rate")
    fig = px.scatter(sig_routes, x="Shipments", y="Delay_Rate_Pct", size="Avg_Delay_Hours",
                       hover_name="Route", template=PLOTLY_TEMPLATE, color="Delay_Rate_Pct",
                       color_continuous_scale="RdYlGn_r", title="Route Volume vs Delay Rate")
    network_avg = 100 * df["Delay_Flag"].mean()
    fig.add_hline(y=network_avg, line_dash="dash", annotation_text="Network Average")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(routes.head(30), use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: Warehouse Analytics
# ==========================================================================
elif page == "Warehouse Analytics":
    st.title("Warehouse Analytics")
    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    wh = da.warehouse_performance(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(wh, x="Warehouse", y="Delay_Rate_Pct", title="Delay Rate by Warehouse (%)",
                       template=PLOTLY_TEMPLATE, color="Delay_Rate_Pct", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(wh, x="Shipments_Processed", y="Delay_Rate_Pct", size="Avg_Capacity_Utilization_Pct",
                           hover_name="Warehouse", template=PLOTLY_TEMPLATE, color="Avg_Capacity_Utilization_Pct",
                           color_continuous_scale="RdYlGn_r", title="Volume vs Delay Rate (bubble = capacity util.)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Warehouse Performance Table")
    st.dataframe(wh, use_container_width=True, hide_index=True)

    high_vol = wh["Shipments_Processed"].median()
    flagged = wh.loc[(wh["Shipments_Processed"] >= high_vol) & (wh["Delay_Rate_Pct"] > wh["Delay_Rate_Pct"].median())]
    if len(flagged):
        st.subheader("High Volume + High Delay Warehouses")
        st.dataframe(flagged, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: Carrier Analytics
# ==========================================================================
elif page == "Carrier Analytics":
    st.title("Carrier Analytics")
    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    min_ship = st.slider("Minimum shipments for fair comparison", 50, 2000, 200)
    carr = da.carrier_performance(df, min_shipments=min_ship)

    fig = px.bar(carr, x="Carrier", y="Delay_Rate_Pct", template=PLOTLY_TEMPLATE, title="Delay Rate by Carrier (%)",
                   color="Sufficient_Sample", color_discrete_map={True: PRIMARY, False: NEUTRAL})
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Carriers below the sample-size threshold are shown in gray to avoid unfair small-sample comparisons.")
    st.dataframe(carr, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: Business Impact
# ==========================================================================
elif page == "Business Impact":
    st.title("Business Impact")
    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    kpis = da.compute_kpis(df)
    st.subheader("Assumptions")
    cost_per_delay = st.number_input("Estimated cost per delayed shipment (USD) -- assumption", 1.0, 500.0, 8.0)
    sla_breach_hours = st.number_input("SLA breach threshold (hours) -- assumption", 1.0, 72.0, 24.0)

    delayed_df = df.loc[df["Delay_Flag"] == 1]
    sla_breaches = int((delayed_df["Delay_Hours"] > sla_breach_hours).sum())
    customers_affected = delayed_df["Customer_ID"].nunique() if "Customer_ID" in delayed_df.columns else None
    est_cost = kpis["delayed_shipments"] * cost_per_delay

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["delayed_shipments"]:,}</div>'
                 f'<div class="kpi-label">Delayed Shipments</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["estimated_delay_impact_hours"]:,.0f}</div>'
                 f'<div class="kpi-label">Total Delay Hours</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-value">{sla_breaches:,}</div>'
                 f'<div class="kpi-label">Estimated SLA Breaches (>{sla_breach_hours:.0f}h)</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-value">{customers_affected:,}</div>'
                 f'<div class="kpi-label">Estimated Customers Affected</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-card">
    <b>Estimated delay cost (ASSUMPTION-BASED)</b>: {kpis['delayed_shipments']:,} delayed shipments x
    ${cost_per_delay:.2f} assumed cost per delayed shipment = <b>${est_cost:,.0f}</b>.
    This is an estimate based on the assumption above, not a measured financial figure.
    </div>
    """, unsafe_allow_html=True)

    cause_impact = da.delay_by_cause(df)
    fig = px.bar(cause_impact, x="Delay_Category", y="Total_Delay_Hours", template=PLOTLY_TEMPLATE,
                   title="Total Delay-Hour Impact by Cause", color_discrete_sequence=[PRIMARY])
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE: Operational Alerts
# ==========================================================================
elif page == "Operational Alerts":
    st.title("Operational Alerts")
    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    al = da.alerts(df)
    if not al:
        st.success("No active alerts for the current filter selection.")
    for a in al:
        st.markdown(f"""
        <div class="alert-card">
        ⚠️ <b>ALERT: {a['type']}</b><br><br>
        <b>Entity:</b> {a['entity']}<br>
        <b>Metric:</b> {a['metric']}<br>
        <b>Detail:</b> {a['detail']}<br>
        <b>Recommended Action:</b> {a['action']}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("High-Risk Shipments (Model-Scored)")
    payload = get_model()
    if payload is not None:
        sample = df.sample(min(3000, len(df)), random_state=3)
        X_sample = sample[mdl.ALL_FEATURES]
        probs = mdl.predict_risk(payload, X_sample)
        sample = sample.assign(Delay_Probability=probs)
        high_risk = sample.loc[sample["Delay_Probability"] >= 0.75].sort_values("Delay_Probability", ascending=False)
        st.caption(f"{len(high_risk):,} shipments (of {len(sample):,} sampled) scored CRITICAL risk (>=75% delay probability).")
        st.dataframe(
            high_risk[["Shipment_ID", "Origin_City", "Destination_City", "Carrier", "Delay_Probability"]].head(25),
            use_container_width=True, hide_index=True,
        )


# ==========================================================================
# PAGE: Intelligent Insights
# ==========================================================================
elif page == "Intelligent Insights":
    st.title("DataQ Intelligent Insights")
    st.markdown('<p class="subtitle">Programmatically generated business insights (no external LLM used)</p>', unsafe_allow_html=True)
    if len(df) == 0:
        st.warning("No shipments match the current filters.")
        st.stop()

    for ins in da.generate_key_insights(df, max_insights=8):
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

    routes = da.route_performance(df)
    routes_sig = routes.loc[routes["Shipments"] >= max(20, int(0.001 * len(df)))]
    if len(routes_sig):
        top3 = routes_sig.sort_values("Total_Delay_Hours" if "Total_Delay_Hours" in routes_sig.columns else "Delay_Rate_Pct",
                                        ascending=False).head(3) if "Total_Delay_Hours" in routes_sig.columns else routes_sig.head(3)
        total_hours = (df.loc[df["Delay_Flag"] == 1, "Delay_Hours"]).sum()
        top3_hours = df[df.apply(lambda r: r["Origin_City"] + " -> " + r["Destination_City"], axis=1).isin(top3["Route"])]
        top3_hours = top3_hours.loc[top3_hours["Delay_Flag"] == 1, "Delay_Hours"].sum()
        share = 100 * top3_hours / total_hours if total_hours else 0
        st.markdown(f'<div class="insight-card">The top 3 routes by delay-hour impact account for '
                     f'{share:.0f}% of all delay hours.</div>', unsafe_allow_html=True)


# ==========================================================================
# PAGE: Data Quality
# ==========================================================================
elif page == "Data Quality":
    st.title("Data Quality")
    st.markdown('<p class="subtitle">Raw-data assessment and cleaning summary</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-value">{quality_report["n_records"]:,}</div>'
                 f'<div class="kpi-label">Raw Records</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-value">{quality_report["total_missing_cells"]:,}</div>'
                 f'<div class="kpi-label">Missing Cells</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-value">{quality_report["duplicate_full_rows"]:,}</div>'
                 f'<div class="kpi-label">Duplicate Records</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-value">{quality_report["data_completeness_pct"]}%</div>'
                 f'<div class="kpi-label">Data Completeness</div></div>', unsafe_allow_html=True)

    st.subheader("Missing Values by Column (raw)")
    if quality_report["missing_by_column"]:
        mv = pd.DataFrame(list(quality_report["missing_by_column"].items()), columns=["Column", "Missing Count"])
        st.dataframe(mv, use_container_width=True, hide_index=True)
    else:
        st.write("No missing values detected.")

    st.subheader("Outliers Detected (raw)")
    st.json(quality_report["outlier_counts_by_column"])

    st.subheader("Cleaning Pipeline Steps Applied")
    for step in cleaning_report["steps"]:
        st.markdown(f"- {step}")
    st.write(f"**Records before cleaning:** {cleaning_report['records_before']:,} | "
             f"**Records after cleaning:** {cleaning_report['records_after']:,} | "
             f"**Removed:** {cleaning_report['records_removed']:,}")


# ==========================================================================
# PAGE: Model Governance
# ==========================================================================
elif page == "Model Governance":
    st.title("Model Governance")
    payload = get_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` to train and save a model.")
        st.stop()

    st.subheader("Model Information")
    info = {
        "Model used": payload["model_name"],
        "Training date": payload["trained_at"],
        "Training records": f"{payload['training_records']:,}",
        "Test records": f"{payload['test_records']:,}",
        "Features used": len(payload["feature_list"]),
        "ROC-AUC": payload["metrics"]["roc_auc"],
        "Precision": payload["metrics"]["precision"],
        "Recall": payload["metrics"]["recall"],
        "F1": payload["metrics"]["f1"],
    }
    for k, v in info.items():
        st.markdown(f"**{k}:** {v}")

    with st.expander("Full feature list"):
        st.write(payload["feature_list"])

    st.subheader("Model Limitations")
    st.markdown("""
    - Trained on **synthetic** data with designed causal relationships; real-world performance will differ
      and must be validated against actual operational outcomes before production use.
    - Predictions reflect **historical/simulated patterns**, not guaranteed future outcomes.
    - The model does not account for one-off, unmodeled events (e.g. sudden regulatory changes, natural disasters
      outside the simulated disruption patterns).
    - Fairness across carriers/regions with small sample sizes has not been separately audited.
    - Correlational drivers shown in Root Cause Analysis and Driver Analysis are **associations**, not proven causal effects.
    """)
    st.info(DISCLAIMER)
