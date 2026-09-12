"""
Delay analytics: KPIs, root-cause decomposition, driver analysis,
route/warehouse/carrier performance, and dynamically generated insights.
No hard-coded numbers -- everything is computed from the (filtered) dataframe.
"""

import numpy as np
import pandas as pd

NETWORK_AVG_LABEL = "Network Average"


def compute_kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {k: 0 for k in [
            "total_shipments", "delayed_shipments", "delay_rate", "avg_delay_hours",
            "median_delay_hours", "on_time_pct", "avg_delivery_days", "estimated_delay_impact_hours",
        ]}
    delayed = df["Delay_Flag"].sum()
    delayed_df = df.loc[df["Delay_Flag"] == 1]
    return {
        "total_shipments": n,
        "delayed_shipments": int(delayed),
        "delay_rate": round(100 * delayed / n, 2),
        "avg_delay_hours": round(delayed_df["Delay_Hours"].mean(), 2) if delayed else 0.0,
        "median_delay_hours": round(delayed_df["Delay_Hours"].median(), 2) if delayed else 0.0,
        "on_time_pct": round(100 * (1 - delayed / n), 2),
        "avg_delivery_days": round(df["Actual_Delivery_Days"].mean(), 2),
        "estimated_delay_impact_hours": round(df["Delay_Hours"].sum(), 0),
    }


def delay_by_cause(df: pd.DataFrame) -> pd.DataFrame:
    delayed = df.loc[df["Delay_Flag"] == 1]
    if len(delayed) == 0:
        return pd.DataFrame(columns=["Delay_Category", "Shipments", "Share_of_Delayed_Pct", "Total_Delay_Hours", "Share_of_Hours_Pct"])
    counts = delayed["Delay_Category"].value_counts()
    hours = delayed.groupby("Delay_Category")["Delay_Hours"].sum()
    out = pd.DataFrame({
        "Delay_Category": counts.index,
        "Shipments": counts.values,
    })
    out["Share_of_Delayed_Pct"] = round(100 * out["Shipments"] / out["Shipments"].sum(), 2)
    out["Total_Delay_Hours"] = out["Delay_Category"].map(hours).fillna(0).round(1)
    out["Share_of_Hours_Pct"] = round(100 * out["Total_Delay_Hours"] / out["Total_Delay_Hours"].sum(), 2)
    return out.sort_values("Shipments", ascending=False).reset_index(drop=True)


def segment_decomposition(df: pd.DataFrame, segment_col: str, segment_value) -> dict:
    """Root-cause decomposition for a chosen segment vs. network average."""
    network_rate = 100 * df["Delay_Flag"].mean()
    seg = df.loc[df[segment_col] == segment_value]
    if len(seg) == 0:
        return {"segment_rate": 0, "network_rate": round(network_rate, 2), "n": 0, "top_causes": pd.DataFrame()}
    seg_rate = 100 * seg["Delay_Flag"].mean()
    causes = delay_by_cause(seg)
    return {
        "segment_rate": round(seg_rate, 2),
        "network_rate": round(network_rate, 2),
        "n": len(seg),
        "delta_pct_points": round(seg_rate - network_rate, 2),
        "relative_ratio": round(seg_rate / network_rate, 2) if network_rate > 0 else np.nan,
        "top_causes": causes,
    }


def strongest_associated_driver(df: pd.DataFrame, segment_mask: pd.Series) -> str:
    """Identify which operational driver most separates delayed vs on-time within a segment."""
    seg = df.loc[segment_mask]
    if len(seg) < 30 or seg["Delay_Flag"].nunique() < 2:
        return "Insufficient data"
    candidates = {
        "Warehouse capacity utilization": "Warehouse_Capacity_Utilization",
        "Number of handoffs": "Number_of_Handoffs",
        "Distance": "Distance_KM",
        "Pickup delay": "Pickup_Delay_Hours",
    }
    best_name, best_gap = None, 0
    for name, col in candidates.items():
        if col not in seg.columns:
            continue
        g = seg.groupby("Delay_Flag")[col].mean()
        if len(g) < 2:
            continue
        gap = abs(g.get(1, 0) - g.get(0, 0)) / (seg[col].std() + 1e-9)
        if gap > best_gap:
            best_gap, best_name = gap, name
    return best_name or "No dominant driver detected"


def route_performance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["Origin_City", "Destination_City"])
    out = g.agg(
        Shipments=("Shipment_ID", "count"),
        Delay_Rate_Pct=("Delay_Flag", lambda s: round(100 * s.mean(), 2)),
        Avg_Delay_Hours=("Delay_Hours", "mean"),
        Avg_Transit_Hours=("Transit_Time_Hours", "mean"),
    ).reset_index()
    out["Route"] = out["Origin_City"] + " -> " + out["Destination_City"]
    out["Avg_Delay_Hours"] = out["Avg_Delay_Hours"].round(2)
    out["Avg_Transit_Hours"] = out["Avg_Transit_Hours"].round(2)
    out["On_Time_Pct"] = round(100 - out["Delay_Rate_Pct"], 2)
    return out.sort_values("Shipments", ascending=False).reset_index(drop=True)


def warehouse_performance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Origin_Warehouse")
    out = g.agg(
        Shipments_Processed=("Shipment_ID", "count"),
        Avg_Processing_Hours=("Warehouse_Processing_Time_Hours", "mean"),
        Avg_Capacity_Utilization_Pct=("Warehouse_Capacity_Utilization", "mean"),
        Delay_Rate_Pct=("Delay_Flag", lambda s: round(100 * s.mean(), 2)),
        Avg_Delay_Hours=("Delay_Hours", "mean"),
    ).reset_index().rename(columns={"Origin_Warehouse": "Warehouse"})
    for c in ["Avg_Processing_Hours", "Avg_Capacity_Utilization_Pct", "Avg_Delay_Hours"]:
        out[c] = out[c].round(2)
    return out.sort_values("Delay_Rate_Pct", ascending=False).reset_index(drop=True)


def carrier_performance(df: pd.DataFrame, min_shipments: int = 200) -> pd.DataFrame:
    g = df.groupby("Carrier")
    out = g.agg(
        Shipments=("Shipment_ID", "count"),
        Delay_Rate_Pct=("Delay_Flag", lambda s: round(100 * s.mean(), 2)),
        Avg_Delay_Hours=("Delay_Hours", "mean"),
        Avg_Transit_Hours=("Transit_Time_Hours", "mean"),
    ).reset_index()
    out["On_Time_Pct"] = round(100 - out["Delay_Rate_Pct"], 2)
    out["Avg_Delay_Hours"] = out["Avg_Delay_Hours"].round(2)
    out["Avg_Transit_Hours"] = out["Avg_Transit_Hours"].round(2)
    out["Sufficient_Sample"] = out["Shipments"] >= min_shipments
    return out.sort_values("Delay_Rate_Pct", ascending=False).reset_index(drop=True)


def generate_key_insights(df: pd.DataFrame, max_insights: int = 6) -> list:
    """Dynamically compute a set of natural-language insight strings from the data."""
    insights = []
    n = len(df)
    if n == 0:
        return ["No data available for the current filter selection."]

    overall_rate = 100 * df["Delay_Flag"].mean()

    # peak season effect
    if df["Peak_Season_Flag"].nunique() > 1:
        peak_rate = 100 * df.loc[df["Peak_Season_Flag"] == 1, "Delay_Flag"].mean()
        off_rate = 100 * df.loc[df["Peak_Season_Flag"] == 0, "Delay_Flag"].mean()
        if off_rate > 0:
            change = peak_rate - off_rate
            insights.append(
                f"Delay rate is {abs(change):.1f} percentage points {'higher' if change > 0 else 'lower'} "
                f"during peak season ({peak_rate:.1f}%) versus the rest of the year ({off_rate:.1f}%)."
            )

    # dominant delay cause
    causes = delay_by_cause(df)
    if len(causes):
        top = causes.iloc[0]
        insights.append(
            f"{top['Delay_Category']} accounts for {top['Share_of_Delayed_Pct']:.0f}% of all delayed shipments "
            f"and {top['Share_of_Hours_Pct']:.0f}% of total delay hours."
        )

    # worst route vs network average (min sample size)
    routes = route_performance(df)
    routes_sig = routes.loc[routes["Shipments"] >= max(30, int(0.002 * n))]
    if len(routes_sig) and overall_rate > 0:
        worst = routes_sig.sort_values("Delay_Rate_Pct", ascending=False).iloc[0]
        ratio = worst["Delay_Rate_Pct"] / overall_rate
        insights.append(
            f"Route {worst['Route']} has a {ratio:.1f}x higher delay rate ({worst['Delay_Rate_Pct']:.1f}%) "
            f"than the network average ({overall_rate:.1f}%), based on {int(worst['Shipments']):,} shipments."
        )

    # worst warehouse
    wh = warehouse_performance(df)
    wh_sig = wh.loc[wh["Shipments_Processed"] >= max(30, int(0.005 * n))]
    if len(wh_sig):
        worst_wh = wh_sig.iloc[0]
        insights.append(
            f"Warehouse {worst_wh['Warehouse']} shows the highest delay rate at {worst_wh['Delay_Rate_Pct']:.1f}% "
            f"(network average {overall_rate:.1f}%), with average capacity utilization of "
            f"{worst_wh['Avg_Capacity_Utilization_Pct']:.0f}%."
        )

    # carrier comparison (sufficient sample only)
    carr = carrier_performance(df, min_shipments=max(200, int(0.01 * n)))
    carr_sig = carr.loc[carr["Sufficient_Sample"]]
    if len(carr_sig) > 1:
        worst_c = carr_sig.iloc[0]
        insights.append(
            f"{worst_c['Carrier']} has the highest delay rate ({worst_c['Delay_Rate_Pct']:.1f}%) among carriers "
            f"handling more than {max(200, int(0.01 * n)):,} shipments."
        )

    # weather effect
    if "Weather_Condition" in df.columns:
        wrate = df.groupby("Weather_Condition")["Delay_Flag"].mean() * 100
        if "Clear" in wrate.index and len(wrate) > 1:
            worst_weather = wrate.drop("Clear", errors="ignore").idxmax()
            ratio = wrate[worst_weather] / wrate["Clear"] if wrate["Clear"] > 0 else np.nan
            if pd.notna(ratio):
                insights.append(
                    f"Shipments experiencing {worst_weather} conditions have a {ratio:.1f}x higher delay rate "
                    f"than shipments in Clear weather."
                )

    return insights[:max_insights]


def alerts(df: pd.DataFrame) -> list:
    """Operational alerts: abnormal routes, near-capacity warehouses, deteriorating carriers, high-risk concentration."""
    alert_list = []
    n = len(df)
    if n == 0:
        return alert_list
    overall_rate = 100 * df["Delay_Flag"].mean()

    wh = warehouse_performance(df)
    for _, row in wh.iterrows():
        if row["Avg_Capacity_Utilization_Pct"] >= 88 and row["Delay_Rate_Pct"] > overall_rate * 1.3:
            alert_list.append({
                "type": "Warehouse Congestion",
                "entity": row["Warehouse"],
                "metric": f"Capacity utilization {row['Avg_Capacity_Utilization_Pct']:.0f}%",
                "detail": f"Delay rate {row['Delay_Rate_Pct']:.1f}% vs network {overall_rate:.1f}%",
                "action": "Redistribute incoming shipments to a nearby warehouse or add temporary processing capacity.",
            })

    routes = route_performance(df)
    routes_sig = routes.loc[routes["Shipments"] >= max(30, int(0.002 * n))]
    for _, row in routes_sig.sort_values("Delay_Rate_Pct", ascending=False).head(3).iterrows():
        if row["Delay_Rate_Pct"] > overall_rate * 1.5:
            alert_list.append({
                "type": "Abnormal Route Delay Rate",
                "entity": row["Route"],
                "metric": f"Delay rate {row['Delay_Rate_Pct']:.1f}%",
                "detail": f"{int(row['Shipments']):,} shipments vs network average {overall_rate:.1f}%",
                "action": "Investigate route-specific causes (weather, strikes, handoffs) and consider rerouting.",
            })

    carr = carrier_performance(df, min_shipments=max(200, int(0.01 * n)))
    for _, row in carr.loc[carr["Sufficient_Sample"]].iterrows():
        if row["Delay_Rate_Pct"] > overall_rate * 1.25:
            alert_list.append({
                "type": "Carrier Performance Deterioration",
                "entity": row["Carrier"],
                "metric": f"Delay rate {row['Delay_Rate_Pct']:.1f}%",
                "detail": f"{int(row['Shipments']):,} shipments vs network average {overall_rate:.1f}%",
                "action": "Review SLAs with carrier; consider volume reallocation to better-performing carriers.",
            })

    return alert_list
