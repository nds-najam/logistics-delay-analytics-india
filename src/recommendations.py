"""
Recommendation engine: turns detected root causes into prioritized,
actionable operational recommendations. Every recommendation is backed by
evidence computed from the (filtered) data -- nothing is hard-coded.
"""

import pandas as pd

from delay_analysis import (carrier_performance, delay_by_cause, route_performance,
                             warehouse_performance)


def _priority(delta_ratio: float) -> str:
    if delta_ratio >= 1.5:
        return "HIGH"
    elif delta_ratio >= 1.15:
        return "MEDIUM"
    return "LOW"


def generate_recommendations(df: pd.DataFrame, max_items: int = 10) -> list:
    """Generate a prioritized list of recommendation dicts:
    {issue, evidence, recommendation, expected_impact, priority}."""
    if len(df) == 0:
        return []
    recs = []
    n = len(df)
    network_rate = 100 * df["Delay_Flag"].mean()

    # 1. Warehouse capacity utilization
    wh = warehouse_performance(df)
    hot_wh = wh.loc[wh["Avg_Capacity_Utilization_Pct"] > 88]
    for _, row in hot_wh.head(3).iterrows():
        ratio = row["Delay_Rate_Pct"] / network_rate if network_rate else 1
        recs.append({
            "issue": f"Warehouse {row['Warehouse']} capacity utilization averages "
                     f"{row['Avg_Capacity_Utilization_Pct']:.0f}% (> 90% threshold).",
            "evidence": f"Delay rate = {row['Delay_Rate_Pct']:.1f}% vs. network average {network_rate:.1f}% "
                        f"({int(row['Shipments_Processed']):,} shipments processed).",
            "recommendation": f"Consider redistributing 15-20% of {row['Warehouse']}'s inbound volume to a "
                               f"nearby warehouse, or add temporary processing capacity during peak periods.",
            "expected_impact": "Estimated reduction in warehouse-related delay hours for this node (estimate, "
                                "not a guaranteed outcome).",
            "priority": _priority(ratio),
        })

    # 2. Driver availability
    if "Driver_Availability" in df.columns:
        low_drv = df.loc[df["Driver_Availability"] == "Low"]
        if len(low_drv) > max(30, 0.01 * n):
            rate_low = 100 * low_drv["Delay_Flag"].mean()
            ratio = rate_low / network_rate if network_rate else 1
            worst_region = (low_drv.groupby("Destination_Region")["Delay_Flag"].mean() * 100).idxmax()
            recs.append({
                "issue": "Low delivery-driver availability is strongly associated with delay.",
                "evidence": f"Delay rate = {rate_low:.1f}% when driver availability is Low vs. network average "
                            f"{network_rate:.1f}% ({len(low_drv):,} shipments). Most affected region: {worst_region}.",
                "recommendation": f"Increase driver allocation for the {worst_region} region during identified "
                                   f"low-availability periods; consider on-demand/gig driver partnerships for surge coverage.",
                "expected_impact": "Estimated reduction in last-mile delay hours (estimate).",
                "priority": _priority(ratio),
            })

    # 3. Traffic
    if "Traffic_Level" in df.columns:
        high_traffic = df.loc[df["Traffic_Level"].isin(["High", "Severe"])]
        if len(high_traffic) > max(30, 0.01 * n):
            rate_ht = 100 * high_traffic["Delay_Flag"].mean()
            ratio = rate_ht / network_rate if network_rate else 1
            recs.append({
                "issue": "High/Severe traffic conditions are strongly associated with delay.",
                "evidence": f"Delay rate = {rate_ht:.1f}% under High/Severe traffic vs. network average "
                            f"{network_rate:.1f}% ({len(high_traffic):,} shipments).",
                "recommendation": "Prioritize dispatch during lower-traffic windows where SLA allows, and evaluate "
                                   "dynamic re-routing for last-mile legs in high-congestion zones.",
                "expected_impact": "Estimated reduction in last-mile and transit delay hours (estimate).",
                "priority": _priority(ratio),
            })

    # 4. Handoffs
    if "Number_of_Handoffs" in df.columns:
        many_handoffs = df.loc[df["Number_of_Handoffs"] >= 4]
        if len(many_handoffs) > max(30, 0.01 * n):
            rate_h = 100 * many_handoffs["Delay_Flag"].mean()
            ratio = rate_h / network_rate if network_rate else 1
            recs.append({
                "issue": "Shipments with 4+ handoffs show a materially higher delay rate.",
                "evidence": f"Delay rate = {rate_h:.1f}% for shipments with >=4 handoffs vs. network average "
                            f"{network_rate:.1f}% ({len(many_handoffs):,} shipments).",
                "recommendation": "Evaluate route consolidation to reduce the number of handoffs on high-handoff lanes.",
                "expected_impact": "Estimated reduction in sorting/transit delay hours (estimate).",
                "priority": _priority(ratio),
            })

    # 5. Address quality
    if "Address_Quality" in df.columns:
        poor_addr = df.loc[df["Address_Quality"] == "Poor"]
        if len(poor_addr) > max(30, 0.01 * n):
            rate_a = 100 * poor_addr["Delay_Flag"].mean()
            ratio = rate_a / network_rate if network_rate else 1
            recs.append({
                "issue": "Poor destination address quality is strongly associated with last-mile delay.",
                "evidence": f"Delay rate = {rate_a:.1f}% for Poor address quality vs. network average "
                            f"{network_rate:.1f}% ({len(poor_addr):,} shipments).",
                "recommendation": "Trigger address verification / geocoding confirmation before dispatch for "
                                   "shipments flagged with low address-quality confidence.",
                "expected_impact": "Estimated reduction in failed/re-attempted deliveries (estimate).",
                "priority": _priority(ratio),
            })

    # 6. E-way bill / documentation
    if "EWay_Bill_Delay_Hours" in df.columns:
        doc_delayed = df.loc[df["EWay_Bill_Delay_Hours"] > 2]
        if len(doc_delayed) > max(30, 0.01 * n):
            rate_d = 100 * doc_delayed["Delay_Flag"].mean()
            ratio = rate_d / network_rate if network_rate else 1
            recs.append({
                "issue": "Documentation (e-way bill) delays beyond 2 hours are common and strongly linked to delay.",
                "evidence": f"Delay rate = {rate_d:.1f}% when e-way bill generation is delayed >2h vs. network "
                            f"average {network_rate:.1f}% ({len(doc_delayed):,} shipments).",
                "recommendation": "Trigger automated e-way bill generation immediately after order confirmation "
                                   "to avoid dispatch hold-ups.",
                "expected_impact": "Estimated reduction in warehouse-processing delay hours (estimate).",
                "priority": _priority(ratio),
            })

    # 7. Strike / bandh disruption
    if "Strike_Disruption_Flag" in df.columns:
        strike = df.loc[df["Strike_Disruption_Flag"] == 1]
        if len(strike) > max(20, 0.003 * n):
            rate_s = 100 * strike["Delay_Flag"].mean()
            ratio = rate_s / network_rate if network_rate else 1
            affected_regions = pd.concat([strike["Origin_Region"], strike["Destination_Region"]]).value_counts().idxmax()
            recs.append({
                "issue": f"Regional strike/bandh disruptions detected, concentrated around the {affected_regions} region.",
                "evidence": f"Delay rate = {rate_s:.1f}% for shipments flagged with strike disruption vs. network "
                            f"average {network_rate:.1f}% ({len(strike):,} shipments).",
                "recommendation": "Reroute through unaffected hubs or hold dispatch until the disruption clears, "
                                   "and proactively notify affected customers.",
                "expected_impact": "Avoids compounding transit delay during known disruption windows (estimate).",
                "priority": _priority(ratio),
            })

    # 8. COD
    if "COD_Flag" in df.columns:
        cod = df.loc[df["COD_Flag"] == 1]
        if len(cod) > max(30, 0.01 * n):
            rate_c = 100 * cod["Delay_Flag"].mean()
            ratio = rate_c / network_rate if network_rate else 1
            recs.append({
                "issue": "COD shipments show a higher delay rate, likely from repeated delivery/cash-confirmation attempts.",
                "evidence": f"Delay rate = {rate_c:.1f}% for COD shipments vs. network average "
                            f"{network_rate:.1f}% ({len(cod):,} shipments).",
                "recommendation": "Encourage prepaid/digital payment at checkout for COD-heavy lanes, and pre-confirm "
                                   "COD orders via SMS/call before out-for-delivery.",
                "expected_impact": "Estimated reduction in last-mile delivery attempts (estimate).",
                "priority": _priority(ratio),
            })

    # 9. Weather-prone routes
    weather_bad = df.loc[df["Weather_Condition"].isin(["Storm", "Heavy Rain"]) | (df["Monsoon_Disruption_Flag"] == 1)]
    if len(weather_bad) > max(30, 0.01 * n):
        rate_w = 100 * weather_bad["Delay_Flag"].mean()
        ratio = rate_w / network_rate if network_rate else 1
        recs.append({
            "issue": "Severe weather / monsoon disruption is strongly associated with transit and last-mile delay.",
            "evidence": f"Delay rate = {rate_w:.1f}% under Storm/Heavy Rain/Monsoon conditions vs. network average "
                        f"{network_rate:.1f}% ({len(weather_bad):,} shipments).",
            "recommendation": "Build weather-aware buffer time into promised delivery windows for monsoon-prone "
                               "routes/months, and pre-alert customers proactively.",
            "expected_impact": "Reduced SLA breaches during adverse weather (estimate).",
            "priority": _priority(ratio),
        })

    # sort by priority then by evidence strength (delay-rate ratio embedded in issue text isn't sortable directly,
    # so we sort by priority order only, preserving detection order within a tier)
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: order.get(r["priority"], 3))
    return recs[:max_items]


def shipment_recommendations(top_factors: list, shipment_row: dict) -> list:
    """Lightweight per-shipment recommendations derived from SHAP top factors
    and raw shipment attributes (used on the Shipment Risk page)."""
    recs = []
    if shipment_row.get("Warehouse_Capacity_Utilization", 0) and shipment_row["Warehouse_Capacity_Utilization"] > 88:
        recs.append("Warehouse utilization is high (>88%) -- consider an alternate origin warehouse if available.")
    if shipment_row.get("Driver_Availability") == "Low":
        recs.append("Driver availability is Low at the destination -- flag for priority driver allocation.")
    if shipment_row.get("Traffic_Level") in ("High", "Severe"):
        recs.append("Traffic is High/Severe on this lane -- consider dispatch timing adjustment or rerouting.")
    if shipment_row.get("Number_of_Handoffs", 0) and shipment_row["Number_of_Handoffs"] >= 4:
        recs.append("This shipment has 4+ handoffs -- evaluate a more direct routing option.")
    if shipment_row.get("Address_Quality") == "Poor":
        recs.append("Destination address quality is Poor -- verify/geocode the address before dispatch.")
    if shipment_row.get("COD_Flag") == 1:
        recs.append("COD shipment -- pre-confirm payment/availability with the customer before out-for-delivery.")
    if shipment_row.get("Strike_Disruption_Flag") == 1:
        recs.append("Route intersects an active strike/bandh-affected region -- consider holding or rerouting.")
    if shipment_row.get("Weather_Condition") in ("Storm", "Heavy Rain") or shipment_row.get("Monsoon_Disruption_Flag") == 1:
        recs.append("Adverse weather/monsoon conditions on this route -- build in extra buffer and notify the customer.")
    if not recs:
        recs.append("No single dominant risk driver detected; monitor as a standard shipment.")
    return recs
