"""
Live operations feed + live reports.

The live data source is pluggable:
  * SimulatedSource (default) reads the local LR snapshot, so the dashboard runs offline.
  * HttpSource calls the SRD Logistics application's API when SRD_API_BASE_URL is set.
    Expected contract (to be confirmed against SRD's real API specification):
        GET {SRD_API_BASE_URL}/lrs   (Authorization: Bearer {SRD_API_TOKEN})
        -> JSON list of LR objects with the same field names as data/srd_lr_data.csv
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd

import srd_analytics as an
import srd_generator as gen


class SimulatedSource:
    name = "Simulated feed (local snapshot)"

    def fetch(self) -> pd.DataFrame:
        return an.load_lr()


class HttpSource:
    def __init__(self, base_url: str, token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.name = f"SRD live API ({self.base_url})"

    def fetch(self) -> pd.DataFrame:
        req = urllib.request.Request(f"{self.base_url}/lrs")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            df = pd.DataFrame(json.loads(resp.read().decode("utf-8")))
        for c in [c for c in df.columns if c.endswith("_DT")]:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in ["Delay_Reason", "Condition_Remark", "Operational_Remark"]:
            df[c] = df[c].fillna("") if c in df else ""
        return df


def get_source():
    url = os.environ.get("SRD_API_BASE_URL", "").strip()
    return HttpSource(url, os.environ.get("SRD_API_TOKEN", "")) if url else SimulatedSource()


def live_reports(d: pd.DataFrame, asof: pd.Timestamp = gen.ASOF) -> dict:
    """Build every section-7 report from a prepared LR frame for the day containing `asof`."""
    today = asof.normalize()
    out = {}
    bk = d[d["Booking_Date"] == today]
    out["bookings_today"] = bk
    out["daily_booking"] = {
        "by_source": bk.groupby("Source").agg(LRs=("LR_No", "count"), Weight_T=("Weight_T", "sum"), Freight_INR=("Freight_Amount_INR", "sum")).round(2).reset_index().sort_values("LRs", ascending=False),
        "by_destination": bk.groupby("Destination").agg(LRs=("LR_No", "count"), Weight_T=("Weight_T", "sum"), Freight_INR=("Freight_Amount_INR", "sum")).round(2).reset_index().sort_values("LRs", ascending=False),
        "by_hour": bk.groupby(bk["Booking_DT"].dt.hour).size().rename("LRs").reset_index().rename(columns={"Booking_DT": "Hour"}),
    }

    # lagging / leading: last 7 days service vs target, and today's booking vs same weekday average
    win = d[d["Booking_Date"] > today - pd.Timedelta(days=7)]
    same_dow = d[(d["Booking_Date"].dt.dayofweek == today.dayofweek) & (d["Booking_Date"] < today) & (d["Booking_Date"] >= today - pd.Timedelta(days=28))]
    n_dow = max(same_dow["Booking_Date"].nunique(), 1)
    for key in ("Source", "Destination"):
        perf = an.summarize(win, key)[[key, "Volume", "Target_Pct", "Actual_Pct", "Variation_Pct", "Delay_Pct"]]
        usual = (same_dow.groupby(key).size() / n_dow).rename("Usual_Bookings")
        now = bk.groupby(key).size().rename("Bookings_Today")
        perf = perf.merge(usual, on=key, how="left").merge(now, on=key, how="left").fillna({"Usual_Bookings": 0, "Bookings_Today": 0})
        perf["Usual_Bookings"] = perf["Usual_Bookings"].round(1)
        perf["Performance"] = np.where(perf["Variation_Pct"] >= 0, "Leading", "Lagging")
        out[f"lag_lead_{key.lower()}"] = perf.sort_values("Variation_Pct")

    # daily target achievement: LRs delivered today
    dv = d[d["Delivery_DT"].dt.normalize() == today]
    out["delivered_today"] = dv
    for key in ("Source", "Destination", "Via"):
        t = dv.groupby(key).agg(Delivered_Today=("LR_No", "count"), Within_Target=("On_Track", "sum"), Target_Pct=("Target_Service_Pct", "mean")).reset_index()
        t["Achievement_Pct"] = (100 * t["Within_Target"] / t["Delivered_Today"]).round(2)
        t["Target_Pct"] = t["Target_Pct"].round(2)
        t["Variation_Pct"] = (t["Achievement_Pct"] - t["Target_Pct"]).round(2)
        out[f"target_{key.lower()}"] = t.sort_values("Variation_Pct")

    # godown stock (LRs physically at source / destination, not yet moved on / delivered)
    def godown(mask, key):
        s = d[mask]
        age = (asof - s["Booking_DT"]).dt.total_seconds() / 3600.0
        g = s.assign(Age=age).groupby(key).agg(LRs=("LR_No", "count"), Weight_T=("Weight_T", "sum"), Avg_Age_Hrs=("Age", "mean"),
                                               Max_Age_Hrs=("Age", "max")).round(1).reset_index()
        return g.sort_values("LRs", ascending=False)

    out["godown_source"] = godown(d["Status"].isin(["Booked", "At Source Godown"]), "Source")
    out["godown_destination"] = godown(d["Status"].isin(["At Destination", "Out for Delivery"]), "Destination")

    # main hub dispatch & movement up to end of day
    rows = []
    for hub, g in d.groupby("Main_Hub"):
        rows.append({
            "Main_Hub": hub,
            "Received_Today": int((g["Hub_Receipt_DT"].dt.normalize() == today).sum()),
            "Dispatched_Today": int((g["Hub_Dispatch_DT"].dt.normalize() == today).sum()),
            "Currently_At_Hub": int((g["Status"] == "At Main Hub").sum()),
            "In_Transit_Onward": int((g["Status"] == "In Transit").sum()),
            "Weight_At_Hub_T": round(float(g.loc[g["Status"] == "At Main Hub", "Weight_T"].sum()), 2),
        })
    out["hub_movement"] = pd.DataFrame(rows).sort_values("Currently_At_Hub", ascending=False)
    return out
