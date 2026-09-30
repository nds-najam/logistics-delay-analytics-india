"""
Analytics for the SRD Logistics service-performance dashboard.

Definitions (also shown in the app):
  * Actual days      = calendar days from booking date to delivery date
                       (to the snapshot date for LRs not yet delivered).
  * Delayed LR       = actual days > target days  (an undelivered LR that has already
                       exceeded its target is delayed too).
  * Actual service % = 100 * on-track LRs / total LRs.
  * Variation %      = actual service % - target service %   (negative = below target).
  * Delay point      = journey stage with the largest excess over the route's median
                       stage time.
Targets (days and service %) are configurable per route and are never stored in the LR data.
"""

import json
import os

import numpy as np
import pandas as pd

import srd_generator as gen

STAGES = gen.STAGES
STAGE_START = ["Booking_DT", "Collection_DT", "Dispatch_DT", "Hub_Dispatch_DT", "Dest_Arrival_DT", "Dest_Received_DT"]
STAGE_END = ["Collection_DT", "Dispatch_DT", "Hub_Dispatch_DT", "Dest_Arrival_DT", "Dest_Received_DT", "Delivery_DT"]
NO_POINT = "Within stage norms"
NO_REASON = "Not recorded"

RULES_PATH = os.path.join(gen.DATA_DIR, "srd_rules.json")
DEFAULT_RULES = {
    "default_target_days": 5,
    "default_target_pct": 90.0,
    "hub_receipt_target_hours": 24.0,    # booking -> main hub receipt
    "hub_dispatch_target_hours": 12.0,   # main hub receipt -> main hub dispatch
    "min_route_volume": 30,
    "min_stage_excess_hours": 2.0,
    "min_stage_excess_pct": 20.0,        # excess must also be >= this % of the route-stage median
    "suggest_weeks_window": 8,
    "suggest_weeks_below": 4,
}

STD_COLS = ["Volume", "On_Track", "Delayed", "Target_Days", "Avg_Actual_Days", "Variation_Days",
            "Target_Pct", "Actual_Pct", "Variation_Pct", "Delay_Pct"]


# --------------------------------------------------------------------------
# Loading / configuration
# --------------------------------------------------------------------------
def load_lr(path: str = gen.LR_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[c for c in pd.read_csv(path, nrows=0).columns if c.endswith("_DT")])
    for c in ["Delay_Reason", "Condition_Remark", "Operational_Remark"]:
        df[c] = df[c].fillna("")
    return df


def load_targets(path: str = gen.TARGETS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        t = gen.default_targets()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t.to_csv(path, index=False)
        return t
    return pd.read_csv(path)


def save_targets(t: pd.DataFrame, path: str = gen.TARGETS_PATH):
    t = t[["Source", "Destination", "Target_Days", "Target_Service_Pct"]].copy()
    t["Target_Days"] = t["Target_Days"].astype(int).clip(lower=1)
    t["Target_Service_Pct"] = t["Target_Service_Pct"].astype(float).clip(0, 100)
    t.to_csv(path, index=False)


def load_rules(path: str = RULES_PATH) -> dict:
    rules = dict(DEFAULT_RULES)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rules.update(json.load(f))
        except (OSError, ValueError):
            pass
    return rules


def save_rules(rules: dict, path: str = RULES_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)


# --------------------------------------------------------------------------
# Enrichment (targets, status, stage hours, delay point)
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame, targets: pd.DataFrame, rules: dict, asof: pd.Timestamp = gen.ASOF,
            target_mode: str = "benchmark", target_value: int = 0) -> pd.DataFrame:
    """Attach targets and all derived per-LR measures.

    target_mode: 'benchmark' (per-route table), 'fixed' (every route uses target_value days),
                 'offset' (benchmark + target_value days, may be negative).
    """
    d = df.copy()
    t = targets.rename(columns={"Target_Days": "_Bench_Days", "Target_Service_Pct": "Target_Service_Pct"})
    d = d.merge(t[["Source", "Destination", "_Bench_Days", "Target_Service_Pct"]], on=["Source", "Destination"], how="left")
    d["_Bench_Days"] = d["_Bench_Days"].fillna(rules["default_target_days"])
    d["Target_Service_Pct"] = d["Target_Service_Pct"].fillna(rules["default_target_pct"])
    if target_mode == "fixed":
        d["Target_Days"] = float(target_value)
    elif target_mode == "offset":
        d["Target_Days"] = (d["_Bench_Days"] + target_value).clip(lower=1)
    else:
        d["Target_Days"] = d["_Bench_Days"]
    d = d.drop(columns=["_Bench_Days"])

    d["Route"] = d["Source"] + " → " + d["Destination"]
    d["Delivered"] = d["Delivery_DT"].notna()
    end = d["Delivery_DT"].fillna(asof)
    d["Actual_Days"] = (end.dt.normalize() - d["Booking_DT"].dt.normalize()).dt.days.astype(float)
    d["Delayed"] = d["Actual_Days"] > d["Target_Days"]
    d["On_Track"] = ~d["Delayed"]
    d["Delay_Days"] = (d["Actual_Days"] - d["Target_Days"]).clip(lower=0)
    d["Weight_T"] = d["Weight_KG"] / 1000.0
    d["Booking_Date"] = d["Booking_DT"].dt.normalize()

    hrs = {}
    for k, (s, e) in enumerate(zip(STAGE_START, STAGE_END), start=1):
        finish = d[e].fillna(asof).where(d[s].notna())
        h = (finish - d[s]).dt.total_seconds() / 3600.0
        done = d[e].notna() & d[s].notna()
        hrs[k] = h
        d[f"S{k}_Hrs"] = h.round(2)
        med = h.where(done).groupby([d["Source"], d["Destination"]]).transform("median")
        d[f"S{k}_Med"] = med.round(2)
    excess = pd.concat([(hrs[k] - d[f"S{k}_Med"]) for k in range(1, 7)], axis=1)
    excess.columns = range(1, 7)
    meds = pd.concat([d[f"S{k}_Med"] for k in range(1, 7)], axis=1)
    meds.columns = range(1, 7)
    thr = np.maximum(rules["min_stage_excess_hours"], rules["min_stage_excess_pct"] / 100.0 * meds)
    excess = excess.where(excess >= thr)
    ex_max = excess.max(axis=1)
    arg = excess.fillna(-1e9).idxmax(axis=1)
    point = arg.map(lambda k: STAGES[int(k) - 1])
    d["Delay_Point_Hours"] = ex_max.round(1)
    d["Delay_Point"] = np.where(ex_max.notna(), point, NO_POINT)
    d["Delay_Reason_Disp"] = d["Delay_Reason"].where(d["Delay_Reason"] != "", NO_REASON)
    return d


# --------------------------------------------------------------------------
# Core summaries
# --------------------------------------------------------------------------
def summarize(df: pd.DataFrame, by) -> pd.DataFrame:
    by = [by] if isinstance(by, str) else list(by)
    d = df
    if not by:
        d = df.assign(Overall="Overall")
        by = ["Overall"]
    g = d.groupby(by, observed=True, sort=False)
    out = g.agg(Volume=("LR_No", "count"), Delivered=("Delivered", "sum"), On_Track=("On_Track", "sum"),
                Delayed=("Delayed", "sum"), Target_Days=("Target_Days", "mean"),
                Target_Pct=("Target_Service_Pct", "mean"), Weight_T=("Weight_T", "sum"))
    act = d.loc[d["Delivered"]].groupby(by, observed=True, sort=False)["Actual_Days"].mean().rename("Avg_Actual_Days")
    out = out.join(act)
    out["Actual_Pct"] = 100 * out["On_Track"] / out["Volume"]
    out["Variation_Pct"] = out["Actual_Pct"] - out["Target_Pct"]
    out["Delay_Pct"] = 100 * out["Delayed"] / out["Volume"]
    out["Variation_Days"] = out["Avg_Actual_Days"] - out["Target_Days"]
    for c in ["Target_Days", "Avg_Actual_Days", "Variation_Days", "Target_Pct", "Actual_Pct", "Variation_Pct", "Delay_Pct", "Weight_T"]:
        out[c] = out[c].round(2)
    return out.reset_index()


def with_status(tbl: pd.DataFrame) -> pd.DataFrame:
    t = tbl.copy()
    t["Status"] = np.where(t["Variation_Pct"] >= 0, "Meeting target", "Below target")
    return t


def target_sensitivity(df: pd.DataFrame, days=range(2, 11), by: str = None) -> pd.DataFrame:
    """Service % if every LR in the selection were judged against a fixed N-day target."""
    rows = []
    groups = [("All", df)] if by is None else list(df.groupby(by, observed=True))
    for name, g in groups:
        ad = g["Actual_Days"].to_numpy()
        for n in days:
            delayed = int((ad > n).sum())
            rows.append({"Group": name, "Target_Days": n, "Volume": len(g), "Delayed": delayed,
                         "Service_Pct": round(100 * (1 - delayed / max(len(g), 1)), 2),
                         "Delay_Pct": round(100 * delayed / max(len(g), 1), 2)})
    return pd.DataFrame(rows)


def stage_delay_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Delay % at each journey point among delayed LRs and as a share of all LRs."""
    dl = df[df["Delayed"]]
    tot = max(len(df), 1)
    g = dl.groupby("Delay_Point").agg(Delayed_LRs=("LR_No", "count"), Avg_Excess_Hours=("Delay_Point_Hours", "mean"),
                                      Avg_Delay_Days=("Delay_Days", "mean"))
    g = g.reindex(STAGES + [NO_POINT]).fillna(0).reset_index().rename(columns={"index": "Delay_Point"})
    g["Pct_of_Delayed"] = (100 * g["Delayed_LRs"] / max(len(dl), 1)).round(2)
    g["Pct_of_Volume"] = (100 * g["Delayed_LRs"] / tot).round(2)
    g["Avg_Excess_Hours"] = g["Avg_Excess_Hours"].round(1)
    g["Avg_Delay_Days"] = g["Avg_Delay_Days"].round(2)
    g["Delayed_LRs"] = g["Delayed_LRs"].astype(int)
    return g


def reason_summary(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Delay reason counts/% among delayed LRs, with the dominant IF-condition and operational remark."""
    dl = df[df["Delayed"]]
    keys = ([] if by is None else ([by] if isinstance(by, str) else list(by))) + ["Delay_Reason_Disp"]
    if dl.empty:
        return pd.DataFrame(columns=keys + ["Delayed_LRs", "Pct_of_Delayed", "Condition_Remark", "Operational_Remark"])
    g = dl.groupby(keys).agg(
        Delayed_LRs=("LR_No", "count"),
        Condition_Remark=("Condition_Remark", lambda s: s[s != ""].mode().iloc[0] if (s != "").any() else ""),
        Operational_Remark=("Operational_Remark", lambda s: s[s != ""].mode().iloc[0] if (s != "").any() else ""),
    ).reset_index()
    if by is None:
        g["Pct_of_Delayed"] = (100 * g["Delayed_LRs"] / g["Delayed_LRs"].sum()).round(2)
    else:
        first = keys[:-1]
        g["Pct_of_Delayed"] = (100 * g["Delayed_LRs"] / g.groupby(first)["Delayed_LRs"].transform("sum")).round(2)
    return g.sort_values("Delayed_LRs", ascending=False).reset_index(drop=True)


LR_LIST_COLS = ["LR_No", "Route", "Via", "Business", "Consignor", "Consignee", "Booking_DT", "Status", "Target_Days",
                "Actual_Days", "Delay_Days", "Delay_Point", "Delay_Reason_Disp", "Condition_Remark", "Operational_Remark"]


def lr_list(df: pd.DataFrame, delayed_only: bool = True) -> pd.DataFrame:
    d = df[df["Delayed"]] if delayed_only else df
    return d.sort_values("Delay_Days", ascending=False)[LR_LIST_COLS].rename(columns={"Delay_Reason_Disp": "Delay_Reason"})


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------
PERIOD_KINDS = ["Weekly", "10-Day", "15-Day", "Monthly", "Yearly (FY)"]


def period_columns(dates: pd.Series, kind: str):
    d = dates.dt.normalize()
    if kind == "Weekly":
        key = d - pd.to_timedelta(d.dt.dayofweek, unit="D")
        label = "Wk " + key.dt.strftime("%d %b %y")
    elif kind in ("10-Day", "15-Day"):
        if kind == "10-Day":
            part = np.where(d.dt.day <= 10, 1, np.where(d.dt.day <= 20, 11, 21))
            tag = {1: "01-10", 11: "11-20", 21: "21-EOM"}
        else:
            part = np.where(d.dt.day <= 15, 1, 16)
            tag = {1: "01-15", 16: "16-EOM"}
        key = pd.to_datetime(pd.DataFrame({"year": d.dt.year, "month": d.dt.month, "day": part}))
        label = key.dt.strftime("%b %y") + " " + pd.Series(part, index=d.index).map(tag)
    elif kind == "Monthly":
        key = d.dt.to_period("M").dt.to_timestamp()
        label = key.dt.strftime("%b %Y")
    else:  # financial year, April-March
        fy = np.where(d.dt.month >= 4, d.dt.year, d.dt.year - 1)
        key = pd.to_datetime(pd.DataFrame({"year": fy, "month": 4, "day": 1}))
        label = pd.Series(["FY %d-%02d" % (y, (y + 1) % 100) for y in fy], index=d.index)
    return label, key


def period_summary(df: pd.DataFrame, kind: str, by=None) -> pd.DataFrame:
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    lab, key = period_columns(df["Booking_DT"], kind)
    d = df.assign(Period=lab, Period_Key=key)
    out = summarize(d, ["Period_Key", "Period"] + by)
    return out.sort_values(["Period_Key"] + by).reset_index(drop=True)


def compare_periods(df: pd.DataFrame, a: tuple, b: tuple, by=None) -> pd.DataFrame:
    """Period A (current) vs period B (previous): service %, target %, variation %, delay %, LR volume."""
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))

    def sl(rng):
        s, e = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1)
        return df[(df["Booking_DT"] >= s) & (df["Booking_DT"] < e)]

    cols = ["Volume", "Actual_Pct", "Target_Pct", "Variation_Pct", "Delay_Pct"]
    sa, sb = summarize(sl(a), by), summarize(sl(b), by)
    if by:
        m = sa[by + cols].merge(sb[by + cols], on=by, how="outer", suffixes=("_Current", "_Previous"))
    else:
        m = pd.concat([sa[cols].add_suffix("_Current"), sb[cols].add_suffix("_Previous")], axis=1)
        m.insert(0, "Scope", "Overall")
    for c in cols:
        m[f"{c}_Change"] = (m[f"{c}_Current"].fillna(0) - m[f"{c}_Previous"].fillna(0)).round(2)
    return m


# --------------------------------------------------------------------------
# Booking -> Main hub -> dispatch
# --------------------------------------------------------------------------
AGEING_BINS = [-np.inf, 12, 24, 48, 72, np.inf]
AGEING_LABELS = ["0-12h", "12-24h", "24-48h", "48-72h", ">72h"]
HUB_STAGES = ["Pending at Source", "Dispatched - Not Received at Hub", "At Main Hub - Pending Dispatch", "Dispatched from Hub"]


def hub_movement(df: pd.DataFrame, rules: dict, asof: pd.Timestamp = gen.ASOF) -> pd.DataFrame:
    d = df[["LR_No", "Source", "Destination", "Route", "Main_Hub", "Business", "Consignor", "Weight_T", "Status",
            "Booking_DT", "Collection_DT", "Dispatch_DT", "Hub_Receipt_DT", "Hub_Dispatch_DT",
            "Delay_Point", "Delay_Reason_Disp", "Condition_Remark", "Operational_Remark"]].copy()
    hr = lambda a, b: (d[b] - d[a]).dt.total_seconds() / 3600.0
    d["Booking_to_Receipt_Hrs"] = hr("Booking_DT", "Hub_Receipt_DT").round(1)
    d["Receipt_to_Dispatch_Hrs"] = hr("Hub_Receipt_DT", "Hub_Dispatch_DT").round(1)
    src_pending = d["Dispatch_DT"].isna()
    not_recv = d["Dispatch_DT"].notna() & d["Hub_Receipt_DT"].isna()
    at_hub = d["Hub_Receipt_DT"].notna() & d["Hub_Dispatch_DT"].isna()
    d["Hub_Stage"] = np.select([src_pending, not_recv, at_hub], HUB_STAGES[:3], default=HUB_STAGES[3])
    since = d["Booking_DT"].where(src_pending, d["Dispatch_DT"].where(not_recv, d["Hub_Receipt_DT"]))
    d["Age_Hrs"] = np.where(d["Hub_Stage"] == HUB_STAGES[3], np.nan, ((asof - since).dt.total_seconds() / 3600.0).round(1))
    d["Ageing_Bucket"] = pd.cut(d["Age_Hrs"], AGEING_BINS, labels=AGEING_LABELS)
    t_recv, t_disp = rules["hub_receipt_target_hours"], rules["hub_dispatch_target_hours"]
    late_recv = np.where(d["Hub_Receipt_DT"].notna(), d["Booking_to_Receipt_Hrs"] > t_recv, (asof - d["Booking_DT"]).dt.total_seconds() / 3600.0 > t_recv)
    late_disp = np.where(d["Hub_Dispatch_DT"].notna(), d["Receipt_to_Dispatch_Hrs"] > t_disp,
                         np.where(at_hub, d["Age_Hrs"] > t_disp, False))
    d["Late_Receipt"] = late_recv
    d["Late_Dispatch"] = late_disp
    d["Hub_Delayed"] = d["Late_Receipt"] | d["Late_Dispatch"]
    return d


def hub_summary(h: pd.DataFrame, by: str) -> pd.DataFrame:
    g = h.groupby(by, observed=True)
    out = g.agg(Booked_LRs=("LR_No", "count"),
                Pending_at_Source=("Hub_Stage", lambda s: (s == HUB_STAGES[0]).sum()),
                Not_Received_at_Hub=("Hub_Stage", lambda s: (s == HUB_STAGES[1]).sum()),
                At_Hub_Pending_Dispatch=("Hub_Stage", lambda s: (s == HUB_STAGES[2]).sum()),
                Dispatched=("Hub_Stage", lambda s: (s == HUB_STAGES[3]).sum()),
                Avg_Booking_to_Receipt_Hrs=("Booking_to_Receipt_Hrs", "mean"),
                Avg_Receipt_to_Dispatch_Hrs=("Receipt_to_Dispatch_Hrs", "mean"),
                Delayed_LRs=("Hub_Delayed", "sum")).reset_index()
    out["Delay_Pct"] = (100 * out["Delayed_LRs"] / out["Booked_LRs"]).round(2)
    out["Avg_Booking_to_Receipt_Hrs"] = out["Avg_Booking_to_Receipt_Hrs"].round(1)
    out["Avg_Receipt_to_Dispatch_Hrs"] = out["Avg_Receipt_to_Dispatch_Hrs"].round(1)
    return out.sort_values("Delay_Pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Single LR journey
# --------------------------------------------------------------------------
MILESTONES = [("Booking", "Booking_DT", None), ("Collection", "Collection_DT", 1), ("Source Dispatch", "Dispatch_DT", 2),
              ("Main Hub Receipt", "Hub_Receipt_DT", None), ("Main Hub Dispatch", "Hub_Dispatch_DT", 3),
              ("Via Arrival", "Via_Arrival_DT", None), ("Via Dispatch", "Via_Dispatch_DT", None),
              ("Destination Arrival", "Dest_Arrival_DT", 4), ("Destination Received", "Dest_Received_DT", 5),
              ("Delivery", "Delivery_DT", 6)]


def lr_journey(row: pd.Series) -> pd.DataFrame:
    rows, prev = [], None
    for name, col, stage in MILESTONES:
        ts = row[col]
        if pd.isna(ts):
            if name.startswith("Via") and row["Via"] == "Direct":
                continue
            rows.append({"Milestone": name, "Timestamp": None, "Hours_Since_Previous": None, "Stage": STAGES[stage - 1] if stage else "",
                         "Route_Median_Hrs": row[f"S{stage}_Med"] if stage else None, "Excess_Hrs": None, "Reached": False})
            continue
        gap = None if prev is None else round((ts - prev).total_seconds() / 3600.0, 1)
        ex = None
        if stage:
            ex = round(row[f"S{stage}_Hrs"] - row[f"S{stage}_Med"], 1) if pd.notna(row[f"S{stage}_Med"]) else None
        rows.append({"Milestone": name, "Timestamp": ts, "Hours_Since_Previous": gap, "Stage": STAGES[stage - 1] if stage else "",
                     "Route_Median_Hrs": row[f"S{stage}_Med"] if stage else None, "Excess_Hrs": ex, "Reached": True})
        prev = ts
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# System-generated improvement suggestions
# --------------------------------------------------------------------------
SUGGESTION_TEMPLATES = {
    "Booking to Collection": "Plan extra pickup capacity at {src} and agree fixed pickup slots with top consignors; address '{reason}'.",
    "Dispatch": "Raise dispatch frequency / consolidation cut-offs at {src} godown and pre-clear documentation; address '{reason}'.",
    "Main Hub": "Add sorting shift or connection vehicle at {hub} hub for the {src} to {dst} flow; address '{reason}'.",
    "Transit/Via": "Review line-haul plan via {via} for {src} to {dst} (vehicle condition, relief drivers, contingency routing); address '{reason}'.",
    "Final Arrival": "Add unloading/inward capacity at {dst} godown and pre-plan vehicle arrival windows; address '{reason}'.",
    "Delivery": "Improve last-mile at {dst}: appointment calls before dispatch, more delivery vehicles, consignee follow-up; address '{reason}'.",
    NO_POINT: "No single stage dominates; re-baseline the {days}-day target for {src} to {dst} or review end-to-end planning.",
}


def suggestions(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Route-wise suggestions for routes with recurring delay / below-target achievement."""
    min_vol = rules["min_route_volume"]
    win, need = rules["suggest_weeks_window"], rules["suggest_weeks_below"]
    rt = summarize(df, ["Route", "Source", "Destination", "Via", "Main_Hub"])
    rt = rt[rt["Volume"] >= min_vol]
    wk_lab, wk_key = period_columns(df["Booking_DT"], "Weekly")
    last_keys = sorted(wk_key.unique())[-win:]
    w = df[wk_key.isin(last_keys)].assign(Wk=wk_key[wk_key.isin(last_keys)])
    wk = summarize(w, ["Route", "Wk"])
    wk = wk[wk["Volume"] >= 3]
    wk["below"] = wk["Actual_Pct"] < wk["Target_Pct"]
    rec = wk.groupby("Route").agg(Weeks_Observed=("below", "size"), Weeks_Below_Target=("below", "sum")).reset_index()
    rt = rt.merge(rec, on="Route", how="left").fillna({"Weeks_Observed": 0, "Weeks_Below_Target": 0})
    rt = rt[(rt["Variation_Pct"] < 0) & (rt["Weeks_Below_Target"] >= need)]
    if rt.empty:
        return pd.DataFrame()
    dl = df[df["Delayed"] & df["Route"].isin(rt["Route"])]
    pt = dl.groupby(["Route", "Delay_Point"]).size().rename("n").reset_index()
    pt = pt.sort_values(["Route", "n"], ascending=[True, False]).drop_duplicates("Route")
    tot = dl.groupby("Route").size().rename("tot")
    pt = pt.join(tot, on="Route")
    pt["Point_Share_Pct"] = (100 * pt["n"] / pt["tot"]).round(1)
    rs = dl[dl["Delay_Reason_Disp"] != NO_REASON].merge(pt[["Route", "Delay_Point"]], on=["Route", "Delay_Point"])
    top_reason = rs.groupby("Route")["Delay_Reason_Disp"].agg(lambda s: s.mode().iloc[0] if len(s) else NO_REASON).rename("Top_Reason")
    out = rt.merge(pt[["Route", "Delay_Point", "Point_Share_Pct"]], on="Route", how="left").merge(top_reason, on="Route", how="left")
    out["Delay_Point"] = out["Delay_Point"].fillna(NO_POINT)
    out["Top_Reason"] = out["Top_Reason"].fillna(NO_REASON)

    def text(r):
        tpl = SUGGESTION_TEMPLATES.get(r["Delay_Point"], SUGGESTION_TEMPLATES[NO_POINT])
        return tpl.format(src=r["Source"], dst=r["Destination"], via=r["Via"], hub=r["Main_Hub"], reason=r["Top_Reason"],
                          days=int(round(r["Target_Days"])))

    out["Suggestion"] = out.apply(text, axis=1)
    out["Impact_Score"] = (-out["Variation_Pct"]) * np.sqrt(out["Volume"])
    out["Priority"] = pd.cut(out["Impact_Score"].rank(pct=True), [0, 0.5, 0.8, 1.0], labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True)
    cols = ["Route", "Source", "Destination", "Via", "Volume", "Actual_Pct", "Target_Pct", "Variation_Pct", "Weeks_Below_Target",
            "Weeks_Observed", "Delay_Point", "Point_Share_Pct", "Top_Reason", "Suggestion", "Priority", "Impact_Score"]
    return out[cols].sort_values("Impact_Score", ascending=False).reset_index(drop=True)
