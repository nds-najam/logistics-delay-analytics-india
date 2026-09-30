"""
Synthetic LR-level (lorry-receipt) data generator for the SRD Logistics
service-performance & live-reporting dashboard.

One row = one LR (consignment) travelling:

    Booking -> Collection -> Source Dispatch -> Main Hub (receipt/dispatch)
            -> [Via hub] -> Destination Arrival -> Destination Received -> Delivery

Delays are injected at a specific journey stage with stage-specific reasons,
IF/condition remarks and operational remarks, driven by designed-in causal
factors (problem hubs/sources/destinations, monsoon, weekday, business type,
chronic consignees) so that route, stage, customer and ML analysis all have
real structure to find. Targets are NOT stored in the LR data -- SRD wants them
configurable, so they live in a separate targets table (srd_targets.csv).
"""

import os

import numpy as np
import pandas as pd

from data_generator import CITY_INFO, _haversine_km

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LR_PATH = os.path.join(DATA_DIR, "srd_lr_data.csv")
TARGETS_PATH = os.path.join(DATA_DIR, "srd_targets.csv")

# Data is a snapshot taken at ASOF; anything that would happen later is still open.
ASOF = pd.Timestamp("2026-09-30 18:00")
START = pd.Timestamp("2025-10-01")

NETWORK_CITIES = [
    "Delhi", "Chandigarh", "Jaipur", "Lucknow", "Varanasi", "Mumbai", "Ahmedabad", "Surat",
    "Pune", "Bengaluru", "Chennai", "Hyderabad", "Kochi", "Coimbatore", "Visakhapatnam",
    "Kolkata", "Patna", "Bhubaneswar", "Bhopal", "Nagpur", "Indore", "Guwahati",
]
HUBS = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Hyderabad", "Nagpur"]
BUSINESSES = ["Express Cargo", "Part Load", "Full Truck Load", "E-commerce", "Door Delivery"]

# Journey stages (ordered). Used by analytics as well.
STAGES = ["Booking to Collection", "Dispatch", "Main Hub", "Transit/Via", "Final Arrival", "Delivery"]

# stage -> [(reason, weight, IF-condition remark, operational remark)]
STAGE_REASONS = {
    "Booking to Collection": [
        ("Vehicle not available for pickup", 0.35, "IF pickup vehicle not placed by cut-off THEN pickup moves to next day", "Additional pickup vehicle requested from branch"),
        ("Goods not ready at consignor", 0.30, "IF consignor goods not ready at visit THEN re-visit scheduled", "Consignor contacted; re-visit booked"),
        ("Pickup slot missed", 0.20, "IF pickup slot missed THEN next available slot used", "Pickup rescheduled by branch"),
        ("Booking data correction", 0.15, "IF booking details mismatch THEN LR held for correction", "Branch corrected LR details"),
    ],
    "Dispatch": [
        ("Godown stock pile-up", 0.30, "IF godown stock exceeds capacity THEN dispatch queued behind earlier LRs", "Extra loading shift planned"),
        ("Waiting for load consolidation", 0.30, "IF vehicle load below threshold THEN dispatch waits for consolidation", "Load held for next vehicle"),
        ("E-way bill / documentation pending", 0.25, "IF e-way bill not generated THEN LR cannot be dispatched", "Documentation team chased"),
        ("Loading delayed", 0.15, "IF loading crew short THEN loading extends past cut-off", "Crew reallocated from adjacent bay"),
    ],
    "Main Hub": [
        ("Hub congestion", 0.35, "IF hub inbound volume exceeds sorting capacity THEN LR waits in yard", "Hub manager escalated; extra shift added"),
        ("Late arrival of feeder vehicle", 0.25, "IF feeder vehicle arrives after connection cut-off THEN LR misses outbound vehicle", "LR shifted to next outbound vehicle"),
        ("Sorting backlog", 0.25, "IF sorting backlog > 12h THEN LRs processed in arrival order", "Manual sorting team deployed"),
        ("Manifest / scanning error", 0.15, "IF manifest mismatch found THEN LR held for verification", "Hub audit raised"),
    ],
    "Transit/Via": [
        ("Vehicle breakdown", 0.25, "IF breakdown > 6h THEN cargo is transshipped to an alternate vehicle", "Alternate vehicle arranged from nearest hub"),
        ("Weather / monsoon disruption", 0.20, "IF highway blocked by weather THEN vehicle waits or diverts", "Driver advised to hold at safe point"),
        ("Bandh / strike", 0.10, "IF regional bandh declared THEN movement suspended in region", "Movement resumed after clearance"),
        ("Route diversion / traffic", 0.15, "IF route diversion needed THEN transit time extends", "Route replanned by control room"),
        ("Via hub transshipment delay", 0.20, "IF via-hub transfer vehicle not ready THEN cargo waits at via hub", "Via hub supervisor escalated"),
        ("Driver shortage", 0.10, "IF relief driver unavailable THEN vehicle waits for driver", "Relief driver sent from hub"),
    ],
    "Final Arrival": [
        ("Unloading delay at destination", 0.40, "IF unloading bay occupied THEN vehicle queues", "Additional unloading labour arranged"),
        ("Destination godown congestion", 0.35, "IF inward backlog exists THEN LR inward entry delayed", "Destination manager escalated"),
        ("Short / excess found on inward", 0.25, "IF inward count mismatch THEN LR held until verified", "Shortage claim raised with source"),
    ],
    "Delivery": [
        ("Consignee closed / not available", 0.30, "IF consignee premises closed THEN delivery re-attempted next day", "Re-attempt scheduled; consignee called"),
        ("Appointment delivery requested", 0.20, "IF consignee requests appointment THEN delivery held until slot", "Appointment slot confirmed with consignee"),
        ("Address not traceable", 0.20, "IF address unclear THEN delivery boy calls consignee", "Consignee phone verified with consignor"),
        ("Delivery vehicle unavailable", 0.15, "IF local delivery vehicle short THEN deliveries spill over", "Hired vehicle added for area"),
        ("Consignee refused / payment issue", 0.15, "IF consignee refuses or payment pending THEN LR held at godown", "Consignor informed; instruction awaited"),
    ],
}

CONSIGNOR_WORDS_A = ["Shree", "Ganesh", "Bharat", "Sai", "Om", "Anand", "Vijay", "Kiran", "Lakshmi", "Royal",
                     "Modern", "National", "Prime", "Star", "Global", "Apex", "Metro", "Unity", "Classic", "Eastern"]
CONSIGNOR_WORDS_B = ["Textiles", "Pharma", "Electronics", "Agro Foods", "Auto Parts", "Plastics", "Steel", "FMCG",
                     "Garments", "Hardware", "Chemicals", "Paper Mills", "Appliances", "Traders", "Industries"]
CONSIGNEE_WORDS_A = ["Balaji", "Mahalaxmi", "Gupta", "Sharma", "Reddy", "Iyer", "Khan", "Patel", "Singh", "Das",
                     "Nair", "Joshi", "Verma", "Mehta", "Rao", "Bose", "Kapoor", "Menon", "Shah", "Pillai"]
CONSIGNEE_WORDS_B = ["Traders", "Enterprises", "Distributors", "Agencies", "Stores", "Retail", "Wholesale", "Mart"]


def _dist_matrix():
    lat = np.array([CITY_INFO[c][1] for c in NETWORK_CITIES])
    lon = np.array([CITY_INFO[c][2] for c in NETWORK_CITIES])
    # road distance ~ 1.25 x great-circle
    return 1.25 * _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def _hub_of(dm):
    hub_idx = [NETWORK_CITIES.index(h) for h in HUBS]
    return [HUBS[int(np.argmin([dm[i, j] for j in hub_idx]))] for i in range(len(NETWORK_CITIES))]


def route_reference() -> pd.DataFrame:
    """One row per Source->Destination route with main hub, via, distance, standard stage hours
    and the default benchmark target (days, service %)."""
    dm = _dist_matrix()
    ci = {c: i for i, c in enumerate(NETWORK_CITIES)}
    hub_of = dict(zip(NETWORK_CITIES, _hub_of(dm)))
    rows = []
    for s in NETWORK_CITIES:
        for d in NETWORK_CITIES:
            if s == d:
                continue
            hs, hd = hub_of[s], hub_of[d]
            direct = dm[ci[s], ci[d]]
            if hs == hd:
                via = "Direct"
                line = dm[ci[hs], ci[d]] if hs != d else 0.0
                line = max(line, 0.2 * direct)
            else:
                via = "Nagpur" if (direct > 2000 and "Nagpur" not in (hs, hd)) else hd
                if via == "Nagpur":
                    line = dm[ci[hs], ci["Nagpur"]] + dm[ci["Nagpur"], ci[hd]] + dm[ci[hd], ci[d]]
                else:
                    line = dm[ci[hs], ci[hd]] + dm[ci[hd], ci[d]]
            d_sh = dm[ci[s], ci[hs]]
            s1, s2 = 5.0, 8.0
            s3a = d_sh / 40.0 + 1.0
            s3b = 10.0
            s4 = line / 42.0 + (6.0 if via != "Direct" else 0.0) + 2.0
            s5, s6 = 4.0, 10.0
            total = s1 + s2 + s3a + s3b + s4 + s5 + s6
            target_days = max(1, int(np.ceil(total / 24.0 * 1.1)))
            tgt_pct = 92.0 if direct < 600 else (90.0 if direct < 1400 else 85.0)
            rows.append(dict(
                Source=s, Destination=d, Main_Hub=hs, Via=via, Distance_KM=round(float(direct), 1),
                Line_Haul_KM=round(float(line), 1), Dist_To_Hub_KM=round(float(d_sh), 1),
                Std_S1=s1, Std_S2=s2, Std_S3A=s3a, Std_S3B=s3b, Std_S4=s4, Std_S5=s5, Std_S6=s6,
                Total_Std_Hours=round(total, 1), Target_Days=target_days, Target_Service_Pct=tgt_pct,
            ))
    return pd.DataFrame(rows)


def default_targets() -> pd.DataFrame:
    r = route_reference()
    return r[["Source", "Destination", "Target_Days", "Target_Service_Pct"]].copy()


def _choice_rows(rng, weights):
    """Vectorised categorical sample, one draw per row of the weight matrix."""
    w = weights / weights.sum(axis=1, keepdims=True)
    c = np.cumsum(w, axis=1)
    u = rng.random(len(w))[:, None]
    return (u > c).sum(axis=1).clip(0, w.shape[1] - 1)


def generate_lr_data(n_records: int = 80_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ref = route_reference()
    nc = len(NETWORK_CITIES)
    ci = {c: i for i, c in enumerate(NETWORK_CITIES)}
    ref["rid"] = ref["Source"].map(ci) * nc + ref["Destination"].map(ci)
    ref = ref.set_index("rid").sort_index()
    ridx = ref.index.to_numpy()

    # ---------- entities ----------
    tier = np.array([CITY_INFO[c][3] for c in NETWORK_CITIES])
    city_w = np.where(tier == 1, 3.0, np.where(tier == 2, 1.6, 0.8))
    city_p = city_w / city_w.sum()

    n_consignor = 70
    cons_names = []
    while len(cons_names) < n_consignor:
        nm = f"{rng.choice(CONSIGNOR_WORDS_A)} {rng.choice(CONSIGNOR_WORDS_B)}"
        if nm not in cons_names:
            cons_names.append(nm)
    cons_home = rng.choice(nc, size=n_consignor, p=city_p)
    cons_biz = rng.choice(len(BUSINESSES), size=n_consignor, p=[0.30, 0.25, 0.12, 0.18, 0.15])
    cons_w = 1.0 / np.arange(1, n_consignor + 1) ** 0.75
    cons_w = rng.permutation(cons_w)
    cons_w = cons_w / cons_w.sum()
    cons_mult = np.exp(rng.normal(0, 0.35, n_consignor))  # chronic "goods not ready" consignors

    n_ce = 14  # consignees per destination city
    ce_names = np.empty((nc, n_ce), dtype=object)
    for i, c in enumerate(NETWORK_CITIES):
        used = set()
        for j in range(n_ce):
            while True:
                nm = f"{rng.choice(CONSIGNEE_WORDS_A)} {rng.choice(CONSIGNEE_WORDS_B)}"
                if nm not in used:
                    used.add(nm)
                    break
            ce_names[i, j] = f"{nm} ({c})"
    ce_mult = np.exp(rng.normal(0, 0.55, (nc, n_ce)))  # consignee "closed / appointment" tendency
    ce_w = 1.0 / np.arange(1, n_ce + 1) ** 0.8
    ce_w = ce_w / ce_w.sum()

    # ---------- booking times ----------
    days = pd.date_range(START, ASOF.normalize(), freq="D")
    dow = days.dayofweek.to_numpy()
    month = days.month.to_numpy()
    w_day = np.where(dow == 6, 0.30, 1.0) * np.select(
        [np.isin(month, [10, 11]), month == 3, np.isin(month, [6, 7, 8])], [1.25, 1.15, 0.92], default=1.0)
    w_day = w_day * np.linspace(0.92, 1.08, len(days))
    w_day = w_day / w_day.sum()
    day_i = rng.choice(len(days), size=n_records, p=w_day)
    hour_w = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 1, 2, 3.5, 5, 6, 6, 5, 5, 6, 6, 5.5, 4.5, 3, 1.8, 1.0, 0.6, 0.4, 0.3])
    hour = rng.choice(24, size=n_records, p=hour_w / hour_w.sum())
    book = (days[day_i] + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n_records), unit="s"))
    book = pd.DatetimeIndex(np.minimum(book.values, (ASOF - pd.Timedelta(minutes=5)).to_datetime64()))
    order = np.argsort(book.values, kind="stable")
    book = book[order]
    day_i = day_i[order]

    # ---------- parties, route ----------
    cidx = rng.choice(n_consignor, size=n_records, p=cons_w)
    home_hit = rng.random(n_records) < 0.80
    src = np.where(home_hit, cons_home[cidx], rng.choice(nc, size=n_records, p=city_p))
    dst = rng.choice(nc, size=n_records, p=city_p)
    same = dst == src
    while same.any():
        dst[same] = rng.choice(nc, size=int(same.sum()), p=city_p)
        same = dst == src
    biz_i = np.where(rng.random(n_records) < 0.85, cons_biz[cidx], rng.choice(len(BUSINESSES), size=n_records))
    ce_i = rng.choice(n_ce, size=n_records, p=ce_w)
    rid = src * nc + dst
    pos = np.searchsorted(ridx, rid)
    R = ref.iloc[pos]

    std = np.column_stack([R["Std_S1"], R["Std_S2"], R["Std_S3A"], R["Std_S3B"], R["Std_S4"], R["Std_S5"], R["Std_S6"]])
    via_flag = (R["Via"].to_numpy() != "Direct")

    # ---------- delay propensity ----------
    src_mult_city = np.ones(nc)
    dst_mult_city = np.ones(nc)
    for c, m in {"Surat": 1.6, "Pune": 1.3, "Lucknow": 1.25, "Patna": 1.3}.items():
        src_mult_city[ci[c]] = m
    for c, m in {"Guwahati": 1.8, "Patna": 1.5, "Kochi": 1.3, "Varanasi": 1.4, "Visakhapatnam": 1.25}.items():
        dst_mult_city[ci[c]] = m
    hub_mult = {"Mumbai": 1.5, "Delhi": 1.25, "Kolkata": 1.1}
    route_mult = np.exp(rng.normal(0, 0.30, nc * nc))
    main_hub = R["Main_Hub"].to_numpy()
    hubm = np.array([hub_mult.get(h, 1.0) for h in main_hub])
    bizm = np.array([1.0, 1.35, 1.1, 0.9, 1.2])[biz_i]
    mon = book.month.to_numpy()
    dw = book.dayofweek.to_numpy()
    seas = np.select([np.isin(mon, [7, 8, 9]), np.isin(mon, [10, 11])], [1.30, 1.20], default=1.0)
    seas = seas * np.where(dw == 5, 1.15, 1.0) * (1 + 0.10 * (day_i / max(day_i.max(), 1)))  # slow deterioration trend
    p = 0.20 * src_mult_city[src] * dst_mult_city[dst] * hubm * bizm * seas * route_mult[rid] * cons_mult[cidx] ** 0.5
    p = np.clip(p, 0.02, 0.75)
    inject = rng.random(n_records) < p

    W = np.tile(np.array([0.14, 0.16, 0.20, 0.22, 0.10, 0.14]), (n_records, 1))
    W[:, 0] *= src_mult_city[src] ** 2 * cons_mult[cidx] ** 1.5
    W[:, 1] *= src_mult_city[src]
    W[:, 2] *= np.where(np.isin(main_hub, ["Mumbai", "Delhi"]), 3.0, 1.0)
    W[:, 3] *= np.where(via_flag, 1.6, 1.0) * np.where(np.isin(mon, [6, 7, 8, 9]), 1.6, 1.0) * np.where(R["Via"].to_numpy() == "Nagpur", 1.6, 1.0)
    W[:, 4] *= dst_mult_city[dst]
    W[:, 5] *= dst_mult_city[dst] ** 2 * ce_mult[dst, ce_i] ** 1.5
    stage_k = _choice_rows(rng, W)

    lo = np.array([6, 6, 6, 4, 5, 8], dtype=float)
    hi = np.array([30, 30, 36, 14, 20, 60], dtype=float)
    std_stage = np.column_stack([std[:, 0], std[:, 1], std[:, 2] + std[:, 3], std[:, 4], std[:, 5], std[:, 6]])
    excess = np.where(inject, lo[stage_k] + rng.random(n_records) * (hi[stage_k] - lo[stage_k])
                      + np.where(stage_k == 3, std_stage[np.arange(n_records), 3] * rng.uniform(0.2, 0.6, n_records), 0), 0.0)

    # ---------- durations (hours) ----------
    noise = np.exp(rng.normal(0, 0.12, (n_records, 7)))
    h = std * noise
    ex = np.zeros((n_records, 6))
    ex[np.arange(n_records), stage_k] = excess
    h1 = h[:, 0] + ex[:, 0]
    h2 = h[:, 1] + ex[:, 1]
    to_hub = np.where(rng.random(n_records) < 0.25, 1, 0)  # a quarter of Main-Hub delay sits on the feeder leg
    h3a = h[:, 2] + ex[:, 2] * to_hub
    h3b = h[:, 3] + ex[:, 2] * (1 - to_hub)
    lh_std = std[:, 4] - np.where(via_flag, 6.0, 0.0)
    lh = lh_std * noise[:, 4] + ex[:, 3]
    via_dwell = np.where(via_flag, 6.0 * noise[:, 4], 0.0)
    h5 = h[:, 5] + ex[:, 4]
    h6 = h[:, 6] + ex[:, 5]

    def td(x):
        return pd.to_timedelta(x, unit="h")

    t0 = pd.Series(book)
    t1 = t0 + td(h1)
    t2 = t1 + td(h2)
    t3 = t2 + td(h3a)
    t4 = t3 + td(h3b)
    t_va = t4 + td(0.5 * lh)
    t_vd = t_va + td(via_dwell)
    t6 = t4 + td(lh + via_dwell)
    t7 = t6 + td(h5)
    t8 = t7 + td(h6)

    out = pd.DataFrame({
        "LR_No": [f"SRD{2500000 + i + 1}" for i in range(n_records)],
        "Booking_DT": t0, "Collection_DT": t1, "Dispatch_DT": t2, "Hub_Receipt_DT": t3, "Hub_Dispatch_DT": t4,
        "Via_Arrival_DT": t_va.where(via_flag), "Via_Dispatch_DT": t_vd.where(via_flag),
        "Dest_Arrival_DT": t6, "Dest_Received_DT": t7, "Delivery_DT": t8,
    })
    # ---------- stuck LRs (lost / misrouted / held) ----------
    stuck = (rng.random(n_records) < 0.004) & (t0 < ASOF - pd.Timedelta(days=6)).to_numpy()
    stuck_stage = rng.integers(2, 7, n_records)
    tcols = ["Booking_DT", "Collection_DT", "Dispatch_DT", "Hub_Receipt_DT", "Hub_Dispatch_DT", "Dest_Arrival_DT", "Dest_Received_DT", "Delivery_DT"]
    for k, col in enumerate(tcols):
        out.loc[stuck & (stuck_stage <= k), col] = pd.NaT
    out.loc[stuck & (stuck_stage <= 4), ["Via_Arrival_DT", "Via_Dispatch_DT"]] = pd.NaT
    # anything in the future relative to the snapshot has not happened yet
    for col in tcols + ["Via_Arrival_DT", "Via_Dispatch_DT"]:
        out.loc[out[col] > ASOF, col] = pd.NaT
    # keep ordering consistent after truncation
    chain = ["Booking_DT", "Collection_DT", "Dispatch_DT", "Hub_Receipt_DT", "Hub_Dispatch_DT", "Dest_Arrival_DT", "Dest_Received_DT", "Delivery_DT"]
    for a, b in zip(chain[:-1], chain[1:]):
        out.loc[out[a].isna(), b] = pd.NaT

    # ---------- attributes ----------
    w_params = {"Express Cargo": (4.6, 0.7), "Part Load": (6.3, 0.6), "Full Truck Load": (9.2, 0.25),
                "E-commerce": (2.7, 0.6), "Door Delivery": (4.2, 0.8)}
    biz_names = np.array(BUSINESSES)[biz_i]
    weight = np.array([rng.lognormal(*w_params[b]) for b in biz_names]).round(1).clip(1, 24000)
    out["Source"] = np.array(NETWORK_CITIES)[src]
    out["Destination"] = np.array(NETWORK_CITIES)[dst]
    out["Main_Hub"] = main_hub
    out["Via"] = R["Via"].to_numpy()
    out["Distance_KM"] = R["Distance_KM"].to_numpy()
    out["Business"] = biz_names
    out["Consignor"] = np.array(cons_names)[cidx]
    out["Consignee"] = ce_names[dst, ce_i]
    out["Weight_KG"] = weight
    out["Articles"] = np.maximum(1, (weight / rng.uniform(4, 25, n_records)).round()).astype(int)
    out["Freight_Amount_INR"] = np.maximum(150, weight * R["Distance_KM"].to_numpy() * 0.012 * rng.uniform(0.8, 1.25, n_records)).round(0)

    # ---------- status ----------
    status = np.select(
        [out["Delivery_DT"].notna(), out["Dest_Received_DT"].notna(), out["Dest_Arrival_DT"].notna(),
         out["Hub_Dispatch_DT"].notna(), out["Hub_Receipt_DT"].notna(), out["Dispatch_DT"].notna(),
         out["Collection_DT"].notna()],
        ["Delivered", "Out for Delivery", "At Destination", "In Transit", "At Main Hub", "Dispatched to Hub", "At Source Godown"],
        default="Booked")
    out["Status"] = status

    # ---------- reasons & remarks (recorded once the stage has started) ----------
    stage_start_col = ["Booking_DT", "Collection_DT", "Dispatch_DT", "Hub_Dispatch_DT", "Dest_Arrival_DT", "Dest_Received_DT"]
    reason, cond, opr = np.full(n_records, "", dtype=object), np.full(n_records, "", dtype=object), np.full(n_records, "", dtype=object)
    for k, stg in enumerate(STAGES):
        idx = np.where(inject & (stage_k == k) & out[stage_start_col[k]].notna().to_numpy())[0]
        if len(idx) == 0:
            continue
        opts = STAGE_REASONS[stg]
        wts = np.array([o[1] for o in opts], dtype=float)
        if stg == "Transit/Via":  # weather mostly in monsoon, via delays only for via routes
            wmat = np.tile(wts, (len(idx), 1))
            wmat[:, 1] *= np.where(np.isin(mon[idx], [6, 7, 8, 9]), 4.0, 0.3)
            wmat[:, 4] *= np.where(via_flag[idx], 1.0, 0.0)
            wmat[:, 2] *= np.where(np.isin(out["Source"].to_numpy()[idx], ["Patna", "Kolkata", "Guwahati", "Bhubaneswar"]), 2.5, 0.4)
            pick = _choice_rows(rng, wmat + 1e-9)
        else:
            pick = rng.choice(len(opts), size=len(idx), p=wts / wts.sum())
        reason[idx] = [opts[j][0] for j in pick]
        cond[idx] = [opts[j][2] for j in pick]
        opr[idx] = [opts[j][3] for j in pick]
    # stuck LRs carry an investigation remark
    sidx = np.where(stuck)[0]
    reason[sidx] = "Shortage / misrouted - under investigation"
    cond[sidx] = "IF LR not scanned for 72h THEN trace raised with hubs"
    opr[sidx] = "Trace request open with control room"
    out["Delay_Reason"] = reason
    out["Condition_Remark"] = cond
    out["Operational_Remark"] = opr
    for col in [c for c in out.columns if c.endswith("_DT")]:
        out[col] = out[col].dt.floor("min")
    return out


def build_and_save(n_records: int = 80_000, seed: int = 11):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_lr_data(n_records, seed)
    df.to_csv(LR_PATH, index=False)
    if not os.path.exists(TARGETS_PATH):
        default_targets().to_csv(TARGETS_PATH, index=False)
    return df


if __name__ == "__main__":
    d = build_and_save()
    print(d.shape)
    print(d["Status"].value_counts())
    print(d.head(3).T)
