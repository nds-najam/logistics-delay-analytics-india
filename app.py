"""
SRD Logistics -- Service Performance & Live Operations Dashboard.

Run with:  streamlit run app.py
(The earlier shipment-level delay-intelligence demo is kept in legacy_app.py.)
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import srd_analytics as an
import srd_generator as gen
import srd_live as lv
import srd_model as mdl

st.set_page_config(page_title="SRD Logistics | Service Performance", page_icon="\U0001F69A", layout="wide")

PRIMARY, ACCENT, GOOD, BAD, NEUTRAL = "#2454A6", "#F2994A", "#1E8E5A", "#D64545", "#5B6B79"
TEMPLATE = "plotly_white"
ASOF = gen.ASOF

st.markdown(f"""
<style>
.note {{ border-left: 4px solid {PRIMARY}; background: rgba(36,84,166,0.06); padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; margin-bottom: 8px; }}
.sug {{ border: 1px solid rgba(120,120,120,0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }}
.pill {{ padding: 2px 10px; border-radius: 12px; color: white; font-size: 0.75rem; font-weight: 600; }}
.pill-HIGH {{ background: {BAD}; }} .pill-MEDIUM {{ background: {ACCENT}; }} .pill-LOW {{ background: {NEUTRAL}; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
def _sig(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_resource(show_spinner="Loading LR data...")
def get_raw():
    """Fetch LR data from the configured live source; fall back to the local snapshot if it fails."""
    if not os.path.exists(gen.LR_PATH):
        gen.build_and_save()
    src, err = lv.get_source(), ""
    try:
        return src.fetch(), src.name, err
    except Exception as exc:  # network/auth/schema problems must not take the dashboard down
        err = f"{type(exc).__name__}: {exc}"
        return lv.SimulatedSource().fetch(), lv.SimulatedSource.name + " (fallback)", err


@st.cache_resource(show_spinner="Preparing service-performance measures...")
def get_prepared(targets_sig, rules_sig, raw_sig, mode, value):
    raw, _, _ = get_raw()
    return an.prepare(raw, an.load_targets(), an.load_rules(), ASOF, mode, value)


@st.cache_resource(show_spinner="Loading delay-risk model...")
def get_model(_d, sig):
    if os.path.exists(mdl.MODEL_PATH):
        try:
            return mdl.load()
        except Exception:
            pass  # incompatible pickle -> retrain below
    payload = mdl.train(_d)
    try:
        mdl.save(payload)
    except OSError:
        pass
    return payload


def clear_data_cache():
    get_prepared.clear()
    get_model.clear()


# --------------------------------------------------------------------------
# Sidebar: navigation + global filters (section 9)
# --------------------------------------------------------------------------
PAGES = ["Overview", "Service vs Target", "Network Performance", "Delay Analysis", "LR & Customer",
         "Period Reports", "Booking to Main Hub", "Live Operations", "Improvement Suggestions",
         "Delay Risk (ML)", "Configuration"]

st.sidebar.markdown("## \U0001F69A SRD LOGISTICS")
st.sidebar.markdown("**Service Performance & Live Reporting**")
page = st.sidebar.radio("Navigate", PAGES, key="nav_page")
st.sidebar.divider()

rules = an.load_rules()
st.sidebar.markdown("### Target days basis")
basis = st.sidebar.selectbox("Judge LRs against", ["Benchmark (per route)", "Fixed days for all routes", "Benchmark +/- days"], key="basis")
if basis.startswith("Fixed"):
    t_mode, t_value = "fixed", st.sidebar.number_input("Target days", 1, 30, 6, key="t_fixed")
elif basis.startswith("Benchmark +/-"):
    t_mode, t_value = "offset", st.sidebar.number_input("Add / subtract days", -5, 10, 0, key="t_off")
else:
    t_mode, t_value = "benchmark", 0

raw_df, source_name, source_err = get_raw()
D = get_prepared(_sig(gen.TARGETS_PATH), _sig(an.RULES_PATH), len(raw_df), t_mode, int(t_value))

st.sidebar.markdown("### Filters")
dmin, dmax = D["Booking_Date"].min().date(), D["Booking_Date"].max().date()
date_range = st.sidebar.date_input("Booking period", (dmin, dmax), min_value=dmin, max_value=dmax, key="f_date")


def ms(label, col):
    return st.sidebar.multiselect(label, sorted(D[col].dropna().unique().tolist()), key=f"f_{col}")


F_SEL = {
    "Source": ms("Source", "Source"), "Destination": ms("Destination", "Destination"), "Via": ms("Via", "Via"),
    "Business": ms("Business", "Business"), "Consignor": ms("Consignor", "Consignor"), "Consignee": ms("Consignee", "Consignee"),
}
td_sel = st.sidebar.multiselect("Target days (limit to)", sorted(D["Target_Days"].unique().tolist()), key="f_td")
lr_text = st.sidebar.text_input("LR no. (comma separated, partial ok)", key="f_lr")


def apply_filters(d, use_date=True):
    out = d
    if use_date and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        out = out[(out["Booking_Date"].dt.date >= date_range[0]) & (out["Booking_Date"].dt.date <= date_range[1])]
    for col, sel in F_SEL.items():
        if sel:
            out = out[out[col].isin(sel)]
    if td_sel:
        out = out[out["Target_Days"].isin(td_sel)]
    toks = [t for t in re.split(r"[,\s]+", lr_text.strip()) if t]
    if toks:
        out = out[out["LR_No"].str.contains("|".join(re.escape(t) for t in toks), case=False, regex=True)]
    return out


df = apply_filters(D)
st.sidebar.caption(f"Showing **{len(df):,}** of {len(D):,} LRs")
st.sidebar.caption(f"Data snapshot: {ASOF:%d %b %Y %H:%M}  \nSource: {source_name}")


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def need_data(d=None):
    d = df if d is None else d
    if len(d) == 0:
        st.warning("No LRs match the current filters.")
        st.stop()


def basis_note():
    txt = {"benchmark": "Benchmark target days per route (Configuration page)", "fixed": f"Fixed {t_value}-day target for every route",
           "offset": f"Benchmark target {t_value:+d} day(s)"}[t_mode]
    st.markdown(f'<div class="note"><b>Target basis:</b> {txt}. Service % = LRs on track / total LRs; '
                f'an LR is delayed when actual days exceed target days (open LRs count once they breach).</div>', unsafe_allow_html=True)


def _var_color(v):
    if pd.isna(v) or not isinstance(v, (int, float, np.number)):
        return ""
    return f"color: {BAD}; font-weight:600" if v < 0 else (f"color: {GOOD}" if v > 0 else "")


def show(d, key, height=None, download=True):
    d = d.reset_index(drop=True)
    var_cols = [c for c in d.columns if "Variation" in c and pd.api.types.is_numeric_dtype(d[c])]
    obj = d
    if var_cols and len(d) <= 1500:
        obj = d.style.map(_var_color, subset=var_cols).format(precision=2, na_rep="")
    kw = {"height": height} if height else {}
    st.dataframe(obj, hide_index=True, width="stretch", **kw)
    if download and len(d) <= 50000:
        st.download_button("Download CSV", d.to_csv(index=False).encode("utf-8"), f"{key}.csv", "text/csv", key=f"dl_{key}")


def kpi_row(d):
    s = an.summarize(d, []).iloc[0]
    c = st.columns(6)
    c[0].metric("Total LRs", f"{int(s.Volume):,}")
    c[1].metric("On-track", f"{int(s.On_Track):,}")
    c[2].metric("Delayed", f"{int(s.Delayed):,}", f"{s.Delay_Pct:.1f}% delay", delta_color="inverse")
    c[3].metric("Target service %", f"{s.Target_Pct:.1f}%")
    c[4].metric("Actual service %", f"{s.Actual_Pct:.1f}%", f"{s.Variation_Pct:+.1f} pts vs target")
    c[5].metric("Avg days (actual / target)", f"{s.Avg_Actual_Days:.1f} / {s.Target_Days:.1f}", f"{s.Variation_Days:+.1f} days", delta_color="inverse")
    return s


def target_bar(tbl, x, title, n=20, worst=True):
    t = tbl.sort_values("Variation_Pct", ascending=worst).head(n)
    fig = go.Figure()
    fig.add_bar(x=t[x], y=t["Actual_Pct"], name="Actual %", marker_color=PRIMARY)
    fig.add_scatter(x=t[x], y=t["Target_Pct"], name="Target %", mode="markers", marker=dict(color=ACCENT, size=11, symbol="diamond"))
    fig.update_layout(template=TEMPLATE, title=title, yaxis_title="Service %", height=380, margin=dict(t=50, b=10))
    fig.update_yaxes(range=[max(0, float(t["Actual_Pct"].min()) - 10), 101])
    st.plotly_chart(fig, width="stretch")


def std_cols(first):
    first = [first] if isinstance(first, str) else list(first)
    return first + an.STD_COLS


DIMENSIONS = {"Overall": [], "Source": "Source", "Destination": "Destination", "Route (Source → Destination)": ["Route", "Via"],
              "Via": "Via", "Business": "Business", "Consignor": "Consignor", "Consignee": "Consignee"}


# ==========================================================================
# Overview
# ==========================================================================
if page == "Overview":
    st.title("SRD Logistics - Service Performance Overview")
    need_data()
    basis_note()
    s = kpi_row(df)

    c1, c2 = st.columns(2)
    with c1:
        m = an.period_summary(df, "Monthly")
        fig = go.Figure()
        fig.add_bar(x=m["Period"], y=m["Volume"], name="LR volume", marker_color="rgba(36,84,166,0.25)", yaxis="y2")
        fig.add_scatter(x=m["Period"], y=m["Actual_Pct"], name="Actual %", line=dict(color=PRIMARY, width=3))
        fig.add_scatter(x=m["Period"], y=m["Target_Pct"], name="Target %", line=dict(color=ACCENT, dash="dash"))
        fig.update_layout(template=TEMPLATE, title="Monthly service % vs target", height=360, margin=dict(t=50, b=10),
                          yaxis=dict(title="Service %", range=[max(0, float(m["Actual_Pct"].min()) - 8), 101]),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="LRs"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        sd = an.stage_delay_summary(df)
        sd = sd[sd["Delayed_LRs"] > 0]
        fig = px.bar(sd, x="Delay_Point", y="Pct_of_Delayed", template=TEMPLATE, color_discrete_sequence=[BAD],
                     title="Where delays occur (% of delayed LRs by journey point)", text="Pct_of_Delayed")
        fig.update_layout(height=360, margin=dict(t=50, b=10), yaxis_title="% of delayed LRs", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    rt = an.summarize(df, ["Route"])
    rt = rt[rt["Volume"] >= rules["min_route_volume"]]
    with c1:
        st.subheader("Routes furthest below target")
        show(an.with_status(rt).sort_values("Variation_Pct").head(10)[["Route", "Volume", "Target_Pct", "Actual_Pct", "Variation_Pct", "Delay_Pct"]], "ov_worst", download=False)
    with c2:
        st.subheader("Routes furthest above target")
        show(rt.sort_values("Variation_Pct", ascending=False).head(10)[["Route", "Volume", "Target_Pct", "Actual_Pct", "Variation_Pct", "Delay_Pct"]], "ov_best", download=False)

    st.subheader("Current LR status")
    sc = df["Status"].value_counts().reset_index()
    sc.columns = ["Status", "LRs"]
    fig = px.bar(sc, x="Status", y="LRs", template=TEMPLATE, color_discrete_sequence=[PRIMARY])
    fig.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")


# ==========================================================================
# Service vs Target (sections 1 & 9)
# ==========================================================================
elif page == "Service vs Target":
    st.title("Service % vs Target Performance")
    need_data()
    basis_note()
    view = st.selectbox("Report by", list(DIMENSIONS.keys()), index=3, key="svt_dim")
    by = DIMENSIONS[view]
    tbl = an.with_status(an.summarize(df, by))
    lead = ["Overall"] if not by else ([by] if isinstance(by, str) else by)
    if by:
        c1, c2, c3 = st.columns(3)
        min_vol = c1.number_input("Minimum LR volume", 0, 5000, 0, key="svt_minvol")
        only_below = c2.checkbox("Only below-target", key="svt_below")
        sort_by = c3.selectbox("Sort by", ["Variation_Pct", "Delay_Pct", "Volume", "Actual_Pct"], key="svt_sort")
        tbl = tbl[tbl["Volume"] >= min_vol]
        if only_below:
            tbl = tbl[tbl["Variation_Pct"] < 0]
        tbl = tbl.sort_values(sort_by, ascending=sort_by in ("Variation_Pct", "Actual_Pct"))
    if tbl.empty:
        st.info("Nothing matches these settings.")
        st.stop()
    if by and len(tbl) > 1:
        label = lead[0]
        target_bar(tbl, label, f"{view}: actual vs target service % (furthest below target first)")
    show(tbl[std_cols(lead) + ["Status"]], "service_vs_target", height=460)

    st.subheader("Target-day sensitivity")
    st.caption("What would service % be if the selected LRs were judged against a fixed N-day target? "
               "Use this to analyse performance at 4, 5, 6, 7 days and above.")
    days = st.slider("Target days range", 1, 15, (3, 9), key="svt_days")
    sens = an.target_sensitivity(df, range(days[0], days[1] + 1))
    c1, c2 = st.columns([1, 1])
    with c1:
        show(sens.drop(columns="Group"), "target_sensitivity", download=True)
    with c2:
        fig = px.line(sens, x="Target_Days", y="Service_Pct", markers=True, template=TEMPLATE, color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=320, margin=dict(t=20, b=10), yaxis_title="Service %", xaxis_title="Target days")
        st.plotly_chart(fig, width="stretch")
    if by and isinstance(by, str):
        st.markdown(f"**Service % by {view} at fixed target days**")
        pick = st.multiselect("Target days to compare", list(range(2, 13)), default=[4, 5, 6, 7], key="svt_pick")
        if pick:
            rows = an.target_sensitivity(df, pick, by=by)
            pv = rows.pivot(index="Group", columns="Target_Days", values="Service_Pct")
            pv.columns = [f"{c}-day svc %" for c in pv.columns]
            pv = pv.reset_index().rename(columns={"Group": by})
            vol = df.groupby(by).size().rename("Volume").reset_index()
            show(vol.merge(pv, on=by).sort_values("Volume", ascending=False), "svc_by_target_days")


# ==========================================================================
# Network Performance (section 2)
# ==========================================================================
elif page == "Network Performance":
    st.title("Source / Destination / Via / Route Performance")
    need_data()
    basis_note()
    tabs = st.tabs(["Source → Destination", "Source-wise", "Destination-wise", "Via-wise", "Destination delivery",
                    "Delay ranking", "Business-wise"])
    with tabs[0]:
        rt = an.with_status(an.summarize(df, ["Route", "Source", "Destination", "Via"]))
        show(rt.sort_values("Volume", ascending=False)[std_cols(["Route", "Via"]) + ["Status"]], "route_perf", height=420)
        piv = an.summarize(df, ["Source", "Destination"])
        mat = piv.pivot(index="Source", columns="Destination", values="Variation_Pct")
        fig = px.imshow(mat, color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto", template=TEMPLATE,
                        title="Variation % vs target (red = below target)")
        fig.update_layout(height=520, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with tabs[1]:
        t = an.with_status(an.summarize(df, "Source"))
        target_bar(t, "Source", "Source-wise on-track service vs target", n=25)
        show(t.sort_values("Variation_Pct")[std_cols("Source") + ["Status"]], "source_perf")
    with tabs[2]:
        t = an.with_status(an.summarize(df, "Destination"))
        target_bar(t, "Destination", "Destination-wise on-track service vs target", n=25)
        show(t.sort_values("Variation_Pct")[std_cols("Destination") + ["Status"]], "dest_perf")
    with tabs[3]:
        t = an.with_status(an.summarize(df, "Via"))
        freq = df.groupby("Via").agg(Active_Days=("Booking_Date", "nunique")).reset_index()
        t = t.merge(freq, on="Via")
        t["LRs_per_Active_Day"] = (t["Volume"] / t["Active_Days"]).round(1)
        target_bar(t, "Via", "Via-wise service vs target", n=10)
        show(t.sort_values("Variation_Pct")[std_cols("Via") + ["LRs_per_Active_Day", "Status"]], "via_perf")
    with tabs[4]:
        st.caption("Delivery performance at each destination: delivery frequency, customers served and on-time delivery.")
        dv = df[df["Delivered"]]
        g = dv.groupby("Destination").agg(Delivered_LRs=("LR_No", "count"), Delivery_Days=("Delivery_DT", lambda s: s.dt.normalize().nunique()),
                                          Customers_Served=("Consignee", "nunique"), On_Time=("On_Track", "sum")).reset_index()
        g["Deliveries_per_Day"] = (g["Delivered_LRs"] / g["Delivery_Days"]).round(1)
        g["On_Time_Pct"] = (100 * g["On_Time"] / g["Delivered_LRs"]).round(2)
        g["Avg_Delivery_Stage_Hrs"] = dv.groupby("Destination")["S6_Hrs"].mean().round(1).reindex(g["Destination"]).to_numpy()
        show(g.sort_values("On_Time_Pct"), "dest_delivery")
        fig = px.scatter(g, x="Deliveries_per_Day", y="On_Time_Pct", size="Customers_Served", hover_name="Destination",
                         template=TEMPLATE, color_discrete_sequence=[PRIMARY], title="Delivery frequency vs on-time % (bubble = customers served)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    with tabs[5]:
        c1, c2 = st.columns(2)
        order = c1.radio("Order", ["Highest to lowest delay", "Lowest to highest delay"], horizontal=True, key="dr_order")
        metric = c2.radio("Rank by", ["Delay_Pct", "Delayed"], horizontal=True, key="dr_metric")
        asc = order.startswith("Lowest")
        for key, col in (("Source", "Source"), ("Destination", "Destination")):
            t = an.summarize(df, key).sort_values(metric, ascending=asc)
            st.subheader(f"{key}-wise delays")
            show(t[[key, "Volume", "Delayed", "Delay_Pct", "Target_Pct", "Actual_Pct", "Variation_Pct"]], f"delay_rank_{key}", height=300)
    with tabs[6]:
        t = an.with_status(an.summarize(df, "Business"))
        target_bar(t, "Business", "Business-wise achievement vs target", n=10)
        show(t.sort_values("Variation_Pct")[std_cols("Business") + ["Status"]], "business_perf")
        st.subheader("Target-wise source business performance")
        tw = an.summarize(df.assign(Target_Group=df["Target_Days"]), ["Target_Group", "Source", "Business"]).rename(columns={"Target_Group": "Target_Days_Group"})
        show(tw.sort_values(["Target_Days_Group", "Variation_Pct"])[["Target_Days_Group", "Source", "Business", "Volume", "Delayed", "Target_Pct", "Actual_Pct", "Variation_Pct", "Delay_Pct"]],
             "target_source_business", height=360)


# ==========================================================================
# Delay Analysis & drill-down (section 3)
# ==========================================================================
elif page == "Delay Analysis":
    st.title("Delay Analysis & Drill-Down")
    need_data()
    basis_note()
    sd = an.stage_delay_summary(df)
    st.subheader("Delay at each journey point")
    c1, c2 = st.columns([1, 1])
    with c1:
        show(sd, "stage_delay", download=True)
    with c2:
        fig = px.bar(sd[sd["Delayed_LRs"] > 0], x="Delay_Point", y="Pct_of_Volume", template=TEMPLATE, text="Delayed_LRs",
                     color_discrete_sequence=[BAD], title="Delayed LRs by journey point (% of all LRs; label = count)")
        fig.update_layout(height=340, margin=dict(t=50, b=10), xaxis_title="", yaxis_title="% of all LRs")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Delay reasons (with IF / operational remarks)")
    pt_choice = st.selectbox("Journey point", ["All"] + an.STAGES + [an.NO_POINT], key="da_point")
    base = df if pt_choice == "All" else df[df["Delay_Point"] == pt_choice]
    rs = an.reason_summary(base, by="Delay_Point" if pt_choice == "All" else None)
    show(rs, "delay_reasons", height=320)

    st.subheader("Route-wise variation → delay point → LR → reason")
    rt = an.with_status(an.summarize(df, ["Route"]))
    rt = rt[(rt["Volume"] >= rules["min_route_volume"]) & (rt["Delayed"] > 0)].sort_values("Variation_Pct")
    if rt.empty:
        st.info("No routes with delays in this selection.")
        st.stop()
    show(rt[std_cols("Route")], "route_variation", height=300)
    c1, c2 = st.columns(2)
    route = c1.selectbox("1. Select route", rt["Route"].tolist(), key="da_route")
    rdf = df[df["Route"] == route]
    rsd = an.stage_delay_summary(rdf)
    c2.selectbox("2. Select delay point", [p for p in rsd.loc[rsd["Delayed_LRs"] > 0, "Delay_Point"]] or [an.NO_POINT], key="da_rpoint")
    st.markdown(f"**Delay points on {route}**")
    show(rsd[rsd["Delayed_LRs"] > 0], "route_stage_delay", download=False)
    sel_point = st.session_state.get("da_rpoint")
    lrs = an.lr_list(rdf[rdf["Delay_Point"] == sel_point])
    st.markdown(f"**3. Delayed LRs at '{sel_point}' on {route}** ({len(lrs):,})")
    show(lrs, "drill_lrs", height=320)

    st.subheader("Worst-delay Destination / Source / Via")
    t1, t2, t3 = st.tabs(["Destination", "Source", "Via"])
    for tab, key in zip((t1, t2, t3), ("Destination", "Source", "Via")):
        with tab:
            w = an.summarize(df, key)
            w = w[w["Volume"] >= 20].sort_values("Delay_Pct", ascending=False).head(10)
            top = df[df["Delayed"]].groupby(key)["Delay_Point"].agg(lambda s: s.mode().iloc[0] if len(s) else "").rename("Major_Delay_Point")
            show(w.merge(top, on=key, how="left")[[key, "Volume", "Delayed", "Delay_Pct", "Variation_Pct", "Major_Delay_Point"]], f"worst_{key}", download=False)


# ==========================================================================
# LR & Customer (section 4)
# ==========================================================================
elif page == "LR & Customer":
    st.title("LR Journey & Customer Analysis")
    basis_note()
    t_lr, t_cu, t_rank = st.tabs(["LR search", "Consignor / Consignee", "Customer ranking"])

    with t_lr:
        q = st.text_input("Enter LR number (e.g. SRD2500123)", key="lr_q").strip().upper()
        if q:
            hit = D[D["LR_No"].str.upper() == q]
            if hit.empty:
                cand = D[D["LR_No"].str.upper().str.contains(re.escape(q))].head(10)["LR_No"].tolist()
                st.warning("LR not found." + (f" Did you mean: {', '.join(cand)}" if cand else ""))
            else:
                r = hit.iloc[0]
                c = st.columns(5)
                c[0].metric("Status", r["Status"])
                c[1].metric("Target days", f"{r['Target_Days']:.0f}")
                c[2].metric("Actual days" + ("" if r["Delivered"] else " (so far)"), f"{r['Actual_Days']:.0f}")
                c[3].metric("Deviation (days)", f"{r['Actual_Days'] - r['Target_Days']:+.0f}", delta_color="inverse")
                c[4].metric("Delay point", r["Delay_Point"] if r["Delayed"] else "On track")
                st.markdown(f"**{r['Route']}** via **{r['Via']}** (main hub {r['Main_Hub']}) | {r['Business']} | "
                            f"Consignor: **{r['Consignor']}** | Consignee: **{r['Consignee']}** | {r['Weight_KG']:,.0f} kg, {r['Articles']} articles")
                if r["Delay_Reason"]:
                    st.markdown(f'<div class="note"><b>Delay reason:</b> {r["Delay_Reason"]}<br><b>IF / condition:</b> {r["Condition_Remark"]}<br>'
                                f'<b>Operational remark:</b> {r["Operational_Remark"]}</div>', unsafe_allow_html=True)
                jr = an.lr_journey(r)
                show(jr.drop(columns="Reached"), "lr_journey", download=False)
                segs = []
                for k in range(6):
                    s_, e_ = r[an.STAGE_START[k]], r[an.STAGE_END[k]]
                    if pd.notna(s_):
                        segs.append({"Stage": an.STAGES[k], "Start": s_, "Finish": e_ if pd.notna(e_) else ASOF,
                                     "Result": "Delayed stage" if (pd.notna(r[f"S{k+1}_Med"]) and r[f"S{k+1}_Hrs"] - r[f"S{k+1}_Med"] >= max(rules["min_stage_excess_hours"], 0.2 * r[f"S{k+1}_Med"])) else "Normal"})
                if segs:
                    fig = px.timeline(pd.DataFrame(segs), x_start="Start", x_end="Finish", y="Stage", color="Result", template=TEMPLATE,
                                      color_discrete_map={"Delayed stage": BAD, "Normal": PRIMARY}, category_orders={"Stage": an.STAGES})
                    fig.update_yaxes(autorange="reversed")
                    fig.update_layout(height=320, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("Type an LR number to see its booking-to-delivery journey. Tip: pick one from the delayed list below.")
            sample = an.lr_list(df.head(200000)).head(10) if len(df) else pd.DataFrame()
            if len(sample):
                st.caption("Most delayed LRs in the current selection")
                show(sample[["LR_No", "Route", "Consignor", "Consignee", "Delay_Days", "Delay_Point", "Delay_Reason"]], "sample_lrs", download=False)

    with t_cu:
        kind = st.radio("Customer type", ["Consignor", "Consignee"], horizontal=True, key="cu_kind")
        names = sorted(D[kind].unique().tolist())
        who = st.selectbox(f"Search {kind.lower()}", names, key="cu_name")
        cdf = apply_filters(D)
        cdf = cdf[cdf[kind] == who]
        if cdf.empty:
            st.info("No LRs for this customer in the current filter selection.")
        else:
            kpi_row(cdf)
            dim = st.selectbox("Drill-down", ["Source", "Destination", "Route", "Via", "Business", "Period (monthly)"], key="cu_dim")
            if dim.startswith("Period"):
                t = an.period_summary(cdf, "Monthly")
                lead = "Period"
            else:
                t = an.summarize(cdf, dim)
                lead = dim
            t = an.with_status(t)
            show(t[std_cols(lead) + ["Status"]], "customer_drill")
            st.markdown("**Delayed LRs**")
            show(an.lr_list(cdf), "customer_delayed_lrs", height=300)

    with t_rank:
        kind = st.radio("Report for", ["Consignor", "Consignee"], horizontal=True, key="rk_kind")
        mv = st.number_input("Minimum LRs", 1, 500, 20, key="rk_min")
        need_data()
        t = an.with_status(an.summarize(df, kind))
        t = t[t["Volume"] >= mv].sort_values("Variation_Pct")
        if t.empty:
            st.info("No customers meet the minimum volume.")
        else:
            target_bar(t, kind, f"{kind}-wise service vs target (furthest below target)", n=20)
            show(t[std_cols(kind) + ["Status"]], f"{kind.lower()}_ranking", height=420)


# ==========================================================================
# Period reports (section 5)
# ==========================================================================
elif page == "Period Reports":
    st.title("Period-wise & Comparison Reports")
    need_data()
    basis_note()
    t_per, t_mon, t_cmp = st.tabs(["Period performance", "Month-wise comparison", "Compare any two periods"])
    with t_per:
        kind = st.radio("Period", an.PERIOD_KINDS, horizontal=True, index=3, key="pp_kind")
        ps = an.period_summary(df, kind)
        if kind == "Weekly":
            ps = ps.tail(26)
        fig = go.Figure()
        fig.add_bar(x=ps["Period"], y=ps["Volume"], name="LRs", marker_color="rgba(36,84,166,0.25)", yaxis="y2")
        fig.add_scatter(x=ps["Period"], y=ps["Actual_Pct"], name="Actual %", line=dict(color=PRIMARY, width=3))
        fig.add_scatter(x=ps["Period"], y=ps["Target_Pct"], name="Target %", line=dict(color=ACCENT, dash="dash"))
        fig.update_layout(template=TEMPLATE, height=360, margin=dict(t=30, b=10), xaxis=dict(categoryorder="array", categoryarray=ps["Period"].tolist()),
                          yaxis=dict(title="Service %", range=[max(0, float(ps["Actual_Pct"].min()) - 8), 101]),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="LRs"))
        st.plotly_chart(fig, width="stretch")
        show(ps[std_cols("Period")], f"period_{kind}")
    with t_mon:
        dim = st.selectbox("Rows", ["Source", "Destination", "Via", "Business", "Route"], key="mw_dim")
        metric = st.selectbox("Measure", ["Actual_Pct", "Variation_Pct", "Delay_Pct", "Volume", "Avg_Actual_Days"], key="mw_metric")
        ms_ = an.period_summary(df, "Monthly", by=dim)
        order = ms_.drop_duplicates("Period")["Period"].tolist()
        pv = ms_.pivot(index=dim, columns="Period", values=metric).reindex(columns=order)
        if metric in ("Variation_Pct",):
            fig = px.imshow(pv, color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto", template=TEMPLATE)
            fig.update_layout(height=min(900, 120 + 22 * len(pv)))
            st.plotly_chart(fig, width="stretch")
        show(pv.reset_index(), "month_wise", height=420)
    with t_cmp:
        preset = st.radio("Comparison", ["Current month vs previous month", "Last 30 days vs previous 30 days", "Custom"], horizontal=True, key="cmp_preset")
        last = ASOF.normalize()
        if preset.startswith("Current month"):
            a0 = last.replace(day=1)
            b1 = a0 - pd.Timedelta(days=1)
            a, b = (a0, last), (b1.replace(day=1), b1)
        elif preset.startswith("Last 30"):
            a, b = (last - pd.Timedelta(days=29), last), (last - pd.Timedelta(days=59), last - pd.Timedelta(days=30))
        else:
            c1, c2 = st.columns(2)
            ra = c1.date_input("Current period", (last - pd.Timedelta(days=13), last), key="cmp_a")
            rb = c2.date_input("Previous period", (last - pd.Timedelta(days=27), last - pd.Timedelta(days=14)), key="cmp_b")
            if len(ra) != 2 or len(rb) != 2:
                st.info("Select a start and end date for both periods.")
                st.stop()
            a, b = ra, rb
        st.caption(f"Current: {pd.Timestamp(a[0]):%d %b %Y} - {pd.Timestamp(a[1]):%d %b %Y}   |   Previous: {pd.Timestamp(b[0]):%d %b %Y} - {pd.Timestamp(b[1]):%d %b %Y}")
        by = st.selectbox("Compare by", ["Overall", "Source", "Destination", "Via", "Business", "Route"], key="cmp_by")
        cmp_df = apply_filters(D, use_date=False)
        cp = an.compare_periods(cmp_df, a, b, None if by == "Overall" else by)
        ordered = ([by] if by != "Overall" else ["Scope"]) + [f"{m}_{s}" for m in ["Volume", "Actual_Pct", "Target_Pct", "Variation_Pct", "Delay_Pct"] for s in ["Current", "Previous", "Change"]]
        if by != "Overall":
            cp = cp.sort_values("Variation_Pct_Change")
        show(cp[ordered], "period_compare", height=420)


# ==========================================================================
# Booking -> Main hub -> dispatch (section 6)
# ==========================================================================
elif page == "Booking to Main Hub":
    st.title("Booking → Main Hub → Dispatch")
    need_data()
    st.markdown(f'<div class="note">Targets: booking → main-hub receipt within <b>{rules["hub_receipt_target_hours"]:.0f}h</b>, '
                f'receipt → dispatch within <b>{rules["hub_dispatch_target_hours"]:.0f}h</b> (editable in Configuration).</div>', unsafe_allow_html=True)
    H = an.hub_movement(df, rules, ASOF)
    c = st.columns(5)
    for col, stg in zip(c[:4], an.HUB_STAGES):
        col.metric(stg, f"{int((H['Hub_Stage'] == stg).sum()):,}")
    c[4].metric("Delayed vs target", f"{100 * H['Hub_Delayed'].mean():.1f}%")

    pend = H[H["Hub_Stage"] != an.HUB_STAGES[3]]
    t_pend, t_src, t_hub, t_det = st.tabs(["Pending & ageing", "Source-wise", "Main-hub-wise", "LR detail"])
    with t_pend:
        ab = pend.groupby(["Hub_Stage", "Ageing_Bucket"], observed=False).size().rename("LRs").reset_index()
        fig = px.bar(ab, x="Ageing_Bucket", y="LRs", color="Hub_Stage", template=TEMPLATE, barmode="stack",
                     category_orders={"Ageing_Bucket": an.AGEING_LABELS}, title="Pending LRs by ageing")
        fig.update_layout(height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        stg = st.multiselect("Show stages", an.HUB_STAGES[:3], default=an.HUB_STAGES[:3], key="hub_stg")
        p = pend[pend["Hub_Stage"].isin(stg)].sort_values("Age_Hrs", ascending=False)
        show(p[["LR_No", "Source", "Main_Hub", "Booking_DT", "Hub_Receipt_DT", "Hub_Stage", "Status", "Age_Hrs", "Ageing_Bucket",
                "Hub_Delayed", "Delay_Reason_Disp", "Condition_Remark", "Operational_Remark"]], "hub_pending", height=380)
    with t_src:
        show(an.hub_summary(H, "Source"), "hub_by_source", height=420)
    with t_hub:
        hs = an.hub_summary(H, "Main_Hub")
        show(hs, "hub_by_hub", download=True)
        st.markdown("**Delay reasons at pre-hub stages, by main hub**")
        pre = df[df["Delayed"] & df["Delay_Point"].isin(["Booking to Collection", "Dispatch", "Main Hub"])]
        rr = an.reason_summary(pre, by="Main_Hub")
        show(rr, "hub_reasons", height=320)
    with t_det:
        cols = ["LR_No", "Booking_DT", "Source", "Main_Hub", "Hub_Receipt_DT", "Hub_Dispatch_DT", "Status", "Booking_to_Receipt_Hrs",
                "Receipt_to_Dispatch_Hrs", "Late_Receipt", "Late_Dispatch", "Delay_Reason_Disp", "Condition_Remark", "Operational_Remark"]
        only_late = st.checkbox("Only delayed", value=True, key="hub_late")
        det = H[H["Hub_Delayed"]] if only_late else H
        show(det.sort_values("Booking_DT", ascending=False).head(5000)[cols], "hub_detail", height=420)
        st.caption("Showing the latest 5,000 rows.")


# ==========================================================================
# Live operations (section 7)
# ==========================================================================
elif page == "Live Operations":
    st.title("Live Operations")
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f'<div class="note"><b>Feed:</b> {source_name}<br><b>As of:</b> {ASOF:%d %b %Y %H:%M}</div>', unsafe_allow_html=True)
    if isinstance(lv.get_source(), lv.SimulatedSource):
        st.info("Running on the simulated feed (local snapshot). Set SRD_API_BASE_URL (and SRD_API_TOKEN) to connect the SRD Logistics application's live APIs.")
    if source_err:
        st.error(f"Live API unavailable, using local snapshot instead - {source_err}")
    if c2.button("Refresh now", key="live_refresh"):
        get_raw.clear()
        clear_data_cache()
        st.rerun()

    live_df = apply_filters(D, use_date=False)
    need_data(live_df)
    R = lv.live_reports(live_df, ASOF)
    bk, dlv = R["bookings_today"], R["delivered_today"]
    c = st.columns(5)
    c[0].metric("Booked today", f"{len(bk):,}")
    c[1].metric("Weight booked today (T)", f"{bk['Weight_T'].sum():,.1f}")
    c[2].metric("Delivered today", f"{len(dlv):,}")
    c[3].metric("Target achieved (delivered today)", f"{100 * dlv['On_Track'].mean():.1f}%" if len(dlv) else "-")
    c[4].metric("Open LRs in network", f"{int((~live_df['Delivered']).sum()):,}")

    tabs = st.tabs(["Daily booking", "Lagging / leading", "Daily target achievement", "KT weight", "Godown stock", "Main hub movement"])
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            show(R["daily_booking"]["by_source"], "live_book_source", height=330)
        with b:
            show(R["daily_booking"]["by_destination"], "live_book_dest", height=330)
        hr = R["daily_booking"]["by_hour"]
        fig = px.bar(hr, x="Hour", y="LRs", template=TEMPLATE, color_discrete_sequence=[PRIMARY], title="Bookings by hour today")
        fig.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with tabs[1]:
        st.caption("Last 7 days service vs target (Leading = at/above target, Lagging = below) and today's bookings vs the usual for this weekday.")
        for key in ("source", "destination"):
            st.markdown(f"**{key.title()}-wise**")
            show(R[f"lag_lead_{key}"], f"live_lag_{key}", height=300)
    with tabs[2]:
        for key in ("source", "destination", "via"):
            st.markdown(f"**{key.title()}-wise (LRs delivered today)**")
            show(R[f"target_{key}"], f"live_target_{key}", height=260)
    with tabs[3]:
        st.caption("Total weight in tonnes booked today. Assumption: 'KT weight' = total LR weight; confirm the exact definition with SRD.")
        a, b = st.columns(2)
        with a:
            show(R["daily_booking"]["by_source"][["Source", "LRs", "Weight_T"]], "live_kt_source", height=330)
        with b:
            show(R["daily_booking"]["by_destination"][["Destination", "LRs", "Weight_T"]], "live_kt_dest", height=330)
    with tabs[4]:
        a, b = st.columns(2)
        with a:
            st.markdown("**Source godown** (booked / collected, not yet dispatched)")
            show(R["godown_source"], "live_godown_source", height=330)
        with b:
            st.markdown("**Destination godown** (arrived, not yet delivered)")
            show(R["godown_destination"], "live_godown_dest", height=330)
    with tabs[5]:
        show(R["hub_movement"], "live_hub_movement", download=True)
        fig = go.Figure()
        hm = R["hub_movement"]
        fig.add_bar(x=hm["Main_Hub"], y=hm["Received_Today"], name="Received today", marker_color=PRIMARY)
        fig.add_bar(x=hm["Main_Hub"], y=hm["Dispatched_Today"], name="Dispatched today", marker_color=ACCENT)
        fig.add_bar(x=hm["Main_Hub"], y=hm["Currently_At_Hub"], name="Currently at hub", marker_color=BAD)
        fig.update_layout(template=TEMPLATE, barmode="group", height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")


# ==========================================================================
# Improvement suggestions (section 8)
# ==========================================================================
elif page == "Improvement Suggestions":
    st.title("System-Generated Service Improvement Suggestions")
    need_data()
    basis_note()
    st.caption(f"Routes with at least {rules['min_route_volume']} LRs, below target, and below target in {rules['suggest_weeks_below']}+ of the last "
               f"{rules['suggest_weeks_window']} weeks (recurring). Thresholds are editable in Configuration.")
    sg = an.suggestions(df, rules)
    if sg.empty:
        st.success("No routes show recurring below-target performance in this selection.")
        st.stop()
    pr = st.multiselect("Priority", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"], key="sg_pr")
    sgv = sg[sg["Priority"].astype(str).isin(pr)] if pr else sg
    st.metric("Routes needing action", f"{len(sgv):,}")
    for _, r in sgv.head(6).iterrows():
        st.markdown(f'<div class="sug"><span class="pill pill-{r["Priority"]}">{r["Priority"]}</span> <b>{r["Route"]}</b> - '
                    f'{r["Actual_Pct"]:.1f}% vs target {r["Target_Pct"]:.1f}% ({r["Variation_Pct"]:+.1f} pts), below target {int(r["Weeks_Below_Target"])} of '
                    f'{int(r["Weeks_Observed"])} weeks<br>Major delay point: <b>{r["Delay_Point"]}</b> ({r["Point_Share_Pct"]:.0f}% of delays), '
                    f'top reason: <b>{r["Top_Reason"]}</b><br>&rarr; {r["Suggestion"]}</div>', unsafe_allow_html=True)
    show(sgv.drop(columns=["Impact_Score"]), "suggestions", height=380)

    st.subheader("Drill down: route → stage → LRs")
    route = st.selectbox("Route", sgv["Route"].tolist() or sg["Route"].tolist(), key="sg_route")
    rdf = df[df["Route"] == route]
    rsd = an.stage_delay_summary(rdf)
    show(rsd[rsd["Delayed_LRs"] > 0], "sg_stage", download=False)
    stages_ = rsd.loc[rsd["Delayed_LRs"] > 0, "Delay_Point"].tolist()
    if stages_:
        pt = st.selectbox("Delay point", stages_, key="sg_point")
        show(an.lr_list(rdf[rdf["Delay_Point"] == pt]), "sg_lrs", height=300)


# ==========================================================================
# Delay risk (ML)
# ==========================================================================
elif page == "Delay Risk (ML)":
    st.title("Delay Risk Prediction (ML)")
    need_data()
    payload = get_model(D, _sig(mdl.MODEL_PATH))
    m = payload["metrics"]
    st.markdown(f'<div class="note">Predicts whether an LR will miss its target days using booking-time information only '
                f'(route, hub/via, business, consignor, weight, booking time, target days). Model: <b>{payload["model_name"]}</b>. '
                f'Predictions support, not replace, operational judgement.</div>', unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c[1].metric("Recall", f"{m['recall']:.2f}")
    c[2].metric("Precision", f"{m['precision']:.2f}")
    c[3].metric("Base delay rate", f"{100 * payload['base_rate']:.1f}%")
    c[4].metric("Trained on", f"{payload['training_records']:,} LRs")

    t_imp, t_open, t_one = st.tabs(["What drives delay", "LRs at risk now", "Explain one LR"])
    with t_imp:
        imp = mdl.global_importance(payload)
        fig = px.bar(imp.head(12).iloc[::-1], x="Importance", y="Feature", orientation="h", template=TEMPLATE, color_discrete_sequence=[PRIMARY],
                     title="Permutation importance (drop in ROC-AUC when the feature is shuffled)")
        fig.update_layout(height=400, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(payload["all_model_metrics"]).T.reset_index().rename(columns={"index": "Model"}), hide_index=True, width="stretch")
    opn = df[~df["Delivered"]].copy()
    with t_open:
        if opn.empty:
            st.info("No open LRs in the current selection.")
        else:
            opn["Delay_Risk"] = mdl.score(payload, opn).round(3)
            opn["Risk_Level"] = opn["Delay_Risk"].map(mdl.risk_level)
            lv_sel = st.multiselect("Risk level", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"], key="ml_lv")
            view = opn[opn["Risk_Level"].isin(lv_sel)].sort_values("Delay_Risk", ascending=False)
            cc = st.columns(4)
            for col, lvl in zip(cc, ["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
                col.metric(f"{lvl} risk", f"{int((opn['Risk_Level'] == lvl).sum()):,}")
            show(view[["LR_No", "Route", "Via", "Business", "Consignor", "Status", "Booking_DT", "Target_Days", "Actual_Days", "Delayed",
                       "Delay_Risk", "Risk_Level"]].head(2000), "lr_risk", height=420)
    with t_one:
        pool = opn if not opn.empty else df
        options = pool.sort_values("Booking_DT", ascending=False)["LR_No"].head(500).tolist()
        pick = st.selectbox("LR (most recent open LRs)", options, key="ml_lr") if options else None
        if pick:
            row = D[D["LR_No"] == pick]
            p = float(mdl.score(payload, row)[0])
            st.metric("Predicted delay risk", f"{100 * p:.1f}%", mdl.risk_level(p))
            ex = mdl.explain_lr(payload, row)
            ex[["Value", "Typical"]] = ex[["Value", "Typical"]].astype(str)
            fig = px.bar(ex.iloc[::-1], x="Risk_Impact", y="Feature", orientation="h", color="Direction", template=TEMPLATE,
                         color_discrete_map={"Raises risk": BAD, "Lowers risk": GOOD}, title="Why: change in risk vs a typical LR")
            fig.update_layout(height=320, margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")
            show(ex, "lr_explain", download=False)


# ==========================================================================
# Configuration (section 9)
# ==========================================================================
elif page == "Configuration":
    st.title("Configuration")
    t_tgt, t_rules, t_defs = st.tabs(["Route targets", "Service rules", "Definitions & data"])
    with t_tgt:
        st.caption("Benchmark target days and target service % for every Source → Destination route. Edits are saved to data/srd_targets.csv "
                   "and apply everywhere. Use the sidebar 'Target days basis' to test other target days without saving.")
        tg = an.load_targets()
        c1, c2 = st.columns(2)
        fs = c1.multiselect("Source", sorted(tg["Source"].unique()), key="cf_src")
        fd = c2.multiselect("Destination", sorted(tg["Destination"].unique()), key="cf_dst")
        sub = tg
        if fs:
            sub = sub[sub["Source"].isin(fs)]
        if fd:
            sub = sub[sub["Destination"].isin(fd)]
        edited = st.data_editor(
            sub.reset_index(drop=True), hide_index=True, width="stretch", height=380, key="cf_editor",
            disabled=["Source", "Destination"],
            column_config={"Target_Days": st.column_config.NumberColumn("Target days", min_value=1, max_value=30, step=1),
                           "Target_Service_Pct": st.column_config.NumberColumn("Target service %", min_value=0.0, max_value=100.0, step=0.5)})
        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("Save targets", type="primary", key="cf_save"):
            full = tg.set_index(["Source", "Destination"])
            full.update(edited.set_index(["Source", "Destination"]))
            an.save_targets(full.reset_index())
            clear_data_cache()
            st.success(f"Saved {len(edited)} route target(s).")
            st.rerun()
        if b2.button("Reset to defaults", key="cf_reset"):
            an.save_targets(gen.default_targets())
            clear_data_cache()
            st.rerun()
    with t_rules:
        r = dict(rules)
        c1, c2, c3 = st.columns(3)
        r["default_target_days"] = c1.number_input("Default target days (routes without a target)", 1, 30, int(r["default_target_days"]))
        r["default_target_pct"] = c2.number_input("Default target service %", 0.0, 100.0, float(r["default_target_pct"]))
        r["min_route_volume"] = c3.number_input("Minimum LRs for route ranking / suggestions", 1, 1000, int(r["min_route_volume"]))
        c1, c2 = st.columns(2)
        r["hub_receipt_target_hours"] = c1.number_input("Booking → main hub receipt target (hours)", 1.0, 240.0, float(r["hub_receipt_target_hours"]))
        r["hub_dispatch_target_hours"] = c2.number_input("Main hub receipt → dispatch target (hours)", 1.0, 240.0, float(r["hub_dispatch_target_hours"]))
        c1, c2 = st.columns(2)
        r["min_stage_excess_hours"] = c1.number_input("Stage delay: minimum excess hours", 0.0, 48.0, float(r["min_stage_excess_hours"]))
        r["min_stage_excess_pct"] = c2.number_input("Stage delay: minimum excess % of route median", 0.0, 200.0, float(r["min_stage_excess_pct"]))
        c1, c2 = st.columns(2)
        r["suggest_weeks_window"] = c1.number_input("Suggestions: weeks looked back", 2, 26, int(r["suggest_weeks_window"]))
        r["suggest_weeks_below"] = c2.number_input("Suggestions: weeks below target to call it recurring", 1, 26, int(r["suggest_weeks_below"]))
        if st.button("Save rules", type="primary", key="rules_save"):
            an.save_rules(r)
            clear_data_cache()
            st.success("Rules saved.")
            st.rerun()
    with t_defs:
        st.markdown("""
**Definitions**
- **Actual days** - calendar days from booking date to delivery date (to the snapshot date for undelivered LRs).
- **Delayed / On-track** - an LR is delayed when actual days exceed target days; an undelivered LR is delayed once it has breached its target.
- **Actual service %** - on-track LRs / total LRs. **Variation %** - actual service % minus target service % (negative = below target).
- **Delay point** - the journey stage (Booking to Collection, Dispatch, Main Hub, Transit/Via, Final Arrival, Delivery) with the largest excess over that route's median stage time.
- **Delay reason / IF-condition / operational remark** - as recorded against the LR at the stage where the delay occurred.
- **Period reports** are based on the LR booking date; yearly = financial year (April-March).
        """)
        st.markdown("**LR data columns**")
        dd = pd.DataFrame({"Column": raw_df.columns, "Type": [str(t) for t in raw_df.dtypes],
                           "Example": [str(raw_df[c].dropna().iloc[0]) if raw_df[c].notna().any() else "" for c in raw_df.columns]})
        st.dataframe(dd, hide_index=True, width="stretch", height=300)
        st.caption(f"{len(raw_df):,} LRs booked {dmin:%d %b %Y} - {dmax:%d %b %Y}. Data is synthetic, generated by src/srd_generator.py "
                   f"(python src/srd_generator.py to regenerate). Replace data/srd_lr_data.csv, or set SRD_API_BASE_URL, to use real SRD data.")
