"""6G Network Performance vs Manufacturing Efficiency - analytics dashboard.

Thales Group smart-factory telemetry, 100,000 machine-minutes, 50 machines,
1 Jan - 10 Mar 2025.

Every panel reports an effect size and a confidence interval, not just a p-value,
because at n = 100,000 significance is cheap and the headline finding of this
study is a *null* result: latency and packet loss in this extract carry no
measurable relationship to efficiency, throughput, error rate or defect rate.
A null result is only credible when paired with the power to detect an effect,
so each panel also shows the smallest effect the current slice could have found.

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import src.data as D
import src.theme as T

st.set_page_config(page_title="6G Network vs Manufacturing Efficiency",
                   page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(T.CSS, unsafe_allow_html=True)

# responsive=True so charts re-fit when the browser window is resized rather
# than keeping the width they were first laid out at.
PLOT_CFG = {"displaylogo": False, "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# ---------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading telemetry…")
def get_data():
    return D.load()


df_all = get_data()


def table_view(frame, label="Table view (every plotted value)", **kw):
    """Every chart ships a table twin - values are never color- or hover-only."""
    with st.expander(label):
        st.dataframe(frame, width="stretch", **kw)


def note(txt):
    st.markdown(f'<div class="caption">{txt}</div>', unsafe_allow_html=True)


def tiles(specs):
    for col, (label, value, sub) in zip(st.columns(len(specs)), specs):
        col.markdown(T.tile(label, value, sub), unsafe_allow_html=True)


def power_line(n):
    r = D.min_detectable_r(n, 0.80)
    return (f"Slice n = {n:,}. At 80% power / α = .05 this slice could detect "
            f"|r| ≥ <b>{r:.4f}</b> (≈{100*r*r:.4f}% of variance). "
            f"Effects smaller than that are beyond the resolution of this data.")


# ------------------------------------------------------------------- sidebar
MODULES = [
    "Network performance overview",
    "Network vs efficiency",
    "Quality & error impact",
    "6G optimization insights",
    "Efficiency diagnostics",
    "Data & method",
]

# A radio rather than st.tabs, for two concrete reasons. (1) st.tabs renders
# every tab body on every rerun, so all six modules recompute on each filter
# change - roughly six times the necessary work on a 100k-row frame. (2) A chart
# first laid out inside an inactive tab has a zero-width container, so Plotly
# falls back to its 700px default and then overflows its column; neither
# autosize nor responsive recovers from it, because no resize event ever fires.
# Rendering one module at a time makes every chart lay out visible.
st.sidebar.markdown("### Module")
module = st.sidebar.radio("Dashboard module", MODULES, index=0,
                          label_visibility="collapsed", key="f_module")

st.sidebar.markdown("### Filters")
st.sidebar.caption("One control surface — every chart in every module re-renders "
                   "against this slice.")

dmin, dmax = df_all["Date_dt"].min().date(), df_all["Date_dt"].max().date()
preset = st.sidebar.radio(
    "Time window",
    ["Full period", "Last 7 days", "Last 14 days", "Last 30 days", "Custom"],
    index=0, key="f_window")
if preset == "Custom":
    picked = st.sidebar.date_input("Custom range", (dmin, dmax),
                                   min_value=dmin, max_value=dmax,
                                   key="f_range")
    date_range = picked if isinstance(picked, tuple) and len(picked) == 2 else (dmin, dmax)
else:
    days = {"Full period": None, "Last 7 days": 7, "Last 14 days": 14,
            "Last 30 days": 30}[preset]
    date_range = (dmin, dmax) if days is None else \
        (dmax - pd.Timedelta(days=days - 1), dmax)

hours = st.sidebar.slider("Hour of day", 0, 23, (0, 23), key="f_hours")
bands = st.sidebar.multiselect("Network quality band", T.BAND_ORDER,
                               default=T.BAND_ORDER, key="f_bands")
effs = st.sidebar.multiselect("Efficiency class", T.EFF_ORDER,
                              default=T.EFF_ORDER, key="f_effs")
modes = st.sidebar.multiselect("Operation mode",
                               ["Active", "Idle", "Maintenance"],
                               default=["Active", "Idle", "Maintenance"],
                               key="f_modes")
machine_opts = sorted(df_all["Machine_ID"].unique())
machines = st.sidebar.multiselect("Machine ID (blank = all)", machine_opts,
                                  default=[], key="f_machines")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Network Stability Index** — 0–100, higher is more stable. Equal-weight "
    "blend of latency and packet loss, each min–max normalised over the "
    "observed envelope (1–50 ms, 0–5 %) and inverted. Bands are NSI terciles "
    "of the full dataset."
)

df = D.apply_filters(df_all, date_range=date_range, bands=bands, effs=effs,
                     modes=modes, hours=hours,
                     machines=machines if machines else None)

# ---------------------------------------------------------------------- header
st.markdown("## 6G Network Performance vs Manufacturing Efficiency")
st.caption("Thales Group smart-factory telemetry · 50 machines · "
           "1 Jan – 10 Mar 2025 · 100,000 machine-minutes")

if df.empty:
    st.warning("No rows match the current filters. Widen the selection.")
    st.stop()

c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        f'<div class="hero">{len(df):,}</div>'
        f'<div class="hero-sub">machine-minutes in the active slice '
        f'({100*len(df)/len(df_all):.1f}% of the extract) · '
        f'{df["Machine_ID"].nunique()} machines · '
        f'{df["Date_dt"].nunique()} days</div>',
        unsafe_allow_html=True)
with c2:
    r_lat = D.corr_with_ci(df[D.LAT], df[D.SPD])
    r_pl = D.corr_with_ci(df[D.PL], df[D.ERR])
    st.markdown(T.verdict(
        "null",
        "<b>Headline finding — network performance shows no measurable effect.</b><br>"
        f"latency → production speed: r = {r_lat['r']:+.4f} "
        f"(95% CI {r_lat['lo']:+.4f} to {r_lat['hi']:+.4f}) · "
        f"packet loss → error rate: r = {r_pl['r']:+.4f} "
        f"(95% CI {r_pl['lo']:+.4f} to {r_pl['hi']:+.4f}). "
        "Both CIs straddle zero and exclude any practically relevant effect. "
        "Efficiency in this extract is fully determined by error rate and "
        "production speed — see the <i>Efficiency diagnostics</i> module."),
        unsafe_allow_html=True)

st.markdown("")

# ==================================================================== MODULE 1
if module == MODULES[0]:
    st.markdown("#### Network performance overview")

    nsi = df["NSI"]
    unstable = (df[D.LAT] > df_all[D.LAT].quantile(.90)) | \
               (df[D.PL] > df_all[D.PL].quantile(.90))
    tiles([
        ("Mean latency", f"{df[D.LAT].mean():.2f} ms",
         f"p95 {df[D.LAT].quantile(.95):.1f} ms · max {df[D.LAT].max():.1f} ms"),
        ("Mean packet loss", f"{df[D.PL].mean():.3f} %",
         f"p95 {df[D.PL].quantile(.95):.2f} % · max {df[D.PL].max():.2f} %"),
        ("Network Stability Index", f"{nsi.mean():.1f}",
         f"sd {nsi.std():.1f} · range {nsi.min():.0f}–{nsi.max():.0f}"),
        ("Degraded minutes", f"{100*unstable.mean():.1f} %",
         "latency or loss above the p90 of the full extract"),
        ("Latency jitter", f"{df[D.LAT].std():.2f} ms",
         f"IQR {df[D.LAT].quantile(.75)-df[D.LAT].quantile(.25):.1f} ms"),
    ])
    st.markdown("")

    # --- two separate charts: latency and packet loss are different scales.
    # A dual-axis plot would invent a correlation, so they never share an axis.
    daily = df.groupby("Date_dt").agg(
        lat_mean=(D.LAT, "mean"), lat_p95=(D.LAT, lambda s: s.quantile(.95)),
        loss_mean=(D.PL, "mean"), loss_p95=(D.PL, lambda s: s.quantile(.95)),
        nsi_mean=("NSI", "mean"), n=("NSI", "size")).reset_index()

    lc, rc = st.columns(2)

    with lc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date_dt"], y=daily["lat_p95"], name="daily p95",
            mode="lines", line=dict(color=T.SEQ_BLUE[3], width=2),
            hovertemplate="%{x|%d %b}<br>p95 %{y:.2f} ms<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=daily["Date_dt"], y=daily["lat_mean"], name="daily mean",
            mode="lines", line=dict(color=T.SEQ_BLUE[11], width=2),
            hovertemplate="%{x|%d %b}<br>mean %{y:.2f} ms<extra></extra>"))
        fig.add_hline(y=df[D.LAT].mean(), line=dict(color=T.AXIS, width=1),
                      annotation_text=f"period mean {df[D.LAT].mean():.1f} ms",
                      annotation_font=dict(size=10, color=T.MUTED),
                      annotation_position="top left")
        fig.update_layout(**T.layout(title="Latency trend — daily mean and p95",
                                     hovermode="x unified"))
        fig.update_yaxes(title_text="ms", title_font=dict(size=11, color=T.MUTED))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Flat within noise: no drift, no incident signature, no weekly "
             "seasonality. The series is stationary white noise.")

    with rc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date_dt"], y=daily["loss_p95"], name="daily p95",
            mode="lines", line=dict(color=T.SEQ_BLUE[3], width=2),
            hovertemplate="%{x|%d %b}<br>p95 %{y:.3f} %<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=daily["Date_dt"], y=daily["loss_mean"], name="daily mean",
            mode="lines", line=dict(color=T.SEQ_BLUE[11], width=2),
            hovertemplate="%{x|%d %b}<br>mean %{y:.3f} %<extra></extra>"))
        fig.add_hline(y=df[D.PL].mean(), line=dict(color=T.AXIS, width=1),
                      annotation_text=f"period mean {df[D.PL].mean():.2f} %",
                      annotation_font=dict(size=10, color=T.MUTED),
                      annotation_position="top left")
        fig.update_layout(**T.layout(title="Packet-loss trend — daily mean and p95",
                                     hovermode="x unified"))
        fig.update_yaxes(title_text="%", title_font=dict(size=11, color=T.MUTED))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Same picture. Neither KPI has a stable/unstable regime to segment — "
             "there are no identifiable network incidents in this extract.")

    table_view(daily.assign(Date_dt=daily["Date_dt"].dt.date)
                    .rename(columns={"Date_dt": "date", "n": "rows"}).round(4),
               "Table view — daily network KPIs")

    lc, rc = st.columns(2)
    with lc:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["NSI"], nbinsx=50, marker=dict(color=T.SEQ_BLUE[7]),
            hovertemplate="NSI %{x:.0f}<br>%{y:,} minutes<extra></extra>"))
        for q, lab in [(1 / 3, "Low│Medium"), (2 / 3, "Medium│High")]:
            fig.add_vline(x=df_all["NSI"].quantile(q),
                          line=dict(color=T.AXIS, width=1),
                          annotation_text=lab,
                          annotation_font=dict(size=10, color=T.MUTED))
        fig.update_layout(**T.layout(
            title="Network Stability Index distribution", showlegend=False,
            bargap=0.02))
        fig.update_xaxes(title_text="NSI (higher = more stable)",
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_yaxes(title_text="machine-minutes",
                         title_font=dict(size=11, color=T.MUTED))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Symmetric and broad — the sum of two independent uniforms. "
             "Real 6G telemetry is right-skewed with a heavy tail.")

    with rc:
        hourly = df.groupby("Hour").agg(
            lat=(D.LAT, "mean"), loss=(D.PL, "mean"), n=(D.LAT, "size")).reset_index()
        se = df.groupby("Hour")[D.LAT].sem()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly["Hour"], y=hourly["lat"] + 1.96 * se.values, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=hourly["Hour"], y=hourly["lat"] - 1.96 * se.values, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(42,120,214,0.15)", name="95% CI",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=hourly["Hour"], y=hourly["lat"], mode="lines+markers",
            line=dict(color=T.EMPHASIS, width=2),
            marker=dict(size=8, color=T.EMPHASIS,
                        line=dict(color=T.SURFACE, width=2)),
            name="mean latency",
            hovertemplate="%{x:02d}:00<br>mean %{y:.2f} ms<extra></extra>"))
        fig.update_layout(**T.layout(
            title="Mean latency by hour of day (95% CI)", hovermode="x unified"))
        fig.update_xaxes(title_text="hour", dtick=3,
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_yaxes(title_text="ms", title_font=dict(size=11, color=T.MUTED))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Every hour's CI overlaps every other. No shift-pattern or "
             "peak-load congestion effect exists in this data.")

    st.markdown("##### Machine stability scorecard")
    mach = df.groupby("Machine_ID").agg(
        minutes=("NSI", "size"), nsi_mean=("NSI", "mean"),
        lat_mean=(D.LAT, "mean"), lat_p95=(D.LAT, lambda s: s.quantile(.95)),
        loss_mean=(D.PL, "mean"), speed_mean=(D.SPD, "mean"),
        err_mean=(D.ERR, "mean"),
        pct_low=("Efficiency_Status", lambda s: (s.astype(str) == "Low").mean() * 100),
    ).round(3).sort_values("nsi_mean").reset_index()
    a, b = st.columns([3, 2])
    with a:
        st.dataframe(mach.rename(columns={
            "Machine_ID": "machine", "nsi_mean": "NSI", "lat_mean": "latency ms",
            "lat_p95": "latency p95", "loss_mean": "loss %",
            "speed_mean": "speed u/hr", "err_mean": "error %",
            "pct_low": "% low efficiency"}),
            width="stretch", height=300, hide_index=True)
    with b:
        rr = D.corr_with_ci(mach["nsi_mean"], mach["pct_low"])
        st.markdown(T.verdict("null",
            "<b>Machine-level cross-check.</b><br>"
            "If network quality drove efficiency, the machines with the worst "
            "average stability would carry the most low-efficiency minutes. "
            f"Across {len(mach)} machines that correlation is "
            f"r = {rr['r']:+.3f} (95% CI {rr['lo']:+.3f} to {rr['hi']:+.3f}, "
            f"p = {rr['p']:.3f}).<br>"
            f"Spread in % low efficiency is only "
            f"{mach['pct_low'].min():.1f}–{mach['pct_low'].max():.1f} "
            f"(sd {mach['pct_low'].std():.2f} pp) — consistent with binomial "
            "sampling noise around a common rate, not with per-machine "
            "network differences."), unsafe_allow_html=True)

# ==================================================================== MODULE 2
elif module == MODULES[1]:
    st.markdown("#### Network quality vs manufacturing efficiency")
    st.markdown(f'<div class="caption">{power_line(len(df))}</div>',
                unsafe_allow_html=True)

    ct = pd.crosstab(df["Network_Quality"].astype(str),
                     df["Efficiency_Status"].astype(str))
    ct = ct.reindex(index=[b for b in T.BAND_ORDER if b in ct.index],
                    columns=[e for e in T.EFF_ORDER if e in ct.columns],
                    fill_value=0)
    v, p_v = D.cramers_v(ct)

    low_share = (100 * ct.div(ct.sum(axis=1), axis=0))["Low"] \
        if "Low" in ct.columns else pd.Series(dtype=float)
    spread = f"{low_share.max() - low_share.min():.2f} pp" \
        if len(low_share) > 1 else "n/a"

    tiles([
        ("Cramér's V", f"{v:.4f}" if np.isfinite(v) else "n/a",
         "network band × efficiency · 0 = independent"),
        ("χ² p-value", f"{p_v:.3f}" if np.isfinite(p_v) else "n/a",
         "no evidence of association" if (np.isfinite(p_v) and p_v > .05)
         else "association detected"),
        ("Latency η²", f"{D.eta_squared(df, D.LAT):.4f} %",
         "variance in latency explained by efficiency class"),
        ("Packet-loss η²", f"{D.eta_squared(df, D.PL):.4f} %",
         "variance in loss explained by efficiency class"),
        ("Spread in % low", spread, "worst vs best network band"),
    ])
    st.markdown("")

    lc, rc = st.columns([3, 2])
    with lc:
        share = (100 * ct.div(ct.sum(axis=1), axis=0))
        fig = go.Figure()
        for cls in [c for c in T.EFF_ORDER if c in share.columns]:
            fig.add_trace(go.Bar(
                y=share.index, x=share[cls], name=f"{cls} efficiency",
                orientation="h", marker=dict(color=T.ORDINAL[cls],
                                             line=dict(color=T.SURFACE, width=2)),
                text=[f"{v:.1f}%" for v in share[cls]],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=11, color="#ffffff" if cls != "Low" else T.INK),
                hovertemplate="%{y} network<br>" + cls + " %{x:.2f}%<extra></extra>"))
        fig.update_layout(**T.layout(
            title="Efficiency mix by network-quality band", barmode="stack",
            height=320, bargap=0.35))
        fig.update_xaxes(title_text="% of machine-minutes", range=[0, 100],
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_yaxes(title_text="", showgrid=False)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("The three bars are the same bar. Segment labels are shown directly "
             "so identity never rests on colour alone.")

    with rc:
        st.markdown(T.verdict(
            "null" if (np.isfinite(p_v) and p_v > .05) else "real",
            "<b>Efficiency distribution is invariant to network quality.</b><br>"
            f"χ²({(ct.shape[0]-1)*(ct.shape[1]-1)}) test p = {p_v:.3f}, "
            f"Cramér's V = {v:.4f}. For scale, V &lt; 0.1 is conventionally "
            "\"negligible\"; this is an order of magnitude below that.<br><br>"
            "A real latency-sensitive line would show the Low-network band "
            "carrying visibly more low-efficiency minutes. Here the difference "
            "is a fraction of a percentage point in either direction."),
            unsafe_allow_html=True)

    table_view(ct.assign(**{"row total": ct.sum(axis=1)}),
               "Table view — counts by network band × efficiency class")

    st.markdown("##### Does efficiency degrade as latency rises?")
    lc, rc = st.columns(2)
    for col, driver in [(lc, D.LAT), (rc, D.PL)]:
        with col:
            bs = D.binned_share(df, driver, "Low", bins=12)
            if bs.empty:
                col.info("Not enough rows in this slice to bin.")
                continue
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bs["x"], y=bs["hi"], mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=bs["x"], y=bs["lo"], mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                name="95% CI (Wilson)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=bs["x"], y=bs["y"], mode="lines+markers",
                line=dict(color=T.EMPHASIS, width=2),
                marker=dict(size=8, color=T.EMPHASIS,
                            line=dict(color=T.SURFACE, width=2)),
                name="% low efficiency",
                hovertemplate="%{x:.2f}<br>%{y:.2f}% low"
                              "<extra></extra>"))
            overall = (df["Efficiency_Status"].astype(str) == "Low").mean() * 100
            fig.add_hline(y=overall, line=dict(color=T.AXIS, width=1),
                          annotation_text=f"slice mean {overall:.1f}%",
                          annotation_font=dict(size=10, color=T.MUTED),
                          annotation_position="bottom right")
            fig.update_layout(**T.layout(
                title=f"% low efficiency across {D.PRETTY[driver]} deciles",
                hovermode="x unified", height=330))
            fig.update_xaxes(title_text=D.PRETTY[driver],
                             title_font=dict(size=11, color=T.MUTED))
            fig.update_yaxes(title_text="% of minutes rated Low",
                             title_font=dict(size=11, color=T.MUTED))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            rng = bs["y"].max() - bs["y"].min()
            n_cover = int(((bs["lo"] <= overall) & (bs["hi"] >= overall)).sum())
            note(f"{n_cover} of {len(bs)} bin CIs contain the slice mean; total "
                 f"swing across the whole range is {rng:.2f} pp. A tolerance "
                 f"threshold would appear as a step or an elbow — there is "
                 f"neither.")
            table_view(bs.rename(columns={
                "x": f"mean {driver}", "y": "% low", "lo": "CI low",
                "hi": "CI high", "n": "minutes", "k": "low minutes"}).round(3),
                f"Table view — {D.PRETTY[driver]} deciles")

    st.markdown("##### Latency by efficiency class")
    lc, rc = st.columns([2, 3])
    with lc:
        fig = go.Figure()
        for cls in [c for c in T.EFF_ORDER if c in df["Efficiency_Status"].unique()]:
            s = df.loc[df["Efficiency_Status"].astype(str) == cls, D.LAT]
            fig.add_trace(go.Box(
                y=s, name=cls, marker_color=T.ORDINAL[cls], boxmean=True,
                line=dict(width=2), fillcolor="rgba(0,0,0,0)",
                showlegend=False, boxpoints=False,
                hovertemplate=cls + "<br>median %{median:.2f} ms<extra></extra>"))
        fig.update_layout(**T.layout(
            title="Latency distribution per efficiency class", height=330))
        fig.update_yaxes(title_text="latency (ms)",
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_xaxes(title_text="efficiency class", showgrid=False)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Identical boxes. Class labels sit on the axis, so the ordinal "
             "ramp is decoration, not the only encoding.")
    with rc:
        summ = df.groupby("Efficiency_Status", observed=True).agg(
            minutes=(D.LAT, "size"), lat_mean=(D.LAT, "mean"),
            lat_median=(D.LAT, "median"), lat_p95=(D.LAT, lambda s: s.quantile(.95)),
            loss_mean=(D.PL, "mean"), nsi_mean=("NSI", "mean")).round(3)
        st.dataframe(summ.rename(columns={
            "lat_mean": "latency mean", "lat_median": "latency median",
            "lat_p95": "latency p95", "loss_mean": "loss mean",
            "nsi_mean": "NSI mean"}), width="stretch")
        groups = [g[D.LAT].values for _, g in df.groupby("Efficiency_Status",
                                                         observed=True)]
        if len(groups) > 1:
            H, p_kw = stats.kruskal(*groups)
            st.markdown(T.verdict("null",
                f"<b>Kruskal–Wallis on latency across efficiency classes:</b> "
                f"H = {H:.2f}, p = {p_kw:.3f}. The three classes are drawn from "
                f"the same latency distribution. Mean latency differs by "
                f"{summ['lat_mean'].max()-summ['lat_mean'].min():.2f} ms "
                f"between the best and worst class — inside a 1–50 ms envelope."),
                unsafe_allow_html=True)

# ==================================================================== MODULE 3
elif module == MODULES[2]:
    st.markdown("#### Quality and error impact")
    st.markdown(f'<div class="caption">{power_line(len(df))}</div>',
                unsafe_allow_html=True)

    pairs = [(D.PL, D.ERR), (D.PL, D.DEF), (D.LAT, D.ERR), (D.LAT, D.DEF)]
    tiles([(f"{D.PRETTY[a].split(' (')[0]} → {D.PRETTY[b].split(' (')[0]}",
            f"r = {D.corr_with_ci(df[a], df[b])['r']:+.4f}",
            f"95% CI {D.corr_with_ci(df[a], df[b])['lo']:+.4f} to "
            f"{D.corr_with_ci(df[a], df[b])['hi']:+.4f}")
           for a, b in pairs])
    st.markdown("")

    lc, rc = st.columns(2)
    for col, (driver, outcome) in zip([lc, rc], [(D.PL, D.ERR), (D.PL, D.DEF)]):
        with col:
            bo = D.binned_outcome(df, driver, outcome, bins=12)
            if bo.empty:
                col.info("Not enough rows to bin in this slice.")
                continue
            st_ = D.corr_with_ci(df[driver], df[outcome])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bo["x"], y=bo["hi"], mode="lines",
                                     line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=bo["x"], y=bo["lo"], mode="lines",
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(42,120,214,0.15)",
                                     name="95% CI of the bin mean",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=bo["x"], y=bo["y"], mode="lines+markers",
                line=dict(color=T.EMPHASIS, width=2),
                marker=dict(size=8, color=T.EMPHASIS,
                            line=dict(color=T.SURFACE, width=2)),
                name=f"mean {D.PRETTY[outcome]}",
                hovertemplate="%{x:.2f}<br>mean %{y:.3f}<extra></extra>"))
            sl = stats.linregress(df[driver], df[outcome])
            xs = np.array([df[driver].min(), df[driver].max()])
            fig.add_trace(go.Scatter(
                x=xs, y=sl.intercept + sl.slope * xs, mode="lines",
                line=dict(color=T.DEEMPHASIS, width=2),
                name=f"OLS fit ({sl.slope:+.4f}/unit)",
                hovertemplate="OLS<extra></extra>"))
            fig.update_layout(**T.layout(
                title=f"{D.PRETTY[outcome]} vs {D.PRETTY[driver]}",
                hovermode="x unified", height=340))
            fig.update_xaxes(title_text=D.PRETTY[driver],
                             title_font=dict(size=11, color=T.MUTED))
            fig.update_yaxes(title_text=D.PRETTY[outcome],
                             title_font=dict(size=11, color=T.MUTED))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            note(f"r = {st_['r']:+.4f}, explaining {st_['r2_pct']:.4f}% of "
                 f"variance; OLS slope {sl.slope:+.5f} per unit "
                 f"(p = {sl.pvalue:.3f}). The fitted line is flat to the "
                 f"resolution of the axis.")
            table_view(bo.rename(columns={
                "x": f"mean {driver}", "y": f"mean {outcome}", "sd": "sd",
                "n": "minutes", "lo": "CI low", "hi": "CI high"}).round(4),
                f"Table view — {D.PRETTY[outcome]} by {D.PRETTY[driver]} decile")

    st.markdown("##### Do packet-loss spikes coincide with defect spikes?")
    lc, rc = st.columns([3, 2])
    with lc:
        q = st.select_slider(
            "Define a “spike” as the worst N% of minutes on each KPI",
            options=[1, 2, 5, 10, 20], value=5,
            help="Compares the spike population against everything else.")
        thr_pl = df[D.PL].quantile(1 - q / 100)
        thr_lat = df[D.LAT].quantile(1 - q / 100)
        rows = []
        for name, mask in [(f"packet loss ≥ {thr_pl:.2f}%", df[D.PL] >= thr_pl),
                           (f"latency ≥ {thr_lat:.1f} ms", df[D.LAT] >= thr_lat),
                           ("both simultaneously",
                            (df[D.PL] >= thr_pl) & (df[D.LAT] >= thr_lat))]:
            sp, rest = df[mask], df[~mask]
            if len(sp) < 5 or len(rest) < 5:
                continue
            for outcome in [D.ERR, D.DEF, D.SPD]:
                t, pv = stats.ttest_ind(sp[outcome], rest[outcome],
                                        equal_var=False)
                d = (sp[outcome].mean() - rest[outcome].mean()) / df[outcome].std()
                rows.append({"spike definition": name, "minutes": len(sp),
                             "outcome": D.PRETTY[outcome],
                             "spike mean": round(sp[outcome].mean(), 3),
                             "baseline mean": round(rest[outcome].mean(), 3),
                             "Δ": round(sp[outcome].mean() - rest[outcome].mean(), 3),
                             "Cohen's d": round(d, 4), "p": round(pv, 3)})
        spike = pd.DataFrame(rows)
        st.dataframe(spike, width="stretch", hide_index=True, height=330)
    with rc:
        if not spike.empty:
            worst = spike.loc[spike["Cohen's d"].abs().idxmax()]
            d_max = abs(worst["Cohen's d"])
            factor = 0.2 / d_max if d_max > 0 else float("inf")
            st.markdown(T.verdict("null",
                "<b>Spike analysis finds no degradation.</b><br>"
                f"Across all spike definitions the largest standardised effect "
                f"is Cohen's d = {worst['Cohen\'s d']:+.4f} "
                f"({worst['outcome']}, {worst['spike definition']}). "
                "d = 0.2 is the conventional floor for a \"small\" effect; "
                f"even the largest effect here is {factor:.0f}× below that "
                "floor.<br><br>"
                "Communication reliability thresholds cannot be derived from "
                "this extract, because degradation events produce no downstream "
                "signature to threshold against."), unsafe_allow_html=True)

    st.markdown("##### Error rate across the latency × packet-loss plane")
    lc, rc = st.columns([3, 2])
    with lc:
        metric = st.radio("Cell value", [D.ERR, D.DEF, D.SPD],
                          format_func=lambda c: D.PRETTY[c],
                          horizontal=True, key="heat_metric")
        nb = 8
        df_h = df.copy()
        df_h["lat_bin"] = pd.cut(df_h[D.LAT], nb)
        df_h["pl_bin"] = pd.cut(df_h[D.PL], nb)
        piv = df_h.pivot_table(index="pl_bin", columns="lat_bin", values=metric,
                               aggfunc="mean", observed=True)
        cnt = df_h.pivot_table(index="pl_bin", columns="lat_bin", values=metric,
                               aggfunc="size", observed=True)
        fig = go.Figure(go.Heatmap(
            z=piv.values,
            x=[f"{i.left:.0f}–{i.right:.0f}" for i in piv.columns],
            y=[f"{i.left:.2f}–{i.right:.2f}" for i in piv.index],
            colorscale=[[0, T.SEQ_BLUE[0]], [0.5, T.SEQ_BLUE[6]],
                        [1, T.SEQ_BLUE[12]]],
            colorbar=dict(title=dict(text=D.PRETTY[metric].split(" (")[0],
                                     font=dict(size=10, color=T.MUTED)),
                          tickfont=dict(size=10, color=T.MUTED), thickness=12,
                          outlinewidth=0),
            hovertemplate="latency %{x} ms<br>loss %{y} %<br>"
                          "mean %{z:.3f}<extra></extra>",
            xgap=2, ygap=2))
        fig.update_layout(**T.layout(
            title=f"Mean {D.PRETTY[metric]} by latency × packet-loss cell",
            showlegend=False, height=380))
        fig.update_xaxes(title_text="latency band (ms)", showgrid=False,
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_yaxes(title_text="packet-loss band (%)", showgrid=False,
                         title_font=dict(size=11, color=T.MUTED))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    with rc:
        vals = piv.values[np.isfinite(piv.values)]
        if vals.size:
            st.markdown(T.verdict("null",
                "<b>No gradient in either direction.</b><br>"
                f"Across the {piv.shape[0]}×{piv.shape[1]} grid, cell means of "
                f"{D.PRETTY[metric]} span {vals.min():.3f} to {vals.max():.3f} "
                f"(spread {vals.max()-vals.min():.3f}) against a within-cell "
                f"standard deviation of {df[metric].std():.3f}.<br><br>"
                "If communication quality mattered, this plane would darken "
                "toward the top-right corner where both KPIs are worst. It is "
                "flat noise; the visible variation is cell sampling error "
                f"(median {int(np.nanmedian(cnt.values))} minutes per cell)."),
                unsafe_allow_html=True)
        table_view(piv.round(3), "Table view — heatmap cell means")

# ==================================================================== MODULE 4
elif module == MODULES[3]:
    st.markdown("#### 6G optimization insights")

    st.markdown(T.verdict("info",
        "<b>What this module can and cannot deliver.</b> The brief asks for latency "
        "tolerance benchmarks and packet-loss risk zones. A benchmark is only "
        "meaningful if efficiency actually changes somewhere along the KPI range. "
        "This module runs that search exhaustively — every candidate threshold, with "
        "confidence intervals and a formal equivalence test — and reports what it "
        "finds. In this extract it finds no threshold, and the equivalence tests "
        "let us state that as a positive result rather than a failure to detect."),
        unsafe_allow_html=True)

    st.markdown("##### Latency tolerance search — every candidate threshold")
    cuts = np.arange(5, 50, 2.5)
    rows = []
    base_low = (df["Efficiency_Status"].astype(str) == "Low")
    for cut in cuts:
        above, below = df[D.LAT] >= cut, df[D.LAT] < cut
        if above.sum() < 30 or below.sum() < 30:
            continue
        ka, na = int(base_low[above].sum()), int(above.sum())
        kb, nb_ = int(base_low[below].sum()), int(below.sum())
        pa, pa_lo, pa_hi = D.wilson(ka, na)
        pb, pb_lo, pb_hi = D.wilson(kb, nb_)
        pv = D.chi2_p([[ka, na - ka], [kb, nb_ - kb]])
        rows.append({"threshold (ms)": cut, "minutes above": na,
                     "% low above": pa, "above CI low": pa_lo,
                     "above CI high": pa_hi, "% low below": pb,
                     "Δ (pp)": pa - pb, "χ² p": pv})
    scan = pd.DataFrame(rows)

    lc, rc = st.columns([3, 2])
    with lc:
        if scan.empty:
            st.info("Slice too small to scan thresholds.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scan["threshold (ms)"], y=scan["above CI high"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=scan["threshold (ms)"], y=scan["above CI low"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(42,120,214,0.15)",
                name="95% CI above threshold", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=scan["threshold (ms)"], y=scan["% low above"],
                mode="lines+markers", line=dict(color=T.EMPHASIS, width=2),
                marker=dict(size=8, color=T.EMPHASIS,
                            line=dict(color=T.SURFACE, width=2)),
                name="% low efficiency above threshold",
                hovertemplate="≥ %{x:.1f} ms<br>%{y:.2f}% low<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=scan["threshold (ms)"], y=scan["% low below"], mode="lines",
                line=dict(color=T.DEEMPHASIS, width=2),
                name="% low efficiency below threshold",
                hovertemplate="&lt; %{x:.1f} ms<br>%{y:.2f}% low<extra></extra>"))
            fig.update_layout(**T.layout(
                title="Low-efficiency rate above vs below each latency threshold",
                hovermode="x unified", height=360))
            fig.update_xaxes(title_text="candidate latency threshold (ms)",
                             title_font=dict(size=11, color=T.MUTED))
            fig.update_yaxes(title_text="% of minutes rated Low",
                             title_font=dict(size=11, color=T.MUTED))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            note("The two lines are indistinguishable at every threshold. A real "
                 "tolerance limit would show them diverging past a specific ms "
                 "value.")
    with rc:
        if not scan.empty:
            worst = scan.loc[scan["Δ (pp)"].abs().idxmax()]
            evaluable = int(scan["χ² p"].notna().sum())
            nsig = int((scan["χ² p"] < .05).sum())
            sig_txt = (f"{nsig} of {evaluable} evaluable thresholds reach "
                       f"p &lt; .05 (expected by chance alone: "
                       f"{0.05*evaluable:.1f})."
                       if evaluable else
                       "No threshold yields a testable contingency table in "
                       "this slice — the efficiency filter leaves only one "
                       "class, so there is nothing to compare.")
            st.markdown(T.verdict("null",
                "<b>No latency tolerance benchmark is derivable.</b><br>"
                f"{len(scan)} thresholds tested from "
                f"{scan['threshold (ms)'].min():.1f} to "
                f"{scan['threshold (ms)'].max():.1f} ms. The largest gap in "
                f"low-efficiency rate between the above- and below-threshold "
                f"populations is <b>{worst['Δ (pp)']:+.2f} pp</b> at "
                f"{worst['threshold (ms)']:.1f} ms — well inside sampling "
                f"noise. {sig_txt}<br><br>"
                "<b>Practical reading:</b> across the full 1–50 ms envelope "
                "present in this data, no latency budget separates good "
                "production from bad. Any \"tolerance number\" published from "
                "this extract would be an artefact."), unsafe_allow_html=True)
    if not scan.empty:
        table_view(scan.round(4), "Table view — latency threshold scan")

    st.markdown("##### Packet-loss risk zones")
    lc, rc = st.columns([3, 2])
    with lc:
        zones = [(0, 1, "0–1 % · nominal"), (1, 2, "1–2 % · elevated"),
                 (2, 3, "2–3 % · degraded"), (3, 4, "3–4 % · poor"),
                 (4, 5.01, "4–5 % · severe")]
        rows = []
        for lo, hi, lab in zones:
            s = df[(df[D.PL] >= lo) & (df[D.PL] < hi)]
            if len(s) < 30:
                continue
            k = int((s["Efficiency_Status"].astype(str) == "Low").sum())
            p, cl, ch = D.wilson(k, len(s))
            rows.append({"risk zone": lab, "minutes": len(s), "% low": p,
                         "CI low": cl, "CI high": ch,
                         "mean error %": s[D.ERR].mean(),
                         "mean defect %": s[D.DEF].mean(),
                         "mean speed": s[D.SPD].mean()})
        zdf = pd.DataFrame(rows)
        if zdf.empty:
            st.info("Slice too small for zone analysis.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=zdf["risk zone"], y=zdf["% low"],
                marker=dict(color=T.SEQ_BLUE[7],
                            line=dict(color=T.SURFACE, width=2)),
                error_y=dict(type="data", symmetric=False,
                             array=zdf["CI high"] - zdf["% low"],
                             arrayminus=zdf["% low"] - zdf["CI low"],
                             color=T.INK_2, thickness=1.5, width=6),
                text=[f"{v:.1f}%" for v in zdf["% low"]],
                textposition="outside", textfont=dict(size=11, color=T.INK_2),
                hovertemplate="%{x}<br>%{y:.2f}% low efficiency<extra></extra>",
                showlegend=False))
            fig.add_hline(y=(df["Efficiency_Status"].astype(str) == "Low").mean()*100,
                          line=dict(color=T.AXIS, width=1),
                          annotation_text="slice mean",
                          annotation_font=dict(size=10, color=T.MUTED))
            fig.update_layout(**T.layout(
                title="Low-efficiency rate by packet-loss risk zone (95% CI)",
                showlegend=False, height=360, bargap=0.45))
            fig.update_yaxes(title_text="% of minutes rated Low",
                             title_font=dict(size=11, color=T.MUTED))
            fig.update_xaxes(title_text="", showgrid=False)
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            note("Error bars are Wilson 95% intervals. All five zones overlap "
                 "the slice mean — the zones carry no differential risk.")
    with rc:
        if not zdf.empty:
            zlo, zhi = zdf["% low"].min(), zdf["% low"].max()
            st.markdown(T.verdict("null",
                "<b>Risk zones do not separate.</b><br>"
                f"From nominal (0–1 %) to severe (4–5 %) packet loss, the "
                f"low-efficiency rate stays inside {zlo:.2f}–{zhi:.2f}% — a "
                f"total spread of {zhi - zlo:.2f} pp, and not monotone in "
                "packet loss. Every confidence interval overlaps.<br><br>"
                "Mean error rate across zones: "
                f"{zdf['mean error %'].min():.2f}–{zdf['mean error %'].max():.2f}%. "
                "Mean defect rate: "
                f"{zdf['mean defect %'].min():.2f}–{zdf['mean defect %'].max():.2f}%."),
                unsafe_allow_html=True)
            table_view(zdf.round(3), "Table view — packet-loss risk zones")

    st.markdown("##### Formal equivalence testing (TOST)")
    st.caption("A non-significant p-value only means \"not detected\". TOST "
               "reverses the burden of proof: a small TOST p-value lets us "
               "positively conclude the effect is smaller than a bound we care "
               "about. Bound here: |r| < 0.05, i.e. under 0.25 % of variance.")
    rows = []
    for a in [D.LAT, D.PL]:
        for b in [D.SPD, D.ERR, D.DEF, D.PMS]:
            c = D.corr_with_ci(df[a], df[b])
            pt = D.tost_r(c["r"], c["n"])
            rows.append({"network driver": D.PRETTY[a], "outcome": D.PRETTY[b],
                         "r": round(c["r"], 5),
                         "95% CI": f"[{c['lo']:+.4f}, {c['hi']:+.4f}]",
                         "% variance": round(c["r2_pct"], 4),
                         "p (H₀: r = 0)": round(c["p"], 3),
                         "p (TOST: |r| < 0.05)": f"{pt:.2e}" if np.isfinite(pt) else "n/a",
                         "equivalent to zero": "yes" if (np.isfinite(pt) and pt < .05) else "no"})
    tost = pd.DataFrame(rows)
    st.dataframe(tost, width="stretch", hide_index=True)
    neq = (tost["equivalent to zero"] == "yes").sum()
    st.markdown(T.verdict("null" if neq == len(tost) else "info",
        f"<b>{neq} of {len(tost)}</b> network→outcome relationships are "
        "statistically <i>equivalent to zero</i> at the |r| &lt; 0.05 bound. "
        "This is the strongest form the null result can take: not \"we failed to "
        "find an effect\", but \"we can rule out any effect large enough to "
        "matter\"."), unsafe_allow_html=True)

    st.markdown("##### Where the optimization budget should actually go")
    st.markdown(T.verdict("real",
        "<b>Evidence-based priorities from this extract.</b><br>"
        "1. <b>Do not fund latency reduction on the strength of this data.</b> "
        "Between 1 ms and 50 ms there is no measurable production benefit. The "
        "business case for network spend must come from instrumented A/B trials "
        "or from telemetry that captures the closed control loop, not from this "
        "extract.<br>"
        "2. <b>Target error rate.</b> It is the dominant discriminator between "
        "efficiency classes — class membership accounts for 38.4 % of the "
        "variance in error rate, against under 0.01 % for latency — and it "
        "gates every High rating (High requires error rate ≤ 2 %).<br>"
        "3. <b>Target throughput.</b> Production speed is the second "
        "discriminator (η² = 11.3 %); the Medium/High boundary sits at "
        "400 units/hr and the Low/Medium boundary at 200 units/hr.<br>"
        "4. <b>Fix the measurement chain before the network.</b> The reason "
        "network effects are invisible here is that this extract carries no "
        "causal link between the two — see <i>Data &amp; method</i>. Re-instrument "
        "with paired network/production timestamps before re-running this study."),
        unsafe_allow_html=True)

# ==================================================================== MODULE 5
elif module == MODULES[4]:
    st.markdown("#### Efficiency diagnostics — what actually drives the label")
    st.markdown(T.verdict("real",
        "<b>Efficiency_Status is a deterministic function of two columns.</b> "
        "A depth-4 decision tree on error rate and production speed alone "
        "reproduces the label for <b>99,998 of 100,000 rows (99.998 %)</b>. The "
        "two exceptions sit exactly on the error-rate = 5.000 boundary. This is "
        "not a model — it is the recovered labelling rule:<br><br>"
        "&nbsp;&nbsp;<code>High</code> &nbsp; if error rate ≤ 2 % <b>and</b> speed &gt; 400 units/hr<br>"
        "&nbsp;&nbsp;<code>Medium</code> if error rate ≤ 5 % <b>and</b> speed &gt; 200 units/hr<br>"
        "&nbsp;&nbsp;<code>Low</code> &nbsp;&nbsp;otherwise<br><br>"
        "Network KPIs are absent from the rule, which is the mechanical reason "
        "every network panel in this dashboard reads flat."),
        unsafe_allow_html=True)

    st.markdown("##### Variance in each feature explained by efficiency class (η²)")
    lc, rc = st.columns([3, 2])
    with lc:
        eta = pd.Series({c: D.eta_squared(df, c) for c in D.NUMERIC}) \
                .sort_values(ascending=True)
        drivers = {D.ERR, D.SPD}
        # Emphasis form: the two real drivers carry the accent hue, the seven
        # non-drivers recede to gray. Colour follows the entity, not the rank.
        colors = [T.EMPHASIS if c in drivers else T.DEEMPHASIS for c in eta.index]
        fig = go.Figure(go.Bar(
            x=eta.values, y=[D.PRETTY[c] for c in eta.index], orientation="h",
            marker=dict(color=colors, line=dict(color=T.SURFACE, width=2)),
            text=[f"{v:.3f}%" if v < 1 else f"{v:.1f}%" for v in eta.values],
            textposition="outside", textfont=dict(size=11, color=T.INK_2),
            hovertemplate="%{y}<br>η² = %{x:.4f}%<extra></extra>",
            showlegend=False))
        fig.update_layout(**T.layout(
            title="η² — % of each feature's variance explained by efficiency class",
            showlegend=False, height=420, bargap=0.4))
        fig.update_xaxes(title_text="η² (%), log scale", type="log",
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Log x-axis — a linear axis would render the seven non-drivers as "
             "invisible slivers. Every bar is directly labelled, so the two "
             "highlighted bars are not the only cue.")
    with rc:
        st.dataframe(
            eta.sort_values(ascending=False).round(5).rename("η² (%)")
               .rename_axis("feature").reset_index()
               .assign(feature=lambda d: d["feature"].map(D.PRETTY)),
            width="stretch", hide_index=True, height=360)
        st.markdown(T.verdict("null",
            "Error rate and production speed are separated from every other "
            "feature by roughly <b>four orders of magnitude</b>. Latency and "
            "packet loss sit alongside temperature and vibration at η² &lt; "
            "0.01 % — indistinguishable from noise."), unsafe_allow_html=True)

    st.markdown("##### The decision boundary in error-rate × speed space")
    lc, rc = st.columns([3, 2])
    with lc:
        n_pts = min(6000, len(df))
        samp = df.sample(n_pts, random_state=0)
        fig = go.Figure()
        for cls in [c for c in T.EFF_ORDER if c in samp["Efficiency_Status"].unique()]:
            s = samp[samp["Efficiency_Status"].astype(str) == cls]
            fig.add_trace(go.Scattergl(
                x=s[D.ERR], y=s[D.SPD], mode="markers", name=f"{cls} efficiency",
                marker=dict(size=4, color=T.ORDINAL[cls], opacity=0.55),
                hovertemplate=cls + "<br>error %{x:.2f}%<br>"
                              "speed %{y:.0f} u/hr<extra></extra>"))
        for xv in (2.0, 5.0):
            fig.add_vline(x=xv, line=dict(color=T.INK_2, width=1))
        for yv in (200.0, 400.0):
            fig.add_hline(y=yv, line=dict(color=T.INK_2, width=1))
        fig.add_annotation(x=1.0, y=450, text="High", showarrow=False,
                           font=dict(size=13, color=T.INK))
        fig.add_annotation(x=3.5, y=300, text="Medium", showarrow=False,
                           font=dict(size=13, color=T.INK))
        fig.add_annotation(x=10, y=125, text="Low", showarrow=False,
                           font=dict(size=13, color=T.INK))
        fig.update_layout(**T.layout(
            title=f"Recovered decision boundary ({n_pts:,}-point sample)",
            height=430))
        fig.update_xaxes(title_text=D.PRETTY[D.ERR],
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_yaxes(title_text=D.PRETTY[D.SPD],
                         title_font=dict(size=11, color=T.MUTED))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Three axis-aligned rectangles with no overlap and no fuzzy margin. "
             "Region names are annotated directly on the plot.")
    with rc:
        pred = D.efficiency_rule(df[D.ERR].values, df[D.SPD].values)
        acc = (pred == df["Efficiency_Status"].astype(str).values).mean()
        st.markdown(T.verdict("real",
            f"<b>Rule accuracy on the active slice: {100*acc:.3f} %</b> "
            f"({int((pred != df['Efficiency_Status'].astype(str).values).sum())} "
            f"mismatches in {len(df):,} rows).<br><br>"
            "A boundary this crisp is the signature of a generated label, not a "
            "measured business outcome. Real efficiency ratings have overlapping "
            "class distributions because they depend on unrecorded factors."),
            unsafe_allow_html=True)
        st.markdown("**Rule vs actual label**")
        st.dataframe(pd.crosstab(df["Efficiency_Status"].astype(str),
                                 pd.Series(pred, index=df.index),
                                 rownames=["actual"], colnames=["rule"]),
                     width="stretch")

    st.markdown("##### Model benchmarks — the honest predictability ceiling")
    bench = pd.DataFrame([
        {"feature set": "Majority class only (predict “Low”)",
         "accuracy": 0.7782, "balanced accuracy": 0.3333,
         "reading": "the baseline every model must beat"},
        {"feature set": "Latency + packet loss only",
         "accuracy": 0.7782, "balanced accuracy": 0.3333,
         "reading": "exactly the baseline — zero information added"},
        {"feature set": "All 9 features, target shuffled",
         "accuracy": 0.7782, "balanced accuracy": 0.3333,
         "reading": "permutation control — matches the network-only model"},
        {"feature set": "Error rate + production speed",
         "accuracy": 0.9983, "balanced accuracy": 0.9931,
         "reading": "two columns recover almost the whole label"},
        {"feature set": "All 9 features",
         "accuracy": 0.9984, "balanced accuracy": 0.9941,
         "reading": "the other 7 features add +0.01 pp"},
    ])
    lc, rc = st.columns([3, 2])
    with lc:
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        order = bench.sort_values("balanced accuracy")
        cols = [T.EMPHASIS if v > .5 else T.DEEMPHASIS
                for v in order["balanced accuracy"]]
        fig.add_trace(go.Bar(
            x=order["balanced accuracy"] * 100, y=order["feature set"],
            orientation="h",
            marker=dict(color=cols, line=dict(color=T.SURFACE, width=2)),
            text=[f"{v*100:.2f}%" for v in order["balanced accuracy"]],
            textposition="outside", textfont=dict(size=11, color=T.INK_2),
            hovertemplate="%{y}<br>balanced accuracy %{x:.2f}%<extra></extra>",
            showlegend=False))
        fig.add_vline(x=100 / 3, line=dict(color=T.AXIS, width=1),
                      annotation_text="chance (33.3%)",
                      annotation_font=dict(size=10, color=T.MUTED))
        fig.update_layout(**T.layout(
            title="Balanced accuracy by feature set (3-class, chance = 33.3%)",
            showlegend=False, height=330, bargap=0.4,
            margin=dict(l=240, r=70, t=48, b=48)))
        fig.update_xaxes(title_text="balanced accuracy (%)", range=[0, 112],
                         title_font=dict(size=11, color=T.MUTED))
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Balanced accuracy, not raw accuracy — with 77.8 % of rows in one "
             "class, raw accuracy makes a useless model look 78 % correct. The "
             "network-only model lands exactly on chance.")
    with rc:
        st.dataframe(bench, width="stretch", hide_index=True, height=300)
        st.caption("Held-out 25 % test split, stratified, "
                   "HistGradientBoostingClassifier, seed 0. Figures are fixed "
                   "benchmarks from `analysis/02–03`, computed on the full "
                   "dataset — they do not re-fit on the filtered slice.")

    st.markdown("##### Feature independence")
    cm = df[D.NUMERIC].corr()
    lc, rc = st.columns([3, 2])
    with lc:
        mask = np.triu(np.ones_like(cm, dtype=bool))
        z = cm.mask(mask)
        fig = go.Figure(go.Heatmap(
            z=z.values, x=[D.PRETTY[c].split(" (")[0] for c in cm.columns],
            y=[D.PRETTY[c].split(" (")[0] for c in cm.index],
            zmid=0, zmin=-0.05, zmax=0.05,
            colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#d03b3b"]],
            colorbar=dict(title=dict(text="Pearson r",
                                     font=dict(size=10, color=T.MUTED)),
                          tickfont=dict(size=10, color=T.MUTED), thickness=12,
                          outlinewidth=0),
            hovertemplate="%{y} × %{x}<br>r = %{z:.4f}<extra></extra>",
            xgap=2, ygap=2))
        fig.update_layout(**T.layout(
            title="Correlation matrix, scale clipped to ±0.05",
            showlegend=False, height=460,
            margin=dict(l=170, r=24, t=48, b=140)))
        fig.update_xaxes(showgrid=False, tickangle=-40)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        note("Diverging blue↔red on a neutral gray midpoint, because the sign of "
             "r is the point. The scale is clipped to ±0.05 — on a full −1…+1 "
             "scale every cell would be identical gray.")
    with rc:
        off = cm.where(~np.eye(len(cm), dtype=bool)).abs()
        st.markdown(T.verdict("null",
            "<b>The features are mutually independent.</b><br>"
            f"Largest absolute off-diagonal correlation among all "
            f"{len(cm)*(len(cm)-1)//2} pairs: <b>{off.max().max():.4f}</b>. "
            "In real factory telemetry, temperature, vibration and power "
            "consumption are strongly coupled by physics — a machine drawing "
            "more power runs hotter and vibrates more. Their independence here "
            "is itself evidence that the columns were drawn separately."),
            unsafe_allow_html=True)
        table_view(cm.round(4), "Table view — full correlation matrix")

# ==================================================================== MODULE 6
elif module == MODULES[5]:
    st.markdown("#### Data & method")

    st.markdown("##### Dataset integrity audit")
    tiles([
        ("Rows", f"{len(df_all):,}", "1 Jan – 10 Mar 2025, 1-minute cadence"),
        ("Missing cells", f"{int(df_all[D.NUMERIC].isna().sum().sum())}",
         "no imputation was required"),
        ("Duplicate rows", "0", "full-row duplicates"),
        ("Duplicate keys", "28",
         "(timestamp, machine) pairs appearing twice"),
        ("Class imbalance", "77.8 / 19.2 / 3.0 %", "Low / Medium / High"),
    ])
    st.markdown("")

    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Issues found, and how they were handled**")
        st.markdown("""
| Issue | Evidence | Handling |
|---|---|---|
| `2025-03-01` holds **2,880 rows** instead of 1,440 | 1,440 timestamps each appear twice | Retained; the duplicated block does not bias any aggregate reported here, and dropping it would silently discard 1.4 % of the extract |
| `2025-03-10` is partial (**640 rows**, ends 10:39) | extract cut mid-day | Retained; all trend charts plot daily means, which are unaffected by day length |
| 28 duplicate `(timestamp, machine_id)` pairs | key-uniqueness check | Flagged, not dropped — 0.028 % of rows |
| `Efficiency_Status` is **77.8 % “Low”** | value counts | Balanced accuracy used throughout instead of raw accuracy |
| Column named `Timestamp`, not `Time` as in the brief | schema comparison | Read as-is; combined with `Date` into a proper datetime |
        """)
    with rc:
        st.markdown("**Distributional red flags**")
        uni = pd.DataFrame({
            "feature": [D.PRETTY[c] for c in D.NUMERIC],
            "mean": [df_all[c].mean() for c in D.NUMERIC],
            "sd": [df_all[c].std() for c in D.NUMERIC],
            "skew": [df_all[c].skew() for c in D.NUMERIC],
            "excess kurtosis": [df_all[c].kurtosis() for c in D.NUMERIC],
            "KS p vs Uniform": [
                stats.kstest((df_all[c] - df_all[c].min()) /
                             (df_all[c].max() - df_all[c].min()), "uniform").pvalue
                for c in D.NUMERIC],
        }).round(4)
        st.dataframe(uni, width="stretch", hide_index=True, height=360)
        st.markdown(T.verdict("null",
            "<b>All nine numeric features are uniformly distributed.</b> "
            "Skew ≈ 0 and excess kurtosis ≈ −1.2 is the exact signature of a "
            "uniform distribution (−1.2 is its theoretical value); a "
            "Kolmogorov–Smirnov test fails to reject uniformity for eight of "
            "nine features. Physical measurements are never uniform — "
            "temperature clusters around a setpoint, latency is right-skewed "
            "with a tail."), unsafe_allow_html=True)

    st.markdown("##### Why the null result is a property of the data, not of the analysis")
    st.markdown(T.verdict("info",
        "Four independent lines of evidence converge:<br><br>"
        "<b>1. Power.</b> At n = 100,000 the study could detect |r| ≥ 0.0089 at "
        "80 % power. The largest observed network→outcome correlation is 0.0071. "
        "The analysis is not underpowered — the effects are absent.<br>"
        "<b>2. Equivalence.</b> All eight network→outcome pairs pass TOST at the "
        "|r| &lt; 0.05 bound with p &lt; 10⁻⁴¹. We can positively assert absence, "
        "not merely fail to detect presence.<br>"
        "<b>3. Model ceiling.</b> Gradient boosting on latency + packet loss "
        "achieves 33.33 % balanced accuracy on a 3-class problem — exactly "
        "chance, and identical to the same model trained on a shuffled target.<br>"
        "<b>4. Construction.</b> The label is a deterministic rule over two "
        "other columns, every feature is uniform, and all pairwise correlations "
        "are &lt; 0.008. This is independently generated synthetic data; no "
        "network→production mechanism was encoded in it to be found."),
        unsafe_allow_html=True)

    st.markdown("##### What this means for the research question")
    st.markdown("""
The brief asks how much network performance affects production efficiency. The
defensible answer from this extract is: **not measurably, and the reason is that
this extract cannot answer the question.** That distinction matters, because the
wrong conclusion — *"6G latency does not affect manufacturing"* — is a
substantive claim about the physical world that this data does not support
either. The finding is about the dataset.

**To answer the question properly, the following would be required:**

1. **Paired causal timing.** Network KPIs must be measured on the same control
   loop, and timestamped ahead of, the production outcome they are claimed to
   affect. Cross-sectional per-minute snapshots cannot separate cause from
   coincidence even when a correlation exists.
2. **Real variation in the driver.** A uniform 1–50 ms sweep is a synthetic
   design, not observed 6G behaviour. Real deployments show bursty tail latency,
   and it is the tail — p99, not the mean — that breaks closed-loop control.
3. **A mechanism-bearing outcome.** Efficiency here is a threshold on error rate
   and speed. A network study needs outcomes the network can physically touch:
   control-loop deadline misses, command retransmissions, cycle-time jitter,
   AGV re-route counts.
4. **An intervention.** Network slicing configuration is controllable, so the
   study can be an experiment. Deliberately vary the slice SLA per production
   cell and measure the difference. That converts an inconclusive correlational
   study into a causal one.
    """)

    st.markdown("##### Reproducing this dashboard")
    st.code("""# 1. environment
python -m venv .venv
.venv\\Scripts\\activate            # Windows
pip install -r requirements.txt

# 2. analysis pipeline (writes outputs/, including analysis_ready.parquet)
python analysis/01_profile.py        # structure, missingness, distributions
python analysis/02_signal_test.py    # correlation / chi2 / MI / model ceiling
python analysis/03_label_rule.py     # recovers the labelling rule
python analysis/04_kpis_and_power.py # KPIs, power, TOST, artifact export

# 3. tests (40 headless checks across all six modules)
python -m pytest tests -q

# 4. dashboard
streamlit run app.py""", language="bash")
    st.caption("The dashboard reads `outputs/analysis_ready.parquet` when "
               "present and falls back to the raw CSV otherwise, so it runs "
               "before the pipeline has been executed.")
