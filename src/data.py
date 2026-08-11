"""Data loading, derived features and statistics helpers for the dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "Thales_Group_Manufacturing.csv"
PARQUET = ROOT / "outputs" / "analysis_ready.parquet"

LAT = "Network_Latency_ms"
PL = "Packet_Loss_%"
DEF = "Quality_Control_Defect_Rate_%"
SPD = "Production_Speed_units_per_hr"
ERR = "Error_Rate_%"
PMS = "Predictive_Maintenance_Score"

NUMERIC = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", LAT, PL,
           DEF, SPD, PMS, ERR]
OUTCOMES = [SPD, ERR, DEF]

PRETTY = {
    LAT: "Network latency (ms)", PL: "Packet loss (%)",
    DEF: "Defect rate (%)", SPD: "Production speed (units/hr)",
    ERR: "Error rate (%)", PMS: "Predictive maintenance score",
    "Temperature_C": "Temperature (°C)", "Vibration_Hz": "Vibration (Hz)",
    "Power_Consumption_kW": "Power consumption (kW)",
}


def load() -> pd.DataFrame:
    """Load telemetry, preferring the pre-built parquet, and add derived fields."""
    if PARQUET.exists():
        df = pd.read_parquet(PARQUET)
        df["Date_dt"] = pd.to_datetime(df["Date_dt"])
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df = pd.read_csv(CSV)
        df["ts"] = pd.to_datetime(df["Date"] + " " + df["Timestamp"],
                                  format="%d-%m-%Y %H:%M:%S")
        df["Date_dt"] = df["ts"].dt.normalize()

    for c in ("Operation_Mode", "Efficiency_Status"):
        df[c] = df[c].astype(str)

    if "NSI" not in df.columns:
        df["NSI"] = network_stability_index(df)
    if "Network_Quality" not in df.columns:
        df["Network_Quality"] = pd.qcut(df["NSI"], [0, 1 / 3, 2 / 3, 1.0],
                                        labels=["Low", "Medium", "High"])
    df["Network_Quality"] = pd.Categorical(
        df["Network_Quality"].astype(str), ["Low", "Medium", "High"], ordered=True)
    df["Efficiency_Status"] = pd.Categorical(
        df["Efficiency_Status"], ["Low", "Medium", "High"], ordered=True)
    df["Hour"] = df["ts"].dt.hour
    return df


def network_stability_index(df: pd.DataFrame) -> pd.Series:
    """KPI: Network Stability Index, 0-100, higher = more stable.

    Equal-weight blend of latency and packet loss, each min-max normalised
    against the observed operating envelope, then inverted so that high = good.
    """
    lat = (df[LAT] - df[LAT].min()) / (df[LAT].max() - df[LAT].min())
    pl = (df[PL] - df[PL].min()) / (df[PL].max() - df[PL].min())
    return 100 * (1 - 0.5 * (lat + pl))


# --------------------------------------------------------------------- filters
def apply_filters(df, date_range=None, bands=None, effs=None, modes=None,
                  hours=None, machines=None):
    m = pd.Series(True, index=df.index)
    if date_range:
        lo, hi = date_range
        m &= df["Date_dt"].between(pd.Timestamp(lo), pd.Timestamp(hi))
    if bands:
        m &= df["Network_Quality"].astype(str).isin(bands)
    if effs:
        m &= df["Efficiency_Status"].astype(str).isin(effs)
    if modes:
        m &= df["Operation_Mode"].isin(modes)
    if hours:
        m &= df["Hour"].between(hours[0], hours[1])
    if machines:
        m &= df["Machine_ID"].isin(machines)
    return df[m]


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    """Wilson score interval - honest CIs for proportions in sparse bins."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p * 100, max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def binned_outcome(df, driver, outcome, bins=12):
    """Mean outcome +/- 95% CI per equal-count bin of `driver`.

    Overlapping CIs across bins are the visual test for "no relationship".
    """
    if len(df) < bins * 5:
        bins = max(2, len(df) // 20)
    try:
        b = pd.qcut(df[driver], bins, duplicates="drop")
    except (ValueError, IndexError):
        return pd.DataFrame()
    g = df.groupby(b, observed=True).agg(
        x=(driver, "mean"), y=(outcome, "mean"),
        sd=(outcome, "std"), n=(outcome, "size"))
    g = g[g["n"] >= 2].reset_index(drop=True)
    se = g["sd"] / np.sqrt(g["n"])
    g["lo"], g["hi"] = g["y"] - 1.96 * se, g["y"] + 1.96 * se
    return g


def binned_share(df, driver, cls="Low", bins=12):
    """Share (%) of rows in `cls` per equal-count bin of `driver`, Wilson CI."""
    if len(df) < bins * 5:
        bins = max(2, len(df) // 20)
    try:
        b = pd.qcut(df[driver], bins, duplicates="drop")
    except (ValueError, IndexError):
        return pd.DataFrame()
    rows = []
    for _, g in df.groupby(b, observed=True):
        k = int((g["Efficiency_Status"].astype(str) == cls).sum())
        p, lo, hi = wilson(k, len(g))
        rows.append({"x": g[driver].mean(), "y": p, "lo": lo, "hi": hi,
                     "n": len(g), "k": k})
    return pd.DataFrame(rows)


def corr_with_ci(x, y):
    """Pearson r with Fisher-z 95% CI and the variance it explains."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 10 or np.std(x) == 0 or np.std(y) == 0:
        return dict(r=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=n, r2_pct=np.nan)
    r, p = stats.pearsonr(x, y)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    return dict(r=r, lo=lo, hi=hi, p=p, n=n, r2_pct=100 * r * r)


def tost_r(r, n, bound=0.05):
    """TOST equivalence test on Fisher-z. Small p => |r| is inside +/- bound."""
    if n < 10 or not np.isfinite(r):
        return np.nan
    se = 1 / np.sqrt(n - 3)
    z, zb = np.arctanh(r), np.arctanh(bound)
    return max(stats.norm.sf((z + zb) / se), stats.norm.cdf((z - zb) / se))


def min_detectable_r(n, power=0.80):
    """Smallest |r| detectable at alpha=.05 two-sided for the given power."""
    z_b = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power, 0.8416)
    if n <= 4:
        return np.nan
    return (1.96 + z_b) / np.sqrt(n - 3 + (1.96 + z_b) ** 2)


def chi2_p(table):
    """χ² p-value, or NaN when the table is degenerate.

    scipy raises if any expected frequency is zero, which happens whenever a
    row or column of the contingency table sums to zero - e.g. the user has
    filtered to a single efficiency class, so "not Low" is empty everywhere.
    A degenerate table carries no evidence either way, so NaN is the honest
    return value and every caller renders it as "n/a".
    """
    a = np.asarray(table, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2 or a.sum() == 0:
        return np.nan
    if (a.sum(axis=0) == 0).any() or (a.sum(axis=1) == 0).any():
        return np.nan
    try:
        return stats.chi2_contingency(a)[1]
    except ValueError:
        return np.nan


def cramers_v(ct):
    """Cramér's V and its χ² p-value, or (NaN, NaN) for a degenerate table."""
    a = np.asarray(ct, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2 or a.sum() == 0:
        return np.nan, np.nan
    if (a.sum(axis=0) == 0).any() or (a.sum(axis=1) == 0).any():
        return np.nan, np.nan
    try:
        chi2, p = stats.chi2_contingency(a)[:2]
    except ValueError:
        return np.nan, np.nan
    return np.sqrt(chi2 / (a.sum() * (min(a.shape) - 1))), p


def eta_squared(df, col, group="Efficiency_Status"):
    """Share of `col` variance explained by group membership, in percent."""
    grand = df[col].mean()
    groups = [g[col].values for _, g in df.groupby(group, observed=True)]
    if len(groups) < 2:
        return np.nan
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((df[col] - grand) ** 2).sum()
    return 100 * ss_b / ss_t if ss_t else np.nan


def efficiency_rule(err, spd):
    """The labelling rule recovered from the data (99.998% exact).

    Verified in analysis/04: a depth-4 tree on Error_Rate_% and Production_Speed
    reproduces Efficiency_Status for 99,998 of 100,000 rows; the 2 exceptions sit
    exactly on Error_Rate_% == 5.000.
    """
    out = np.full(len(err), "Low", dtype=object)
    out[(np.asarray(err) <= 5.0) & (np.asarray(spd) > 200.0)] = "Medium"
    out[(np.asarray(err) <= 2.0) & (np.asarray(spd) > 400.0)] = "High"
    return out
