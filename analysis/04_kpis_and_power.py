"""Step 4 - Rigorous quantification of the network effect, KPIs, and artifact export.

Steps 2-3 established:
  * Efficiency_Status = deterministic f(Error_Rate_%, Production_Speed) - tree
    recovers it at 99.998% in-sample.
  * Network_Latency_ms and Packet_Loss_% are independent of everything
    (max |r| = 0.0075, MI = 0, chi2 p = 0.74).

A non-significant p-value alone is weak evidence. This step makes the null
claim defensible:
  1. Exact label rule verification.
  2. Timeline / key integrity audit.
  3. Power analysis - what is the smallest effect we COULD have detected?
  4. TOST equivalence tests - formally accept "no practically relevant effect".
  5. The four project KPIs, computed as specified.
  6. Operation-mode interaction (network sensitivity per mode).
  7. Export tidy artifacts (JSON + CSV) for the dashboard and the paper.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "Thales_Group_Manufacturing.csv")
df["Efficiency_Status"] = df["Efficiency_Status"].astype(str)
df["Operation_Mode"] = df["Operation_Mode"].astype(str)

E = "Error_Rate_%"
S = "Production_Speed_units_per_hr"
LAT = "Network_Latency_ms"
PL = "Packet_Loss_%"
DEF = "Quality_Control_Defect_Rate_%"
NET = [LAT, PL]
OUTCOMES = [S, E, DEF]
NUM = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", LAT, PL, DEF, S,
       "Predictive_Maintenance_Score", E]

art = {}

# ---------------------------------------------------------------- 1. label rule
print("#" * 78)
print("1. EXACT LABEL RULE VERIFICATION")
print("#" * 78)


def exact_rule(err, spd):
    """Recovered from the depth-4 tree in step 3."""
    out = np.full(len(err), "Low", dtype=object)
    med = (err <= 5.0) & (spd > 200.0)
    out[med] = "Medium"
    out[(err <= 2.0) & (spd > 400.0)] = "High"
    return out


pred = exact_rule(df[E].to_numpy(), df[S].to_numpy())
acc = (pred == df["Efficiency_Status"].to_numpy()).mean()
mism = df.loc[pred != df["Efficiency_Status"].to_numpy(),
              [E, S, "Efficiency_Status"]]
print(f"exact-rule accuracy = {acc:.6f}   mismatches = {len(mism)} / {len(df)}")
print("\nmismatching rows (boundary ties):")
print(mism.to_string())
print("\nconfusion:")
print(pd.crosstab(df["Efficiency_Status"], pred,
                 rownames=["actual"], colnames=["rule"]).to_string())
art["label_rule"] = {
    "rule": "High if Error_Rate_% <= 2 and Speed > 400; "
            "Medium if Error_Rate_% <= 5 and Speed > 200; else Low",
    "accuracy": round(float(acc), 6),
    "n_mismatch": int(len(mism)),
}

# deterministic-tree confirmation
dt = DecisionTreeClassifier(max_depth=4, random_state=0).fit(df[[E, S]], df["Efficiency_Status"])
art["label_rule"]["tree_depth4_insample_acc"] = round(float(dt.score(df[[E, S]], df["Efficiency_Status"])), 6)

# ------------------------------------------------------------- 2. integrity
print("\n" + "#" * 78)
print("2. TIMELINE & KEY INTEGRITY AUDIT")
print("#" * 78)
df["Date_dt"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df["ts"] = pd.to_datetime(df["Date"] + " " + df["Timestamp"],
                          format="%d-%m-%Y %H:%M:%S")
print("range:", df["ts"].min(), "->", df["ts"].max())
print("n rows:", len(df), " n unique ts:", df["ts"].nunique())
print("monotonic non-decreasing ts:", df["ts"].is_monotonic_increasing)

per_day = df["Date_dt"].value_counts().sort_index()
odd = per_day[per_day != 1440]
print(f"\ndays with != 1440 rows ({len(odd)}):")
print(odd.to_string())

dup_key = df.duplicated(subset=["ts", "Machine_ID"]).sum()
dup_ts = df["ts"].duplicated().sum()
print(f"\nduplicate (ts, Machine_ID) pairs : {dup_key}")
print(f"rows sharing a timestamp          : {dup_ts}")
gap = df["ts"].sort_values().diff().value_counts().head(8)
print("\nmost common inter-row gaps:")
print(gap.to_string())
art["integrity"] = {
    "rows": int(len(df)), "cols": int(df.shape[1] - 2),
    "start": str(df["ts"].min()), "end": str(df["ts"].max()),
    "n_days": int(df["Date_dt"].nunique()),
    "machines": int(df["Machine_ID"].nunique()),
    "missing_cells": int(df.isna().sum().sum()),
    "full_row_duplicates": 0,
    "duplicate_ts_machine_pairs": int(dup_key),
    "days_not_1440_rows": {str(k.date()): int(v) for k, v in odd.items()},
}

# ------------------------------------------------------- 3. power analysis
print("\n" + "#" * 78)
print("3. POWER ANALYSIS - smallest detectable effect at n=100,000")
print("#" * 78)
n = len(df)
for power, z_b in [(0.80, 0.8416), (0.95, 1.6449)]:
    z_a = 1.9600  # two-sided alpha = .05
    r_min = (z_a + z_b) / np.sqrt(n - 3 + (z_a + z_b) ** 2)
    print(f"power={power:.0%}: minimum detectable |r| = {r_min:.5f} "
          f"(explains {100*r_min**2:.4f}% of variance)")
    art.setdefault("power", {})[f"min_detectable_r_at_{int(power*100)}pct"] = round(float(r_min), 5)

obs = {f"{a}~{b}": float(stats.pearsonr(df[a], df[b])[0]) for a in NET for b in OUTCOMES}
print("\nobserved |r| for network->outcome pairs:")
for k, v in obs.items():
    print(f"  {k:55s} r={v:+.5f}  |r|={abs(v):.5f}")
print(f"\nlargest observed |r| among these = {max(abs(v) for v in obs.values()):.5f}")
art["power"]["observed_network_outcome_r"] = {k: round(v, 5) for k, v in obs.items()}

# ------------------------------------------- 4. TOST equivalence testing
print("\n" + "#" * 78)
print("4. TOST EQUIVALENCE TESTS  (H1: |r| < 0.05, i.e. <0.25% of variance)")
print("#" * 78)
BOUND = 0.05


def tost_r(r, n, bound=BOUND):
    """Two one-sided tests on Fisher-z transformed correlation."""
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh(r)
    zb = np.arctanh(bound)
    p_lower = stats.norm.sf((z - (-zb)) / se)   # H0: r <= -bound
    p_upper = stats.norm.cdf((z - zb) / se)     # H0: r >= +bound
    return max(p_lower, p_upper)


rows = []
for a in NET:
    for b in OUTCOMES + ["Predictive_Maintenance_Score"]:
        r, p = stats.pearsonr(df[a], df[b])
        lo, hi = np.tanh(np.arctanh(r) - 1.96 / np.sqrt(n - 3)), \
                 np.tanh(np.arctanh(r) + 1.96 / np.sqrt(n - 3))
        pt = tost_r(r, n)
        rows.append({"driver": a, "outcome": b, "r": round(r, 5),
                     "ci95_low": round(lo, 5), "ci95_high": round(hi, 5),
                     "p_nhst": round(p, 4), "p_TOST": f"{pt:.3g}",
                     "equivalent_to_zero": pt < 0.05})
tost = pd.DataFrame(rows)
print(tost.to_string(index=False))
print(f"\nEquivalence established for {tost.equivalent_to_zero.sum()}/{len(tost)} pairs.")
tost.to_csv(OUTDIR / "tost_equivalence.csv", index=False)

# --------------------------------------------------------------- 5. KPIs
print("\n" + "#" * 78)
print("5. PROJECT KPIs")
print("#" * 78)

# Network Stability Index: 0-100, higher = more stable. Percentile-normalised
# blend of latency and packet loss against the observed operating envelope.
lat_lo, lat_hi = df[LAT].min(), df[LAT].max()
pl_lo, pl_hi = df[PL].min(), df[PL].max()
df["NSI"] = 100 * (1 - 0.5 * ((df[LAT] - lat_lo) / (lat_hi - lat_lo)
                              + (df[PL] - pl_lo) / (pl_hi - pl_lo)))
print(f"Network Stability Index: mean={df['NSI'].mean():.2f} "
      f"sd={df['NSI'].std():.2f} min={df['NSI'].min():.2f} max={df['NSI'].max():.2f}")

# Network quality bands (terciles of NSI)
df["Network_Quality"] = pd.qcut(df["NSI"], [0, 1/3, 2/3, 1.0],
                                labels=["Low", "Medium", "High"])
band = pd.crosstab(df["Network_Quality"], df["Efficiency_Status"], normalize="index") * 100
print("\nEfficiency mix (%) by network-quality band:")
print(band.round(3).to_string())
chi2, p_band, dof, _ = stats.chi2_contingency(
    pd.crosstab(df["Network_Quality"], df["Efficiency_Status"]))
print(f"chi2={chi2:.3f} dof={dof} p={p_band:.4f}")
band.round(4).to_csv(OUTDIR / "efficiency_by_network_band.csv")

# Latency Sensitivity Score: OLS slope of outcome on latency (units per ms)
print("\nLatency Sensitivity Score (OLS slope +/- 95% CI):")
lss = {}
for b in OUTCOMES:
    res = stats.linregress(df[LAT], df[b])
    ci = 1.96 * res.stderr
    lss[b] = {"slope_per_ms": round(res.slope, 6),
              "ci95": [round(res.slope - ci, 6), round(res.slope + ci, 6)],
              "p": round(res.pvalue, 4), "r2_pct": round(100 * res.rvalue ** 2, 5)}
    span = (df[LAT].max() - df[LAT].min()) * res.slope
    print(f"  {b:32s} {res.slope:+.6f}/ms  CI[{res.slope-ci:+.6f},{res.slope+ci:+.6f}]  "
          f"p={res.pvalue:.3f}  effect across full 1-50ms span = {span:+.3f}")
    lss[b]["effect_across_full_latency_span"] = round(span, 4)
art["latency_sensitivity"] = lss

# Packet Loss Impact Ratio: mean outcome in worst loss decile / best decile
print("\nPacket Loss Impact Ratio (worst decile vs best decile):")
d10 = pd.qcut(df[PL], 10, labels=False)
plir = {}
for b in OUTCOMES + ["Efficiency_Status"]:
    if b == "Efficiency_Status":
        lo_v = (df.loc[d10 == 0, b] == "Low").mean() * 100
        hi_v = (df.loc[d10 == 9, b] == "Low").mean() * 100
        print(f"  %Low efficiency: best-loss-decile={lo_v:.2f}%  "
              f"worst={hi_v:.2f}%  ratio={hi_v/lo_v:.4f}")
        plir["pct_low_efficiency"] = {"best_decile": round(lo_v, 3),
                                      "worst_decile": round(hi_v, 3),
                                      "ratio": round(hi_v / lo_v, 4)}
    else:
        lo_v, hi_v = df.loc[d10 == 0, b].mean(), df.loc[d10 == 9, b].mean()
        t, pv = stats.ttest_ind(df.loc[d10 == 0, b], df.loc[d10 == 9, b])
        print(f"  {b:32s} best={lo_v:.3f} worst={hi_v:.3f} "
              f"ratio={hi_v/lo_v:.4f} t-test p={pv:.3f}")
        plir[b] = {"best_decile": round(lo_v, 3), "worst_decile": round(hi_v, 3),
                   "ratio": round(hi_v / lo_v, 4), "p": round(pv, 4)}
art["packet_loss_impact_ratio"] = plir

# Network-Efficiency Correlation / breakpoint search
print("\nNetwork-Efficiency breakpoint scan: %Low efficiency above each latency cut")
scan = []
for cut in range(5, 50, 5):
    above = df[df[LAT] >= cut]
    below = df[df[LAT] < cut]
    pa = (above["Efficiency_Status"] == "Low").mean() * 100
    pb = (below["Efficiency_Status"] == "Low").mean() * 100
    _, pv = stats.chi2_contingency(pd.crosstab(df[LAT] >= cut,
                                               df["Efficiency_Status"]))[:2]
    scan.append({"latency_cut_ms": cut, "pct_low_below": round(pb, 3),
                 "pct_low_at_or_above": round(pa, 3), "delta_pp": round(pa - pb, 3),
                 "chi2_p": round(pv, 4)})
scan = pd.DataFrame(scan)
print(scan.to_string(index=False))
print(f"\nlargest |delta| across all cuts = {scan.delta_pp.abs().max():.3f} pp")
scan.to_csv(OUTDIR / "latency_breakpoint_scan.csv", index=False)
art["breakpoint_scan"] = {"max_abs_delta_pp": round(float(scan.delta_pp.abs().max()), 3),
                          "any_significant": bool((scan.chi2_p < 0.05).any())}

# ------------------------------------------ 6. operation-mode interaction
print("\n" + "#" * 78)
print("6. OPERATION-MODE INTERACTION - is any mode network-sensitive?")
print("#" * 78)
rows = []
for mode, g in df.groupby("Operation_Mode"):
    for a in NET:
        for b in OUTCOMES:
            r, p = stats.pearsonr(g[a], g[b])
            rows.append({"mode": mode, "n": len(g), "driver": a, "outcome": b,
                         "r": round(r, 5), "p": round(p, 4)})
mode_r = pd.DataFrame(rows)
print(mode_r.to_string(index=False))
print(f"\nmax |r| within any mode = {mode_r.r.abs().max():.5f}")
print(f"pairs significant at .05 = {(mode_r.p < .05).sum()} / {len(mode_r)} "
      f"(expected by chance: {0.05*len(mode_r):.1f})")
mode_r.to_csv(OUTDIR / "operation_mode_network_correlations.csv", index=False)
art["operation_mode"] = {
    "max_abs_r": round(float(mode_r.r.abs().max()), 5),
    "n_significant": int((mode_r.p < .05).sum()), "n_tests": int(len(mode_r)),
    "expected_false_positives": round(0.05 * len(mode_r), 1),
}

print("\nefficiency mix (%) by mode x network band:")
piv = (df.groupby(["Operation_Mode", "Network_Quality"], observed=True)["Efficiency_Status"]
         .apply(lambda s: (s == "Low").mean() * 100).unstack())
print(piv.round(3).to_string())
piv.round(4).to_csv(OUTDIR / "mode_band_pct_low.csv")

# ------------------------------------------------------- 7. export tables
print("\n" + "#" * 78)
print("7. EXPORTING ARTIFACTS")
print("#" * 78)

df[NUM].corr().round(5).to_csv(OUTDIR / "correlation_matrix.csv")

desc = df[NUM].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
desc["skew"] = df[NUM].skew()
desc["excess_kurtosis"] = df[NUM].kurtosis()
# uniformity test against the observed [min,max] support
ks = []
for c in NUM:
    lo, hi = df[c].min(), df[c].max()
    ks.append(stats.kstest((df[c] - lo) / (hi - lo), "uniform").pvalue)
desc["ks_uniform_p"] = ks
desc.round(5).to_csv(OUTDIR / "univariate_summary.csv")
print("\nuniformity check (KS test vs Uniform on observed support):")
print(desc[["mean", "std", "skew", "excess_kurtosis", "ks_uniform_p"]].round(4).to_string())
art["uniformity"] = {c: round(float(p), 4) for c, p in zip(NUM, ks)}

eff_by_mode = pd.crosstab(df["Operation_Mode"], df["Efficiency_Status"],
                          normalize="index").round(5) * 100
eff_by_mode.to_csv(OUTDIR / "efficiency_by_mode.csv")

# machine-level rollup for the dashboard
mach = df.groupby("Machine_ID").agg(
    n=("Machine_ID", "size"), lat_mean=(LAT, "mean"), lat_p95=(LAT, lambda s: s.quantile(.95)),
    loss_mean=(PL, "mean"), nsi_mean=("NSI", "mean"), speed_mean=(S, "mean"),
    err_mean=(E, "mean"), defect_mean=(DEF, "mean"),
    pct_low=("Efficiency_Status", lambda s: (s == "Low").mean() * 100),
    pct_high=("Efficiency_Status", lambda s: (s == "High").mean() * 100),
).round(4)
mach.to_csv(OUTDIR / "machine_rollup.csv")
print(f"\nmachine rollup: %Low ranges {mach.pct_low.min():.2f}-{mach.pct_low.max():.2f}, "
      f"sd={mach.pct_low.std():.2f}")
art["machine_variation"] = {"pct_low_min": float(mach.pct_low.min()),
                            "pct_low_max": float(mach.pct_low.max()),
                            "pct_low_sd": round(float(mach.pct_low.std()), 3)}

daily = df.groupby("Date_dt").agg(
    lat_mean=(LAT, "mean"), loss_mean=(PL, "mean"), nsi_mean=("NSI", "mean"),
    speed_mean=(S, "mean"), err_mean=(E, "mean"), defect_mean=(DEF, "mean"),
    pct_low=("Efficiency_Status", lambda s: (s == "Low").mean() * 100),
).round(4)
daily.to_csv(OUTDIR / "daily_trend.csv")

art["class_balance"] = df["Efficiency_Status"].value_counts().to_dict()
art["mode_balance"] = df["Operation_Mode"].value_counts().to_dict()
art["eta_squared_vs_efficiency"] = {}
for c in NUM:
    groups = [g[c].values for _, g in df.groupby("Efficiency_Status")]
    grand = df[c].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((df[c] - grand) ** 2).sum()
    art["eta_squared_vs_efficiency"][c] = round(float(100 * ss_b / ss_t), 5)

art["model_benchmarks"] = {
    "majority_baseline_acc": 0.7782, "all_features_acc": 0.9984,
    "all_features_balanced_acc": 0.9941,
    "two_driver_acc": 0.9983, "two_driver_balanced_acc": 0.9931,
    "network_only_acc": 0.7782, "network_only_balanced_acc": 0.3333,
    "shuffled_target_acc": 0.7782, "shuffled_target_balanced_acc": 0.3333,
}

with open(OUTDIR / "findings.json", "w") as f:
    json.dump(art, f, indent=2, default=str)

# analysis-ready parquet for the dashboard
keep = ["ts", "Date_dt", "Machine_ID", "Operation_Mode"] + NUM + \
       ["Efficiency_Status", "NSI", "Network_Quality"]
df[keep].to_parquet(OUTDIR / "analysis_ready.parquet", index=False)

print("\nwrote:")
for p in sorted(OUTDIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
