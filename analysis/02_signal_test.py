"""Step 2 - Does the network -> efficiency signal actually exist?

The profile showed every numeric feature is ~Uniform and independent-looking.
Before writing any narrative we must establish, with tests, whether
Network_Latency_ms / Packet_Loss_% carry ANY information about
Efficiency_Status, Production_Speed, Error_Rate or Defect_Rate.

Tests run:
  A. Pearson + Spearman correlation matrix (with p-values) on all numerics
  B. Chi-square: Efficiency_Status vs latency/packet-loss deciles
  C. Kruskal-Wallis + ANOVA: latency/loss distribution across Efficiency_Status
  D. Effect sizes (Cramer's V, eta-squared) - significance is cheap at n=100k
  E. Mutual information of every feature vs the target
  F. Can a model beat the majority-class baseline? (honest ceiling check)
  G. Is Efficiency_Status a deterministic rule of any single feature?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "Thales_Group_Manufacturing.csv")

NUM = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
       "Network_Latency_ms", "Packet_Loss_%", "Quality_Control_Defect_Rate_%",
       "Production_Speed_units_per_hr", "Predictive_Maintenance_Score",
       "Error_Rate_%"]
NET = ["Network_Latency_ms", "Packet_Loss_%"]
OUT = ["Production_Speed_units_per_hr", "Error_Rate_%",
       "Quality_Control_Defect_Rate_%"]

print("#" * 78)
print("A. CORRELATION: network drivers vs production outcomes")
print("#" * 78)
rows = []
for a in NET:
    for b in OUT + ["Predictive_Maintenance_Score", "Temperature_C",
                    "Vibration_Hz", "Power_Consumption_kW"]:
        r, pr = stats.pearsonr(df[a], df[b])
        rho, ps = stats.spearmanr(df[a], df[b])
        rows.append({"x": a, "y": b, "pearson_r": round(r, 5), "p_pearson": f"{pr:.3g}",
                     "spearman_rho": round(rho, 5), "p_spearman": f"{ps:.3g}",
                     "r_squared_pct": round(100 * r * r, 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n--- full Pearson matrix (all numerics) ---")
print(df[NUM].corr().round(4).to_string())
print("\nmax |r| off-diagonal:",
      round(df[NUM].corr().abs().where(~np.eye(len(NUM), dtype=bool)).max().max(), 5))

print("\n" + "#" * 78)
print("B. CHI-SQUARE: Efficiency_Status vs latency / packet-loss deciles")
print("#" * 78)


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))


for c in NET:
    dec = pd.qcut(df[c], 10, labels=False, duplicates="drop")
    ct = pd.crosstab(dec, df["Efficiency_Status"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"\n{c}: chi2={chi2:.2f} dof={dof} p={p:.4g} CramersV={cramers_v(ct):.4f}")
    print((100 * pd.crosstab(dec, df["Efficiency_Status"], normalize="index")).round(2).to_string())

print("\n--- same for Operation_Mode ---")
ct = pd.crosstab(df["Operation_Mode"], df["Efficiency_Status"])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"chi2={chi2:.2f} p={p:.4g} CramersV={cramers_v(ct):.4f}")
print((100 * pd.crosstab(df["Operation_Mode"], df["Efficiency_Status"],
                         normalize="index")).round(2).to_string())

print("\n" + "#" * 78)
print("C/D. DISTRIBUTION OF EACH FEATURE ACROSS Efficiency_Status (+ effect size)")
print("#" * 78)
res = []
for c in NUM:
    groups = [g[c].values for _, g in df.groupby("Efficiency_Status")]
    H, p_kw = stats.kruskal(*groups)
    F, p_an = stats.f_oneway(*groups)
    # eta^2 from one-way ANOVA
    grand = df[c].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((df[c] - grand) ** 2).sum()
    res.append({"feature": c,
                "mean_High": round(df.loc[df.Efficiency_Status == "High", c].mean(), 3),
                "mean_Medium": round(df.loc[df.Efficiency_Status == "Medium", c].mean(), 3),
                "mean_Low": round(df.loc[df.Efficiency_Status == "Low", c].mean(), 3),
                "kruskal_p": f"{p_kw:.3g}", "anova_p": f"{p_an:.3g}",
                "eta_sq_pct": round(100 * ss_b / ss_t, 4)})
print(pd.DataFrame(res).sort_values("eta_sq_pct", ascending=False).to_string(index=False))

print("\n" + "#" * 78)
print("E. MUTUAL INFORMATION vs Efficiency_Status (nats)")
print("#" * 78)
X = df[NUM].copy()
X["Operation_Mode_code"] = df["Operation_Mode"].astype("category").cat.codes
y = df["Efficiency_Status"]
mi = mutual_info_classif(X, y, random_state=0)
print(pd.Series(mi, index=X.columns).sort_values(ascending=False).round(5).to_string())
print("\nTarget entropy (nats):",
      round(stats.entropy(y.value_counts(normalize=True).values), 5))

print("\n" + "#" * 78)
print("F. PREDICTABILITY CEILING - can any model beat majority class?")
print("#" * 78)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
maj = y.value_counts(normalize=True).max()
print(f"Majority-class baseline accuracy = {maj:.4f}")
clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"HistGradientBoosting accuracy    = {accuracy_score(yte, pred):.4f}")
print(f"Balanced accuracy                = {balanced_accuracy_score(yte, pred):.4f}"
      f"   (chance = {1/3:.4f})")
print(classification_report(yte, pred, zero_division=0))

# permutation sanity check: shuffle the target, same pipeline
yshuf = y.sample(frac=1, random_state=1).values
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, yshuf, test_size=0.25, random_state=0,
                                          stratify=yshuf)
clf2 = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr2, ytr2)
print(f"Same model on SHUFFLED target    = {accuracy_score(yte2, clf2.predict(Xte2)):.4f}"
      f" / balanced {balanced_accuracy_score(yte2, clf2.predict(Xte2)):.4f}")

print("\n" + "#" * 78)
print("G. IS THE LABEL A DETERMINISTIC RULE OF ANY FEATURE?")
print("#" * 78)
for c in NUM:
    g = df.groupby("Efficiency_Status")[c].agg(["min", "max", "mean"])
    overlap = (g["min"].max() < g["max"].min())
    print(f"{c:32s} ranges overlap across classes: {overlap}")
    print(g.round(3).to_string())
    print()
