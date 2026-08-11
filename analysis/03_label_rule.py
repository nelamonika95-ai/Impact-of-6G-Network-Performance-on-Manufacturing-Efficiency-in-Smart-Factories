"""Step 3 - Reverse-engineer the Efficiency_Status labelling rule.

Step 2 showed Efficiency_Status is ~99.8% predictable, but ONLY from
Error_Rate_% and Production_Speed_units_per_hr (eta^2 38.4% and 11.3%);
every network and mechanical feature had eta^2 < 0.004% and MI = 0.

If the label is a deterministic threshold rule on those two columns, we can
recover it exactly - and that explains why the network shows no effect.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "Thales_Group_Manufacturing.csv")
df["Efficiency_Status"] = df["Efficiency_Status"].astype(str)

E, S = "Error_Rate_%", "Production_Speed_units_per_hr"
y = df["Efficiency_Status"].to_numpy()

print("#" * 78)
print("PERMUTATION CONTROL (fixes the crash in step 2)")
print("#" * 78)
X = df[["Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
        "Network_Latency_ms", "Packet_Loss_%", "Quality_Control_Defect_Rate_%",
        S, "Predictive_Maintenance_Score", E]].to_numpy()
rng = np.random.default_rng(1)
yshuf = rng.permutation(y)
a, b, c, d = train_test_split(X, yshuf, test_size=.25, random_state=0, stratify=yshuf)
m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(a, c)
p = m.predict(b)
print(f"shuffled-target accuracy={accuracy_score(d, p):.4f} "
      f"balanced={balanced_accuracy_score(d, p):.4f} (real run: 0.9984 / 0.9941)")

print("\n" + "#" * 78)
print("NETWORK-ONLY MODEL: can latency + packet loss alone predict efficiency?")
print("#" * 78)
Xn = df[["Network_Latency_ms", "Packet_Loss_%"]].to_numpy()
a, b, c, d = train_test_split(Xn, y, test_size=.25, random_state=0, stratify=y)
m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(a, c)
p = m.predict(b)
print(f"latency+loss only : accuracy={accuracy_score(d, p):.4f} "
      f"balanced={balanced_accuracy_score(d, p):.4f}")
print(f"majority baseline : accuracy={pd.Series(y).value_counts(normalize=True).max():.4f} "
      f"balanced=0.3333")

print("\n" + "#" * 78)
print("TWO-FEATURE MODEL: Error_Rate_% + Production_Speed only")
print("#" * 78)
X2 = df[[E, S]].to_numpy()
a, b, c, d = train_test_split(X2, y, test_size=.25, random_state=0, stratify=y)
m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(a, c)
p = m.predict(b)
print(f"accuracy={accuracy_score(d, p):.4f} balanced={balanced_accuracy_score(d, p):.4f}")

print("\n" + "#" * 78)
print("SHALLOW DECISION TREE ON THE TWO DRIVERS (rule recovery)")
print("#" * 78)
dt = DecisionTreeClassifier(max_depth=4, random_state=0).fit(df[[E, S]], y)
print(f"in-sample accuracy = {dt.score(df[[E, S]], y):.5f}")
print(export_text(dt, feature_names=[E, S], decimals=3))

print("\n" + "#" * 78)
print("OBSERVED FEASIBLE REGION PER CLASS")
print("#" * 78)
print(df.groupby("Efficiency_Status")[[E, S]]
        .agg(["min", "max", "mean", "count"]).round(3).to_string())

print("\n" + "#" * 78)
print("CANDIDATE HAND RULE  (thresholds read off the tree)")
print("#" * 78)


def rule(r):
    if r[E] < 2.0 and r[S] > 400:
        return "High"
    if r[E] < 5.0 and r[S] > 300:
        return "Medium"
    return "Low"


pred = df[[E, S]].apply(rule, axis=1)
print("hand-rule accuracy:", round((pred == df["Efficiency_Status"]).mean(), 5))
print(pd.crosstab(df["Efficiency_Status"], pred,
                 rownames=["actual"], colnames=["hand_rule"]).to_string())

print("\n" + "#" * 78)
print("CLASS REGIONS: joint distribution over Error_Rate x Speed bins")
print("#" * 78)
eb = pd.cut(df[E], [0, 1, 2, 3, 5, 7.5, 10, 15], right=False)
sb = pd.cut(df[S], [0, 100, 200, 300, 350, 400, 450, 500], right=False)
for cls in ["High", "Medium", "Low"]:
    sub = df[df["Efficiency_Status"] == cls]
    print(f"\n--- {cls} (n={len(sub)}) counts by Error_Rate(rows) x Speed(cols) ---")
    print(pd.crosstab(eb[sub.index], sb[sub.index], dropna=False).to_string())

print("\n" + "#" * 78)
print("PURITY CHECK: is the label pure inside the recovered regions?")
print("#" * 78)
df["_region"] = np.select(
    [(df[E] < 2.0) & (df[S] > 400), (df[E] < 5.0) & (df[S] > 300)],
    ["High_region", "Medium_region"], default="Low_region")
print((100 * pd.crosstab(df["_region"], df["Efficiency_Status"],
                         normalize="index")).round(3).to_string())
print("\nrows per region:")
print(df["_region"].value_counts().to_string())
