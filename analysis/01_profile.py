"""Step 1 - Raw data profiling.

Prints structure, dtypes, missingness, duplicates, cardinality and univariate
statistics for the Thales 6G smart-factory telemetry extract. Read-only.
"""

import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "Thales_Group_Manufacturing.csv"

df = pd.read_csv(CSV)

print("=" * 78)
print("SHAPE:", df.shape)
print("=" * 78)
print("\n--- DTYPES ---")
print(df.dtypes)

print("\n--- HEAD ---")
print(df.head(5))

print("\n--- MISSING VALUES ---")
miss = df.isna().sum()
print(pd.DataFrame({"n_missing": miss, "pct": (100 * miss / len(df)).round(3)}))

print("\n--- FULL-ROW DUPLICATES ---", df.duplicated().sum())
print("--- DUPLICATE (Date, Timestamp, Machine_ID) ---",
      df.duplicated(subset=["Date", "Timestamp", "Machine_ID"]).sum())

print("\n--- CARDINALITY ---")
print(df.nunique().sort_values())

print("\n--- CATEGORICALS ---")
for c in ["Operation_Mode", "Efficiency_Status"]:
    print(f"\n{c}:")
    print(df[c].value_counts(dropna=False))
    print((100 * df[c].value_counts(normalize=True)).round(2))

print("\n--- NUMERIC DESCRIBE ---")
num = df.select_dtypes("number")
print(num.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)

print("\n--- SKEW / KURTOSIS ---")
print(pd.DataFrame({"skew": num.skew().round(3), "kurtosis": num.kurtosis().round(3)}))

print("\n--- DATE COVERAGE ---")
d = pd.to_datetime(df["Date"], format="%d-%m-%Y")
print("min:", d.min(), " max:", d.max(), " n_days:", d.nunique())
print("rows per day (describe):")
print(d.value_counts().describe())

print("\n--- TIMESTAMP GRANULARITY ---")
print("unique timestamps:", df["Timestamp"].nunique())
print(df["Timestamp"].head(3).tolist(), "...", df["Timestamp"].tail(3).tolist())

print("\n--- MACHINE_ID RANGE ---")
print("min", df["Machine_ID"].min(), "max", df["Machine_ID"].max(),
      "n", df["Machine_ID"].nunique())
print("rows per machine (describe):")
print(df["Machine_ID"].value_counts().describe())

print("\n--- PHYSICAL PLAUSIBILITY / NEGATIVES ---")
for c in num.columns:
    neg = (df[c] < 0).sum()
    if neg:
        print(f"{c}: {neg} negative values")
print("done")
