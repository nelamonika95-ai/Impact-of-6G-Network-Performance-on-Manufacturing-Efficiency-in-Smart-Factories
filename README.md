<div align="center">

# ⚡ Impact of 6G Network Performance on Manufacturing Efficiency

### *Statistical & Empirical Analysis of 100,000 Machine-Minutes of Telemetry in Smart Factories*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Executive Summary

Connectivity-first analysis evaluating **100,000 machine-minutes** of Thales Group smart-factory telemetry across **50 industrial machines** (*1 Jan – 10 Mar 2025*). This study investigates the degree to which **6G network performance metrics** (latency, packet loss) directly influence production efficiency, error rates, and product quality.

> [!IMPORTANT]
> **Headline Finding:** 6G Network performance **does not** impact production efficiency in this dataset — primarily because the dataset's target labels were synthetic rules generated independently of network KPIs.

---

## 📊 Key Findings & Evidence Matrix

| Analytical Metric | Statistical Result | Practical Interpretation |
| :--- | :--- | :--- |
| **Max Network → Production Correlation** | $\|r\| = 0.0071$ | Below detection floor at 80% power ($0.0089$) |
| **Efficiency Mix across Network Bands** | $\chi^2(4), p = 0.857$, Cramér's $V = 0.0026$ | No association between network quality and efficiency |
| **Latency Threshold Searches (5–47.5 ms)** | `0` of `18` tests significant | Zero operational break-points discovered |
| **Equivalence Testing (TOST, $\|r\| < 0.05$)** | `8` of `8` pass, $p < 10^{-41}$ | Statistically confirmed equivalence to zero effect |
| **Gradient Boosting (Latency + Loss)** | $33.3\%$ Balanced Accuracy | Equivalent to random chance |
| **Recovered Label Decision Rule** | **$99.998\%$ accuracy from non-network features** | `Efficiency_Status` is deterministic without network metrics |

### 🔍 Recovered Decision Rule
The dataset's underlying label generator was reverse-engineered with $99.998\%$ fidelity using only machine operational metrics:

```python
if Error_Rate_% <= 2 and Production_Speed > 400:
    Efficiency_Status = "High"
elif Error_Rate_% <= 5 and Production_Speed > 200:
    Efficiency_Status = "Medium"
else:
    Efficiency_Status = "Low"
```

> [!NOTE]
> Network KPIs appear **nowhere** in the label assignment logic. All numeric features exhibit uniform distributions (excess kurtosis $\approx -1.2$) and mutual independence (max $\|r\| = 0.0075$) — the structural signature of synthetic data generation.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Ensure you have **Python 3.9+** installed. Create and activate a virtual environment:

```bash
# Clone the repository (if applicable)
git clone <your-repository-url>
cd 6G-Impact

# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell / Command Prompt:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
Launch the multi-module Streamlit analytics platform:

```bash
streamlit run app.py
```

### 3. Run Analysis Pipelines & Regenerate Artifacts
To execute the statistical profiling scripts and generate parquet/CSV cache files:

```bash
python analysis/01_profile.py
python analysis/02_signal_test.py
python analysis/03_label_rule.py
python analysis/04_kpis_and_power.py
```

### 4. Run Test Suite
Run the unit and smoke test suite (40 headless execution tests):

```bash
python -m pytest tests -q
```

---

## 📂 Deliverables & Documentation Index

| Deliverable | Description | Location |
| :--- | :--- | :--- |
| 📑 **Research Paper** | Complete empirical analysis, EDA, insights, and recommendations | [`docs/research_paper.md`](docs/research_paper.md) |
| 🏛️ **Executive Summary** | Policy & strategic overview tailored for government stakeholders | [`docs/executive_summary.md`](docs/executive_summary.md) |
| 📊 **Interactive Dashboard** | Live Streamlit analytics application with 6 detailed modules | [`app.py`](app.py) |
| 🧪 **Test Suite** | Comprehensive headless smoke tests covering edge cases | [`tests/test_app_smoke.py`](tests/test_app_smoke.py) |

---

## 🎛️ Dashboard Analytics Modules

The interactive application features **6 specialized diagnostic modules**:

1. **🌐 Network Performance Overview**: Latency & packet-loss trends, Network Stability Index (NSI) scorecards, per-machine stability metrics, and hourly telemetry profiles.
2. **📈 Network vs. Efficiency**: Efficiency class distributions across network quality bands, low-efficiency rates across latency deciles with Wilson confidence intervals.
3. **⚠️ Quality & Error Impact**: Defect and error rate heatmaps against packet loss, spike detection, and bivariate latency-loss planes.
4. **💡 6G Optimization Insights**: Threshold search algorithms, packet-loss risk zones, formal TOST equivalence verification, and network budget recommendations.
5. **🔬 Efficiency Diagnostics**: Feature importance ($\eta^2$), decision boundary visualization, model benchmark comparisons, and feature-independence matrices.
6. **📑 Data Integrity & Method**: Audit of missing values, distributional flags, synthetic data diagnostics, and reproduction steps.

---

## 📐 Implemented KPIs & Mathematical Definitions

| KPI | Mathematical / Algorithmic Definition |
| :--- | :--- |
| **Network Stability Index (NSI)** | $\text{NSI} = 100 \times \left(1 - \frac{1}{2}\left[\text{Latency}_{\text{norm}} + \text{Loss}_{\text{norm}}\right]\right)$ (Min-Max normalized over envelope) |
| **Latency Sensitivity Score** | OLS regression slope of production outcome on latency with 95% Confidence Intervals |
| **Packet Loss Impact Ratio** | $\frac{\text{Mean Outcome in Worst Loss Decile}}{\text{Mean Outcome in Best Loss Decile}}$ |
| **Network-Efficiency Threshold** | Grid-search algorithm finding optimal split point maximizing Information Gain |

---

## 🏗️ Repository Architecture

```
6G-Impact/
├── app.py                      # Main Streamlit Dashboard (6 Analytical Modules)
├── requirements.txt            # Project Python dependencies
├── Thales_Group_Manufacturing.csv # Raw telemetry dataset (100k rows)
├── src/                        # Core Application Code
│   ├── data.py                 # Data loaders, statistical helpers, NSI logic, filters
│   └── theme.py                # Color vision deficiency (CVD) validated themes & Plotly styling
├── analysis/                   # Statistical Pipeline & Modeling Scripts
│   ├── 01_profile.py           # Exploratory data profiling & distribution checks
│   ├── 02_signal_test.py       # Signal detection: Pearson/Spearman, Chi-Square, MI
│   ├── 03_label_rule.py        # Decision tree rule extraction for target labels
│   └── 04_kpis_and_power.py    # Power analysis, TOST equivalence tests, artifact generation
├── docs/                       # Project Documentation & Formal Reports
│   ├── research_paper.md       # Comprehensive scientific research document
│   └── executive_summary.md    # Briefing document for leadership & stakeholders
├── outputs/                    # Processed cache files, CSVs, JSONs, and Parquet data
└── tests/                      # Automated Quality Assurance
    └── test_app_smoke.py       # Headless Streamlit smoke tests (40 tests)
```

---

## 🔬 Methodological Rigor & Integrity

- **Effect Sizes over p-Values**: With $N = 100,000$, trivial correlations ($r = 0.009$) reach $p < 0.05$ while explaining $< 0.01\%$ variance. All claims emphasize effect size metrics and confidence intervals.
- **Two One-Sided Tests (TOST)**: Utilized TOST equivalence testing at $\|r\| < 0.05$ bounds to statistically prove the absence of meaningful correlation.
- **Non-Parametric & Non-Linear Scans**: Spearman rank correlation, Kruskal-Wallis, Mutual Information, and Gradient Boosting to detect non-linear dependencies.
- **Color Vision Deficiency (CVD) Compliant**: Color palettes (`#2a78d6, #eb6834, #1baf7a`) engineered with $\Delta E > 9.2$ across all CVD spectrums for optimal accessibility.
- **Data Quality Audits**: Cleanly handled timestamp duplicates ($2,880$ rows on `2025-03-01`), partial date files ($640$ rows on `2025-03-10`), and verified zero missing cells.

---

<div align="center">

*Developed for Smart Factory 6G Telemetry Evaluation — 2025*

</div>