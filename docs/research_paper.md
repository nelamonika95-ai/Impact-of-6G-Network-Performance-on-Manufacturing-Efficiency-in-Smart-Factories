# Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories

**A connectivity-first analysis of 100,000 machine-minutes of smart-factory telemetry**

Prepared for Thales Group · Unified Mentor industrial analytics programme
Dataset: `Thales_Group_Manufacturing.csv` · 1 January – 10 March 2025 · 50 machines

---

## Abstract

This study set out to quantify how much 6G network performance — latency and packet
loss — affects manufacturing efficiency, throughput and product quality in a smart
factory. We analysed 100,000 machine-minutes of telemetry across 50 machines and 69
days, applying correlation analysis, contingency testing, threshold search,
distributional profiling, mutual information, gradient-boosted classification and
formal equivalence testing.

**The result is a null finding, and it is a strong one.** Network latency and packet
loss have no measurable relationship with any production outcome in this dataset. The
largest correlation between any network driver and any outcome is |r| = 0.0071, against
a study design powered to detect |r| ≥ 0.0089. All eight network→outcome relationships
pass two-one-sided equivalence tests at the |r| < 0.05 bound with p < 10⁻⁴¹, which
licenses the positive claim that no practically relevant effect exists — not merely that
we failed to find one. A gradient-boosted classifier given only latency and packet loss
achieves 33.33 % balanced accuracy on the three-class efficiency target: exactly chance,
and identical to the same model trained on a randomly shuffled target.

The diagnostic explanation is unambiguous. `Efficiency_Status` is a **deterministic
threshold rule over two other columns** in the same table — error rate and production
speed — which we recovered exactly, reproducing the label for 99,998 of 100,000 rows
(99.998 %). Network KPIs appear nowhere in that rule. Furthermore, all nine numeric
features are uniformly distributed (skew ≈ 0, excess kurtosis ≈ −1.2, the theoretical
signature of a uniform) and mutually independent (largest pairwise |r| = 0.0075).

We therefore report two distinct conclusions, and the distinction is the most important
contribution of this paper. First, **within this dataset**, network performance is not a
driver of manufacturing efficiency. Second, and critically, **this dataset cannot answer
the research question it was assembled to answer**, because no network→production
mechanism is encoded in it. The correct inference is about the data, not about 6G
physics. Section 8 specifies the instrumentation and experimental design that would be
required to answer the question properly.

---

## 1. Introduction

### 1.1 Background

In Industry 4.0 and emerging Industry 5.0 environments, production machinery is no
longer mechanically autonomous. Machines depend on ultra-low-latency communication for
closed-loop control; AI-driven scheduling and quality decisions depend on real-time data
exchange; and network degradation can silently reduce throughput without any mechanical
fault presenting itself. This last property is what makes connectivity analytics
valuable: a plant can lose output to a network problem while every mechanical
diagnostic reads normal.

6G network slicing promises deterministic, per-application service guarantees inside the
factory. Realising that promise requires manufacturers to know the answer to a specific
question:

> How much does network performance actually affect production efficiency and quality?

### 1.2 Problem statement

Manufacturing organisations currently lack clarity on three points:

1. whether efficiency drops are caused by network delays or by packet loss;
2. how much latency variation is tolerable before production quality degrades;
3. which efficiency levels are most sensitive to network instability.

Without this understanding, network investment may be misallocated, and production
issues may be misdiagnosed as machine faults.

### 1.3 Contribution and scope

This paper reframes manufacturing analytics through a connectivity-first lens, as the
brief intends. Its contribution is threefold:

- a complete connectivity-first analytical pipeline (profiling → network-vs-efficiency →
  latency diagnostics → packet-loss diagnostics → operation-mode interaction), executed
  as specified;
- a rigorous, quantified **null result**, established by power analysis and equivalence
  testing rather than by the absence of a significant p-value;
- a **data-validity finding** — the recovery of the label-generating rule — that explains
  the null result mechanically and determines what can and cannot be concluded.

A note on how this paper handles its own headline. A null result is easy to produce
badly: an underpowered study, a mis-specified model or a coding error all produce
"no effect". Sections 4 through 7 are therefore written as an adversarial case against
our own conclusion, testing it from independent directions — parametric and
non-parametric, linear and non-linear, global and within-stratum, correlational and
predictive. The conclusion survives all of them.

---

## 2. Data and methodology

### 2.1 Dataset

| Property | Value |
|---|---|
| Rows | 100,000 |
| Columns | 14 |
| Machines | 50 (`Machine_ID` 1–50) |
| Period | 1 Jan 2025 00:00 → 10 Mar 2025 10:39 |
| Days covered | 69 |
| Cadence | 1 minute |
| Missing cells | 0 |
| Full-row duplicates | 0 |

Fields comprise mechanical telemetry (`Temperature_C`, `Vibration_Hz`,
`Power_Consumption_kW`), network telemetry (`Network_Latency_ms`, `Packet_Loss_%`),
production outcomes (`Production_Speed_units_per_hr`,
`Quality_Control_Defect_Rate_%`, `Error_Rate_%`), an AI-derived
`Predictive_Maintenance_Score`, a categorical `Operation_Mode`, and the target
`Efficiency_Status` ∈ {High, Medium, Low}.

One schema deviation from the brief: the time column is named `Timestamp`, not `Time`.
It was read as-is and combined with `Date` into a datetime index.

### 2.2 Analytical methodology

The five-stage methodology specified in the brief was executed in full:

1. **Network performance profiling** — latency and packet-loss distributions, stability
   segmentation, and a three-level network quality banding.
2. **Network vs efficiency analysis** — efficiency distribution across network quality
   bands, efficiency shift with rising latency, sensitivity-zone identification.
3. **Latency impact diagnostics** — production speed against latency, slowdown-range
   identification, real-time vs delayed-communication comparison.
4. **Packet loss impact diagnostics** — packet loss against error rate, defect-rate
   behaviour during loss spikes, reliability-threshold search.
5. **Operation mode interaction analysis** — network impact within each operation mode,
   most-sensitive-mode identification, efficiency stability under matched mechanical
   conditions.

Because every stage returned a null, three additional stages were added to establish
whether the null was a property of the data or an artefact of the analysis:

6. **Power analysis** — the minimum effect size the design could detect.
7. **Equivalence testing (TOST)** — formal acceptance of absence at a pre-specified bound.
8. **Target reverse-engineering** — recovery of the label-generating function.

### 2.3 Statistical approach

At n = 100,000, statistical significance is nearly free: a correlation of r = 0.009 —
explaining 0.008 % of variance and of no conceivable operational interest — reaches
p < 0.05. Every result in this paper is therefore reported as an **effect size with a
confidence interval**, with p-values as secondary evidence.

Specific choices:

- **Effect sizes**: Pearson r with Fisher-z 95 % CIs; η² for continuous-by-categorical;
  Cramér's V for categorical-by-categorical; Cohen's d for two-group contrasts.
- **Proportions**: Wilson score intervals, which behave correctly in sparse bins where
  the normal approximation does not.
- **Non-parametric confirmation**: Spearman ρ alongside Pearson r, and Kruskal–Wallis
  alongside one-way ANOVA, so no conclusion rests on a normality assumption. (In this
  dataset the features are uniform, not normal, making this check essential.)
- **Non-linear detection**: mutual information and gradient-boosted trees, so a
  threshold or U-shaped relationship invisible to correlation would still be caught.
- **Class imbalance**: `Efficiency_Status` is 77.8 % "Low", so a model predicting "Low"
  unconditionally scores 77.8 % raw accuracy. **Balanced accuracy** is used throughout;
  chance is 33.3 % on this three-class problem.
- **Multiplicity**: where many tests are run (the threshold scan, the per-mode
  correlation grid), the count of significant results is compared against the count
  expected by chance rather than interpreted individually.

### 2.4 Data-quality issues and handling

| Issue | Evidence | Handling |
|---|---|---|
| `2025-03-01` holds 2,880 rows instead of 1,440 | each of 1,440 timestamps appears twice | Retained. The duplicated block does not bias the daily-mean aggregates reported here, and dropping it would silently discard 1.4 % of the extract. |
| `2025-03-10` is partial (640 rows, ends 10:39) | extract cut mid-day | Retained. All trend charts plot daily means, which are insensitive to day length. |
| 28 duplicate `(timestamp, Machine_ID)` pairs | key-uniqueness check | Flagged, not dropped — 0.028 % of rows. |
| Severe target imbalance (77.8 / 19.2 / 3.0 %) | value counts | Balanced accuracy and stratified splits used throughout. |

None of these materially affects the findings; all are reported for transparency.

---

## 3. Exploratory data analysis

### 3.1 Univariate profiling — the first red flag

| Feature | Mean | SD | Min | Max | Skew | Excess kurtosis | KS p vs Uniform |
|---|---|---|---|---|---|---|---|
| Temperature_C | 60.041 | 17.323 | 30.0 | 90.0 | −0.001 | −1.196 | 0.825 |
| Vibration_Hz | 2.550 | 1.414 | 0.1 | 5.0 | −0.000 | −1.202 | 0.862 |
| Power_Consumption_kW | 5.746 | 2.451 | 1.5 | 10.0 | −0.001 | −1.198 | 0.762 |
| Network_Latency_ms | 25.556 | 14.121 | 1.0 | 50.0 | −0.001 | −1.195 | 0.306 |
| Packet_Loss_% | 2.493 | 1.443 | 0.0 | 5.0 | 0.008 | −1.195 | 0.163 |
| Quality_Control_Defect_Rate_% | 5.009 | 2.884 | 0.0 | 10.0 | 0.000 | −1.201 | 0.468 |
| Production_Speed_units_per_hr | 275.916 | 130.097 | 50.0 | 500.0 | −0.010 | −1.205 | 0.022 |
| Predictive_Maintenance_Score | 0.499 | 0.289 | 0.0 | 1.0 | 0.003 | −1.200 | 0.390 |
| Error_Rate_% | 7.504 | 4.336 | 0.0 | 15.0 | 0.001 | −1.201 | 0.787 |

Every one of the nine numeric features is **uniformly distributed**. Skew ≈ 0 combined
with excess kurtosis ≈ −1.2 is the exact analytical signature of a uniform distribution,
whose theoretical excess kurtosis is −1.2 (= −6/5). A Kolmogorov–Smirnov test against
Uniform fails to reject for eight of nine features.

This matters because **physical measurements are never uniform**:

- machine temperature clusters around a thermal setpoint and is bounded by cooling;
- network latency is strongly right-skewed with a heavy tail — the tail is the whole
  engineering problem in real deployments;
- defect rates are low-mean and right-skewed, since most production runs are nominal;
- power draw is multimodal, tracking discrete operating states.

Uniformity across all nine features indicates each column was drawn independently from
a bounded range rather than measured from a physical process.

### 3.2 Categorical distributions

| `Operation_Mode` | n | % | | `Efficiency_Status` | n | % |
|---|---|---|---|---|---|---|
| Active | 70,054 | 70.05 | | Low | 77,825 | 77.83 |
| Idle | 20,057 | 20.06 | | Medium | 19,189 | 19.19 |
| Maintenance | 9,889 | 9.89 | | High | 2,986 | 2.99 |

The target is severely imbalanced. Note also a latent implausibility: machines in
`Idle` and `Maintenance` modes report production speeds and defect rates drawn from
the same distributions as `Active` machines. An idle machine producing 400 units/hr is
not a physically coherent record.

### 3.3 Multivariate structure — the second red flag

Full Pearson correlation matrix over the nine numeric features:

|  | Temp | Vib | Power | Latency | Loss | Defect | Speed | PMS | Error |
|---|---|---|---|---|---|---|---|---|---|
| **Temp** | 1.0000 | 0.0006 | 0.0046 | −0.0025 | 0.0029 | −0.0022 | 0.0015 | −0.0022 | −0.0018 |
| **Vib** | | 1.0000 | 0.0053 | 0.0075 | 0.0004 | 0.0004 | −0.0013 | 0.0040 | 0.0046 |
| **Power** | | | 1.0000 | −0.0032 | −0.0012 | −0.0009 | 0.0020 | 0.0016 | 0.0009 |
| **Latency** | | | | 1.0000 | −0.0069 | −0.0044 | −0.0010 | −0.0029 | 0.0001 |
| **Loss** | | | | | 1.0000 | −0.0049 | −0.0071 | 0.0008 | −0.0024 |
| **Defect** | | | | | | 1.0000 | −0.0049 | −0.0002 | −0.0015 |
| **Speed** | | | | | | | 1.0000 | 0.0035 | 0.0061 |
| **PMS** | | | | | | | | 1.0000 | 0.0045 |
| **Error** | | | | | | | | | 1.0000 |

**The largest absolute off-diagonal correlation among all 36 pairs is 0.0075.** The
features are mutually independent.

This is the second structural red flag, and it is independent of the first. In real
factory telemetry, temperature, vibration and power consumption are coupled by physics:
a machine drawing more power runs hotter and vibrates more. Observing nine mutually
independent columns in an industrial dataset is not a finding about manufacturing — it
is evidence about how the file was produced.

---

## 4. Network performance profiling

### 4.1 Distributions and the Network Stability Index

Latency spans 1.0–50.0 ms (mean 25.56, p95 47.58); packet loss spans 0–5 % (mean 2.49,
p95 4.76). Both are uniform across their range (§3.1).

We define the KPI requested by the brief:

> **Network Stability Index (NSI)** = 100 × (1 − ½ · [ (latency − min)/(max − min) +
> (loss − min)/(max − min) ]), on 0–100, higher = more stable.

Observed NSI: mean 50.01, SD 20.32, range 0.15–99.95. Network quality bands are NSI
terciles: **Low**, **Medium**, **High**.

### 4.2 Stable vs unstable period segmentation — not achievable

The brief asks us to identify stable versus unstable network periods. **This cannot be
done in this dataset, because no such periods exist.**

- Daily mean latency and daily p95 latency are flat across all 69 days; the series is
  stationary white noise with no drift, no weekly seasonality and no incident signature.
- The same holds for packet loss.
- Mean latency by hour of day shows every hour's 95 % CI overlapping every other hour's:
  there is no shift pattern and no peak-load congestion effect.

A real 6G factory network produces burst events, congestion windows and maintenance
signatures. Their complete absence here is consistent with independent per-row sampling
rather than a time series.

### 4.3 Machine-level cross-check

If network quality drove efficiency, machines with the worst average stability should
carry the most low-efficiency minutes. Across the 50 machines that correlation is
**r = +0.199 (95 % CI −0.084 to +0.453, p = 0.165)** — not significant, and pointing in
the *wrong* direction (higher stability associated with more low-efficiency minutes),
which is the signature of noise rather than of a weak true effect.

The spread in per-machine low-efficiency rate is only **76.09 %–80.43 % (SD 0.98 pp)**,
consistent with binomial sampling noise around a single common rate of 77.8 % rather
than with genuine per-machine differences.

---

## 5. Network vs efficiency analysis

### 5.1 Efficiency distribution across network quality bands

Percentage of machine-minutes in each efficiency class, by network quality band:

| Network band | High | Medium | Low |
|---|---|---|---|
| Low (worst network) | 3.030 | 19.080 | 77.890 |
| Medium | 2.922 | 19.329 | 77.749 |
| High (best network) | 3.006 | 19.158 | 77.836 |

**χ²(4) = 1.327, p = 0.857, Cramér's V = 0.0026.**

The three distributions are indistinguishable. For scale, V < 0.1 is conventionally
described as "negligible"; this is an order of magnitude below that threshold. The total
spread in low-efficiency rate between the best and worst network band is **0.14
percentage points**.

Repeating the test at finer granularity — latency deciles rather than terciles — gives
**χ²(18) = 13.85, p = 0.739, V = 0.0083**. No association appears at either resolution.

### 5.2 Latency distribution across efficiency classes

The reverse view. If latency mattered, High-efficiency minutes should show lower latency:

| Efficiency class | n | Mean latency (ms) | Mean packet loss (%) |
|---|---|---|---|
| High | 2,986 | 25.334 | 2.528 |
| Medium | 19,189 | 25.566 | 2.487 |
| Low | 77,825 | 25.562 | 2.494 |

**Kruskal–Wallis H = 0.76, p = 0.684.** Mean latency differs by 0.23 ms between the best
and worst efficiency class, inside a 1–50 ms operating envelope. η² = 0.0008 % of
latency variance is attributable to efficiency class.

### 5.3 Performance sensitivity zones — none found

Binning latency into deciles and computing the low-efficiency rate with Wilson 95 %
intervals produces a flat line: every bin's interval contains the overall mean, and the
total swing across the entire latency range is **1.75 pp** (1.49 pp for packet loss) —
within sampling noise for bins of this size. A sensitivity zone would appear as a step
or an elbow. Neither is present.

---

## 6. Latency and packet-loss impact diagnostics

### 6.1 Latency Sensitivity Score

The brief defines this KPI as efficiency change per unit latency. Implemented as an OLS
slope of each outcome on latency:

| Outcome | Slope per ms | 95 % CI | p | Effect across the full 1–50 ms span |
|---|---|---|---|---|
| Production speed (units/hr) | −0.00967 | [−0.06678, +0.04743] | 0.740 | **−0.47 units/hr** |
| Error rate (%) | +0.00003 | [−0.00187, +0.00193] | 0.975 | +0.001 pp |
| Defect rate (%) | −0.00089 | [−0.00216, +0.00038] | 0.168 | −0.04 pp |

The practical column is the last one. Moving a machine from best-case 1 ms latency to
worst-case 50 ms latency — a 50× degradation — changes predicted output by **0.47
units/hr against a mean of 275.9**, i.e. 0.17 %, with a confidence interval that
comfortably includes zero and both signs.

### 6.2 Packet Loss Impact Ratio

The brief defines this KPI as production degradation due to packet loss. Implemented as
the ratio of mean outcome in the worst packet-loss decile to the best:

| Outcome | Best decile | Worst decile | Ratio | p |
|---|---|---|---|---|
| Production speed | 278.099 | 273.810 | 0.9846 | 0.019 |
| Error rate | 7.514 | 7.504 | 0.9986 | 0.861 |
| Defect rate | 5.021 | 5.014 | 0.9985 | 0.856 |
| % Low efficiency | 77.23 % | 77.76 % | 1.0069 | — |

The production-speed row reaches p = 0.019 and deserves comment, since it is the only
nominally significant result in the diagnostics. Its effect size is a **1.5 % reduction**
corresponding to r = −0.0071, explaining **0.005 % of variance**. Among the ~40 tests
reported in this section, one result at p < 0.05 is precisely what multiplicity predicts.
It also fails to replicate within operation-mode strata (§7). We treat it as noise.

Note also that error and defect rates move in the *wrong* direction — both are marginally
**lower** in the worst packet-loss decile. A genuine reliability mechanism cannot make
packet loss protective.

### 6.3 Spike analysis — do degradation events leave a signature?

Defining a spike as the worst 5 % of minutes on each KPI and comparing against all other
minutes, across error rate, defect rate and production speed, for packet-loss spikes,
latency spikes, and simultaneous spikes:

**The largest standardised effect across all nine contrasts is Cohen's d = −0.0403.**
The conventional floor for a "small" effect is d = 0.2; the largest effect here is 5×
below that floor, and most are 50–100× below it.

**Consequence for the brief:** communication reliability thresholds cannot be derived
from this extract. A threshold requires that degradation events produce a downstream
signature to threshold against, and they produce none.

### 6.4 Exhaustive threshold search — the latency tolerance benchmark

Rather than test a handful of plausible latency limits, we tested **every** candidate
threshold from 5 ms to 47.5 ms in 2.5 ms steps (18 thresholds), comparing the
low-efficiency rate above and below each cut with Wilson intervals and a χ² test.

- Largest gap between above- and below-threshold populations: **+0.28 pp at 40.0 ms**.
- Thresholds reaching p < 0.05: **0 of 18** (expected by chance: 0.9).

**Across the full 1–50 ms envelope present in this data, no latency budget separates
good production from bad.** Any "latency tolerance number" published from this extract
would be an artefact of noise.

### 6.5 Packet-loss risk zones

| Risk zone | Minutes | % Low efficiency | Wilson 95 % CI | Mean error % | Mean defect % | Mean speed |
|---|---|---|---|---|---|---|
| 0–1 % · nominal | 20,076 | 77.56 | [76.98, 78.13] | 7.513 | 5.045 | 277.8 |
| 1–2 % · elevated | 20,130 | 77.96 | [77.38, 78.53] | 7.502 | 5.023 | 275.8 |
| 2–3 % · degraded | 20,033 | 78.13 | [77.55, 78.70] | 7.536 | 4.990 | 275.5 |
| 3–4 % · poor | 19,926 | 77.85 | [77.27, 78.42] | 7.481 | 4.962 | 275.2 |
| 4–5 % · severe | 19,835 | 77.63 | [77.04, 78.20] | 7.488 | 5.023 | 275.3 |

The low-efficiency rate stays within a **0.57 pp** total spread across all five zones and
every Wilson interval overlaps every other. Critically, the series is **not monotone** in
packet loss: it peaks in the *degraded* 2–3 % zone and falls again in the *severe* 4–5 %
zone. A genuine reliability mechanism cannot behave that way. The zones carry no
differential risk.

---

## 7. Operation mode interaction analysis

The brief asks which operation mode is most sensitive to communication delays. We
computed all 18 network-driver × outcome × mode correlations:

| Mode | n | Largest \|r\| in mode | Significant at .05 |
|---|---|---|---|
| Active | 70,054 | 0.0086 (latency → defect rate) | 2 of 6 |
| Idle | 20,057 | 0.0136 (latency → production speed) | 0 of 6 |
| Maintenance | 9,889 | 0.0076 (loss → error rate) | 0 of 6 |

**Maximum |r| within any mode: 0.0136. Significant results: 2 of 18, against 0.9
expected by chance.** No mode is network-sensitive, and the two nominally significant
cells are consistent with multiplicity.

Low-efficiency rate by mode × network band — the "efficiency stability under identical
mechanical conditions" comparison the brief requests:

| Mode | Low band | Medium band | High band |
|---|---|---|---|
| Active | 77.855 | 77.559 | 77.695 |
| Idle | 77.978 | 78.564 | 78.215 |
| Maintenance | 77.962 | 77.432 | 78.078 |

All nine cells sit within 1.1 pp of each other with no ordering by network band.

---

## 8. Why the null result is a property of the data

Four independent lines of evidence converge. Each alone would be suggestive; together
they are conclusive.

### 8.1 The study is not underpowered

At n = 100,000, with α = 0.05 two-sided:

| Power | Minimum detectable \|r\| | Variance explained |
|---|---|---|
| 80 % | 0.00886 | 0.0078 % |
| 95 % | 0.01140 | 0.0130 % |

The largest observed network→outcome correlation is **0.0071** — below the 80 % detection
floor. The analysis had ample resolution; the effects are absent, not merely undetected.

### 8.2 Equivalence testing licenses a positive claim of absence

A non-significant p-value only means "not detected". Two-one-sided-tests (TOST) reverses
the burden of proof: a small TOST p-value permits the positive conclusion that an effect
is smaller than a pre-specified bound. We set the bound at |r| < 0.05 — under 0.25 % of
variance, below any threshold of operational relevance.

| Driver | Outcome | r | 95 % CI | p (H₀: r = 0) | p (TOST) | Equivalent |
|---|---|---|---|---|---|---|
| Latency | Production speed | −0.00105 | [−0.0073, +0.0052] | 0.740 | 1.95×10⁻⁵⁴ | ✔ |
| Latency | Error rate | +0.00010 | [−0.0061, +0.0063] | 0.975 | 1.73×10⁻⁵⁶ | ✔ |
| Latency | Defect rate | −0.00436 | [−0.0106, +0.0018] | 0.168 | 1.31×10⁻⁴⁷ | ✔ |
| Latency | Maintenance score | −0.00290 | [−0.0091, +0.0033] | 0.359 | 1.48×10⁻⁵⁰ | ✔ |
| Packet loss | Production speed | −0.00712 | [−0.0133, −0.0009] | 0.024 | 2.89×10⁻⁴² | ✔ |
| Packet loss | Error rate | −0.00241 | [−0.0086, +0.0038] | 0.447 | 1.41×10⁻⁵¹ | ✔ |
| Packet loss | Defect rate | −0.00490 | [−0.0111, +0.0013] | 0.121 | 1.55×10⁻⁴⁶ | ✔ |
| Packet loss | Maintenance score | +0.00076 | [−0.0054, +0.0070] | 0.811 | 4.60×10⁻⁵⁵ | ✔ |

**All 8 of 8 relationships are statistically equivalent to zero.** This is the strongest
form the finding can take.

### 8.3 No model can extract signal that is not there

Held-out 25 % stratified test split, `HistGradientBoostingClassifier`, seed 0. Balanced
accuracy is reported because chance on this three-class problem is 33.3 %, while
predicting "Low" unconditionally scores 77.8 % raw:

| Feature set | Accuracy | Balanced accuracy | Reading |
|---|---|---|---|
| Majority class only | 0.7782 | 0.3333 | the baseline to beat |
| **Latency + packet loss only** | **0.7782** | **0.3333** | **exactly the baseline — zero information** |
| All 9 features, target shuffled | 0.7782 | 0.3333 | permutation control |
| Error rate + production speed | 0.9983 | 0.9931 | two columns recover the label |
| All 9 features | 0.9984 | 0.9941 | the other 7 add +0.01 pp |

The network-only model performs **identically to the same model trained on a randomly
shuffled target**. Gradient boosting detects thresholds, interactions and non-monotone
relationships; had any existed, this row would exceed chance. Mutual information agrees:
I(latency; efficiency) = 0.000 nats against a target entropy of 0.617 nats.

### 8.4 The target is a deterministic rule over two other columns

This is the mechanical explanation. A depth-4 decision tree fitted to `Error_Rate_%` and
`Production_Speed_units_per_hr` alone reproduces `Efficiency_Status` at **99.998 %
in-sample accuracy**. Reading the thresholds off that tree yields an exact rule:

```
High    if Error_Rate_% ≤ 2   and Production_Speed > 400 units/hr
Medium  if Error_Rate_% ≤ 5   and Production_Speed > 200 units/hr
Low     otherwise
```

Applied to all 100,000 rows, this rule reproduces the label for **99,998 rows**. The two
exceptions lie exactly on `Error_Rate_% == 5.000`, i.e. they are floating-point boundary
ties, not model error.

Confirming evidence from η² — the share of each feature's variance explained by
efficiency-class membership:

| Feature | η² (%) | | Feature | η² (%) |
|---|---|---|---|---|
| **Error_Rate_%** | **38.358** | | Power_Consumption_kW | 0.0013 |
| **Production_Speed_units_per_hr** | **11.277** | | Predictive_Maintenance_Score | 0.0012 |
| Quality_Control_Defect_Rate_% | 0.0037 | | **Network_Latency_ms** | **0.0008** |
| Packet_Loss_% | 0.0021 | | Vibration_Hz | 0.0000 |
| Temperature_C | 0.0015 | | | |

The two rule inputs are separated from every other feature by roughly **four orders of
magnitude**. Latency and packet loss sit alongside vibration and temperature at η² <
0.01 %.

The decision boundary, plotted in error-rate × speed space, consists of three
axis-aligned rectangles with no overlap and no fuzzy margin. Real efficiency ratings
have overlapping class distributions, because they depend on factors not recorded in the
table. A boundary this crisp is the signature of a generated label.

**`Efficiency_Status` contains no information beyond `Error_Rate_%` and
`Production_Speed_units_per_hr`. Network KPIs cannot predict it because it was not
constructed from them.**

---

## 9. Interpretation: two conclusions that must not be conflated

**Conclusion 1 — about the dataset.** Within `Thales_Group_Manufacturing.csv`, network
latency and packet loss have no measurable relationship to manufacturing efficiency,
throughput, error rate or defect rate. This is established at high power and confirmed
by equivalence testing.

**Conclusion 2 — about the research question.** This dataset **cannot** answer whether
6G network performance affects manufacturing efficiency, because no network→production
mechanism is encoded in it. The features are independently drawn uniforms; the target is
a deterministic rule over two of them.

Conflating these would be a serious error in either direction. Reporting
*"6G latency does not affect manufacturing efficiency"* as a finding about the physical
world is **not supported** by this analysis — that is a substantive engineering claim,
and a dataset with no encoded mechanism provides no evidence for it any more than
against it. Equally, the analysis should not be presented as inconclusive: it is
conclusive about the data, and the data-validity finding is itself actionable.

The professional value delivered here is diagnostic. An analyst who reported latency
tolerance benchmarks from this extract — and the numbers are easy to produce; §6.4 shows
a "+0.28 pp effect at 40 ms" available to anyone who stops before computing a confidence
interval — would have handed Thales a network investment case built on noise. The
recovery of the labelling rule is what converts an inconclusive study into a definite
answer about why it is inconclusive.

---

## 10. Recommendations

### 10.1 On network investment (immediate)

1. **Do not fund latency reduction on the strength of this dataset.** Between 1 ms and
   50 ms it shows no measurable production benefit, and equivalence testing rules out
   any effect above 0.25 % of variance. Equally, do not *cancel* planned 6G investment
   on this basis — the dataset is silent on the physics, not evidence against it. The
   business case must come from instrumented trials (§10.3).
2. **Do not use this dataset to diagnose live production issues.** Its most likely
   operational harm is a false negative: an analyst concluding "the network is never the
   problem" and directing maintenance attention to mechanical causes.

### 10.2 On production levers actually visible in the data

3. **Error rate is the dominant discriminator between efficiency classes** (η² = 38.4 %,
   against < 0.01 % for latency) and gates every High rating: High requires error rate
   ≤ 2 %. Reducing error rate below 2 % is the single highest-leverage change available.
4. **Production speed is the second discriminator** (η² = 11.3 %). The class boundaries
   sit at 200 units/hr (Low/Medium) and 400 units/hr (Medium/High). A machine at
   395 units/hr with a 1.5 % error rate is one small throughput gain from a High rating.
5. Treat these as **definitional, not causal**. They describe how the label is
   constructed, which is genuinely useful for target-setting and for auditing how
   "efficiency" is scored — but improving a KPI that appears in the scoring rule is not
   the same as improving the underlying plant.

### 10.3 On instrumentation and study design (to actually answer the question)

The question in the brief is answerable, but not with cross-sectional per-minute
snapshots of independently varying columns. Four requirements:

6. **Paired causal timing.** Network KPIs must be measured on the same control loop as
   the production outcome they are claimed to affect, and timestamped *ahead* of it.
   Without temporal precedence, cause cannot be separated from coincidence even where a
   correlation exists.
7. **Realistic driver variation.** A uniform 1–50 ms sweep is a synthetic design, not
   observed 6G behaviour. Real deployments show bursty tail latency, and it is the tail —
   p99 and p99.9, not the mean — that breaks closed-loop control. Capture and analyse
   tail percentiles per control cycle.
8. **Mechanism-bearing outcomes.** Efficiency here is a threshold on error rate and
   speed. A network study needs outcomes the network can physically touch: control-loop
   deadline misses, command retransmissions, cycle-time jitter, AGV re-route counts,
   robot emergency-stop events.
9. **An intervention, not an observation.** Network slice configuration is directly
   controllable, so this can be an experiment rather than a correlational study.
   Deliberately vary the slice SLA (latency budget, loss tolerance, bandwidth guarantee)
   across matched production cells and measure the difference in output. This converts an
   unanswerable observational question into a causal one, and it is the only design that
   will produce a defensible network investment case.

### 10.4 On data governance

10. **Label the extract's provenance.** This file carries the structural signatures of
    synthetic generation (uniform marginals, independent columns, a deterministic
    target). Distributing it without that label invites downstream analyses that
    mistake generation artefacts for industrial findings. If it is intended as a
    pipeline-development or teaching asset, it serves that purpose well — and should be
    marked as such.
11. **Adopt equivalence testing as standard practice** for connectivity analytics at
    this sample size. At n = 100,000 the conventional significance test is nearly
    uninformative; without a pre-specified relevance bound, teams will keep reporting
    0.005 %-of-variance effects as findings.

---

## 11. Limitations

- **Single extract, no external validation.** All conclusions describe this file. No
  independent telemetry was available to cross-check whether the null generalises.
- **Provenance inferred, not documented.** We infer synthetic generation from
  distributional evidence (§3.1, §3.3, §8.4). No generation script or data dictionary
  was supplied to confirm it. The evidence is strong but circumstantial.
- **The equivalence bound is a judgement.** We set |r| < 0.05. A stakeholder who
  considered r = 0.02 operationally meaningful would need the tests re-run at that bound;
  they would still pass, since observed |r| ≤ 0.0071, but the choice should be explicit.
- **69 days is short for seasonality.** Even had a time-series signal existed, this
  window could not resolve quarterly or seasonal network patterns.
- **Duplicated and partial days retained.** §2.4 documents the 2025-03-01 duplication
  and the partial 2025-03-10. Both were retained deliberately; a sensitivity analysis
  excluding them was not run, as neither affects daily-mean aggregates.

---

## 12. Conclusion

This project set out to demonstrate that network performance is a critical, independent
driver of manufacturing efficiency in 6G-enabled smart factories. The analysis cannot
support that conclusion from this dataset — and establishing that rigorously is the
finding.

Latency and packet loss show no measurable relationship to efficiency, throughput or
quality: largest |r| = 0.0071 against a 0.0089 detection floor, χ²(4) p = 0.857 across
network quality bands, 0 of 18 candidate latency thresholds significant, 8 of 8
relationships formally equivalent to zero, and a network-only classifier at exactly
chance. The mechanical reason is that `Efficiency_Status` is a deterministic threshold
rule over error rate and production speed, recovered here at 99.998 % accuracy, in which
network KPIs play no part.

The connectivity-first framing the brief proposes remains the right one, and the
analytical pipeline built here — profiling, banding, threshold search, equivalence
testing, and target reverse-engineering — is directly reusable on instrumented data. What
this study delivers to Thales and its industrial partners is therefore twofold: a
validated pipeline, and a clear specification (§10.3) of the paired, tail-aware,
intervention-based measurement required before any 6G network investment case can honestly
claim a production-efficiency return.

---

## Appendix A — Reproducibility

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python analysis/01_profile.py         # structure, missingness, distributions
python analysis/02_signal_test.py     # correlation, chi2, MI, model ceiling
python analysis/03_label_rule.py      # recovers the labelling rule
python analysis/04_kpis_and_power.py  # KPIs, power, TOST, artifact export

python -m pytest tests -q             # 40 dashboard smoke tests
streamlit run app.py                  # interactive dashboard
```

All random seeds are fixed at 0. `analysis/04` writes every table cited in this paper to
`outputs/` as CSV, plus `outputs/findings.json` with the headline figures.

## Appendix B — Artifact index

| File | Contents |
|---|---|
| `outputs/findings.json` | headline figures: rule, power, η², benchmarks, integrity |
| `outputs/univariate_summary.csv` | §3.1 distribution table incl. KS uniformity tests |
| `outputs/correlation_matrix.csv` | §3.3 full correlation matrix |
| `outputs/tost_equivalence.csv` | §8.2 equivalence tests |
| `outputs/latency_breakpoint_scan.csv` | §6.4 threshold scan |
| `outputs/efficiency_by_network_band.csv` | §5.1 contingency shares |
| `outputs/operation_mode_network_correlations.csv` | §7 per-mode correlation grid |
| `outputs/mode_band_pct_low.csv` | §7 mode × band low-efficiency rates |
| `outputs/machine_rollup.csv` | §4.3 per-machine scorecard |
| `outputs/daily_trend.csv` | §4.2 daily network and production KPIs |
| `outputs/analysis_ready.parquet` | cleaned analysis table with NSI and bands |
