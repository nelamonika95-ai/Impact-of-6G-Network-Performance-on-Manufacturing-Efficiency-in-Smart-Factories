# Executive Summary — for Government and Public-Sector Stakeholders

## 6G Network Performance and Manufacturing Efficiency in Smart Factories

**Study scope:** 100,000 machine-minutes of smart-factory telemetry · 50 industrial
machines · 1 January – 10 March 2025
**Prepared for:** Thales Group and industrial policy stakeholders
**Full technical report:** [`docs/research_paper.md`](research_paper.md)
**Interactive dashboard:** `streamlit run app.py`

---

## The question we were asked

Industrial 6G is a strategic infrastructure priority. Before public or private capital is
committed to ultra-low-latency factory networks, decision-makers need to know how much
network performance actually affects production efficiency and product quality — and how
much latency a factory can tolerate before output suffers.

This study was commissioned to answer that question from operational telemetry.

## What we found

**The dataset provided cannot answer the question, and we can now say precisely why.**

Across every test applied, network latency and packet loss showed **no measurable
relationship** to production efficiency, throughput, error rates or defect rates:

| Test | Result | Plain reading |
|---|---|---|
| Correlation, network → production | Largest \|r\| = **0.0071** | Explains 0.005 % of variation |
| Efficiency mix across network quality bands | χ² p = **0.857** | The three bands are indistinguishable |
| Latency tolerance thresholds tested | **0 of 18** significant | No tolerance limit exists in the data |
| Packet-loss risk zones (nominal → severe) | **0.57 pp** total spread, not monotone | Severe loss is no worse than nominal |
| AI model using only network data | **33.3 %** balanced accuracy | Exactly random guessing |
| Formal equivalence tests | **8 of 8 pass** | Effects are provably negligible, not just undetected |

The decisive diagnostic: the dataset's own efficiency rating is a **fixed arithmetic rule
applied to two other columns in the same table** — error rate and production speed. We
recovered that rule exactly, and it reproduces the efficiency rating for **99,998 of
100,000 records**:

> Rated **High** if error rate ≤ 2 % and speed > 400 units/hour
> Rated **Medium** if error rate ≤ 5 % and speed > 200 units/hour
> Rated **Low** otherwise

Network performance appears nowhere in that rule. Supporting evidence points the same way:
all nine measured variables follow flat, evenly-spread distributions unlike any real
physical measurement, and all are statistically independent of one another — including
temperature, vibration and power consumption, which in a real machine are physically
coupled.

**The dataset carries the structural signatures of synthetically generated data.** No
network-to-production relationship was built into it, which is why none can be found.

## What this does and does not mean

This distinction is the single most important point in this summary.

**✔ Supported by this study:** Within this dataset, network performance does not affect
manufacturing efficiency. The analysis was statistically powerful enough to detect
effects far smaller than any of operational interest, and formal equivalence testing
rules out effects above 0.25 % of variation.

**✘ NOT supported by this study:** *"6G latency does not affect manufacturing."* That is a
claim about engineering reality, and this dataset provides no evidence for it — or against
it. A dataset with no built-in relationship is silent on the physics, not a verdict on it.

We flag this explicitly because the misreading is consequential in both directions. Cited
as evidence that connectivity does not matter, this study could be used to justify
cancelling warranted industrial network investment. Cited as evidence of a latency
tolerance limit — and a plausible-looking number is easy to extract if the analysis stops
before confidence intervals are computed — it could justify spending against a threshold
that is pure statistical noise.

## Recommendations

### Immediate

1. **Do not use this dataset as a basis for 6G network investment decisions**, in either
   direction. It cannot support a business case for or against.
2. **Do not use it for live production diagnostics.** The most likely operational harm is
   a false negative — concluding "the network is never the problem" and misdirecting
   maintenance effort toward mechanical causes.
3. **Label the extract's provenance.** Distributed without a synthetic-data marker, it
   invites downstream analyses that mistake generation artefacts for industrial findings.
   As a pipeline-development or training asset it is entirely fit for purpose, and should
   be marked as such.

### To answer the original question — required instrumentation

The question is answerable. It requires measurement that this extract does not provide:

4. **Paired causal timing.** Network measurements must sit on the same control loop as the
   production outcome, and be timestamped *before* it. Cross-sectional per-minute
   snapshots cannot separate cause from coincidence.
5. **Tail metrics, not averages.** Real networks fail in bursts. It is the worst 1 % of
   latency events (p99, p99.9) that break closed-loop machine control, not the mean.
   Average latency is close to the wrong metric entirely.
6. **Outcomes the network can physically affect.** Control-loop deadline misses, command
   retransmissions, cycle-time jitter, automated-vehicle re-routes, robot emergency stops
   — measures with a direct physical path from a dropped packet to a production
   consequence.
7. **Run it as an experiment, not an observation.** Network slice configuration is directly
   controllable, which makes a controlled trial feasible: deliberately vary the guaranteed
   service level across matched production cells and measure the difference in output.
   This is the only design that yields a defensible causal figure — and therefore the only
   one that can properly justify infrastructure spend.

### Policy-level observation

8. **Analytical standards matter as much as infrastructure standards.** At datasets of
   this size, conventional statistical significance is nearly meaningless: a relationship
   explaining 0.008 % of variation still registers as "statistically significant". Public
   programmes funding industrial-digitalisation analytics should require **effect sizes
   with confidence intervals, and pre-specified relevance thresholds**, not p-values
   alone. Without that requirement, negligible effects will continue to be reported as
   findings, and genuine null results — which are often the more valuable output — will
   continue to be presented as inconclusive.

## What was delivered

Although the headline finding is a null result, the analytical infrastructure built for
this study is complete and directly reusable on properly instrumented telemetry:

- **A six-module interactive dashboard** covering network performance profiling,
  network-versus-efficiency analysis, quality and error impact, 6G optimisation
  benchmarking, efficiency diagnostics, and a data-integrity audit — with filters for
  network quality, efficiency class, operation mode, time window, hour of day and machine.
  Every panel reports effect sizes with confidence intervals and the smallest effect that
  slice could have detected.
- **A four-stage reproducible analysis pipeline** with fixed random seeds, exporting every
  cited figure to versioned CSV and JSON.
- **Forty automated tests** verifying every dashboard module behaves correctly on
  degenerate data slices.
- **The four KPIs specified in the brief**, implemented and computed: Network Stability
  Index, Latency Sensitivity Score, Packet Loss Impact Ratio, and Network-Efficiency
  Correlation.

Point the same pipeline at telemetry meeting the requirements in items 4–7 above, and it
will produce the latency tolerance benchmarks and packet-loss risk thresholds this brief
originally called for.

---

### One-paragraph summary

We tested whether 6G network latency and packet loss affect manufacturing efficiency
across 100,000 machine-minutes of factory telemetry. They do not — not weakly, but
provably negligibly, confirmed by formal equivalence testing at high statistical power.
The reason is that the dataset's efficiency rating is a fixed arithmetic rule over two
other columns (error rate and production speed), which we recovered exactly at 99.998 %
accuracy, and in which network performance plays no part; the data shows the structural
signatures of synthetic generation throughout. **The correct conclusion is about the
dataset, not about 6G engineering.** No network investment decision should rest on this
extract. Answering the original question requires paired, tail-aware, experimentally
controlled measurement — specified in items 4 through 7 above — and the analytical
pipeline built for this study is ready to run against it.
