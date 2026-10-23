# MP-QGNN radar results, 23 October 2026

## Plain-English outcome

Three real-code MVPs were tested. Two should stop. One should pivot into a
larger readout study.

"Tour excess" means how much longer the predicted tour is than the exact best
tour. For example, 0.05 means 5% longer. This is the main task metric below.

![Summary plot](../../figures/radar_2026_10_23_summary.png)

## Decisions

| Project | Main evidence | Decision | Paper / novelty / likely impact (0–10) |
|---|---|---|---:|
| Graph-size Fourier query | In the first full run it improved unseen-size tour excess by 0.0074. In the larger alignment control it was 0.0251 worse than the baseline, and the correct query was 0.0299 worse than fixing the query at size 5. | **STOP**: the gain is not caused by size alignment. | 2 / 4 / 3 |
| Margin-based model selection | The structured margin reached rho = -0.593 at size 6, but fell to -0.148 at size 7 and -0.044 at size 8. Bootstrap intervals cross zero at sizes 7 and 8. It selected models 0.0014 worse than validation loss in the main run. | **STOP**: the statistic does not transfer across graph sizes. | 3 / 4 / 4 |
| Confidence-stopped readout | At a 2,048-shot cap, adaptive readout used 5.41x fewer binary shots and added only 0.00052 tour excess, or 0.052 percentage points. At 8,192, it used 8.91x fewer shots and added 0.00042. | **PIVOT**: strong resource/utility result, but exact-path agreement is only 79% and 88%, almost the same as fixed shots, because many edge scores are nearly tied. | 7 / 6 / 7 |

## Project 1: graph-size Fourier query

### Mechanism and revision

The first model used a high-frequency Fourier encoding of graph size only at the
final node embeddings. Its unseen-size result was slightly worse than the
baseline. The revision moved a smoother low-frequency query inside every
message-passing layer, where it could change how re-uploaded features were
combined before edge mixing.

### Evidence

- Main full run: 4 seeds, 80 exact TSP instances per training size, 80 epochs at
  each of sizes 4 and 5, 48 test instances at sizes 4–7.
- Larger falsification run: 6 seeds, 100 instances and 100 epochs at each
  training size, 64 test instances at unseen sizes 6–8.
- The crucial control replaced the true target-size query with a fixed size-5
  query. The fixed, mismatched query was better in 17 of 18 seed-size pairs.

This rules out the intended task-alignment mechanism in this implementation.
The result does not rule out richer queries based on geometric sampling or a
query inserted directly into the quantum mixing angles.

## Project 2: margin-based cross-size selection

### Mechanism and revision

The first statistic was a generic lower-tail binary edge margin. It failed due
to class imbalance and selected a much worse model in one pilot seed. The
revision used a structured margin: at each step of the exact tour, compare the
logit of the correct next edge with the best remaining incorrect edge, then
take a scale-normalized 10th percentile.

### Evidence

- Main full run: 5 seeds x 3 depths, 100 training graphs, 60 validation graphs,
  48 unseen size-7 graphs, 90 epochs.
- Robustness run: 6 seeds x 3 depths, 120 training graphs, 72 validation graphs,
  56 graphs at each unseen size 6, 7 and 8, 110 epochs, 2,000 bootstrap draws.
- The desired negative correlation appears at size 6 but disappears at 7 and 8.

The paper's general margin result remains credible for its setting. This result
says the tested margin is not a reliable selector for MP-QGNN graph-size
extrapolation.

## Project 3: confidence-stopped readout

### Mechanism and revision

Every real MP-QGNN edge probability is treated as a Bernoulli expectation. A
fixed policy measures every candidate edge equally. The adaptive policy samples
in batches of 32 and removes candidates whose confidence interval cannot beat
the current best edge. The first Hoeffding rule saved 2.1x shots but agreed with
the exact-probability path only 78% of the time. The revision used a
variance-aware 95% interval and swept the maximum budget.

### Pareto result

| Maximum shots per edge | Shot reduction | Added tour excess | Adaptive exact-path agreement | Fixed exact-path agreement |
|---:|---:|---:|---:|---:|
| 128 | 1.91x | -0.134 percentage points | 47.9% | 47.0% |
| 512 | 3.28x | 0.084 percentage points | 64.4% | 65.0% |
| 2,048 | 5.41x | 0.052 percentage points | 78.9% | 78.9% |
| 8,192 | 8.91x | 0.042 percentage points | 87.6% | 88.3% |

The exact path is a poor sole metric because near-tied edges can change the path
without materially changing its length. The next experiment should report both
task regret and a tie-aware decision criterion, then validate the measurement
model against the PennyLane unary-register circuit.

## Exact commands

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python experiments/radar_2026_10_23.py pilot --output results/radar_2026_10_23_pilot.json
python experiments/radar_2026_10_23.py pilot --output results/radar_2026_10_23_pilot_revised.json
python experiments/radar_2026_10_23.py full --output results/radar_2026_10_23_full.json
python experiments/radar_2026_10_23_robustness.py --output results/radar_2026_10_23_robustness.json
python experiments/plot_radar_2026_10_23.py
python -m pytest -q
```

Raw machine-readable results are in `results/`. All data are exact random TSP
instances from `mp_qgnns.data_io.tsp.random_instances`. Model and measurement
seeds, sample counts, epoch counts and thresholds are recorded in the JSON.

## Caveats

- The experiments use exact reduced-basis simulation, not a QPU.
- The readout noise model samples edge expectations independently. A publishable
  hardware claim requires compiling the observable family to the PennyLane
  circuit and including covariance, device noise, mapping and mitigation.
- The graph-size query is much weaker than the geometry-derived query in the SAR
  paper, so this is a stop decision for this mechanism, not for task-aligned
  spectral queries generally.
- TSP sizes stop at eight cities because exact labels are generated by brute force.

## Collaboration trigger

The readout result is reproducible enough for a technical question, but not yet
for an email claiming a hardware result. The best contacts are Connor Lenihan
and David Z. Manrique. A concrete question is whether their covariance-aware
mixed-estimator allocation can be specialized to correlated MP-QGNN edge
observables while preserving the observed task-regret savings. No email was sent.
