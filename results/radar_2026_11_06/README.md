# Quantum ML radar MVPs — 6 November 2026

## How to read the metric

`tour gap = 100 * (decoded tour length / exact optimum - 1)`.
A 0% gap is an optimal tour; 5% means the route is 5% longer than optimal.
The pilot used beam-5 decoding and saturated near 0%, so all three experiments
were repeated with greedy decoding on harder exact TSP instances. Pilot JSON is
retained beside the revised result.

## Decisions

| Project | Proposed mechanism | Revised result | Decision |
|---|---|---|---|
| Residual-noise selection | Train with RBS over-rotation noise; choose the noise on validation; evaluate with 512 shots | Median TSP-6 noisy gap worsened from 3.94% to 4.62% (-17.3% relative improvement); all three seeds worsened | **STOP** |
| Data-informed depth | Select depth from the empirical pure-state QFIM spectrum before training | Raw rank grew 15/30/45 and effective rank 9.72/13.18/15.65; it selected depth 3, while depth 2 had the best median gap (2.87% vs 3.66%) with 18% fewer parameters | **STOP** for the selection rule; retain depth 2 as an empirical default |
| Conditional cross-size model | Train shared parameters on TSP-5/6, condition on size, deploy on unseen TSP-8 | Unconditioned joint training was best at 7.08% median gap; bounded size conditioning gave 8.10% and improved only one of three seeds | **STOP** for explicit size tokens; **PIVOT** toward shared unconditioned training |

These are feasibility tests, not hardware or advantage claims. The noise model
is Gaussian coherent angle error plus Bernoulli shots; the conditional encoder is
circuit-compatible input preparation but was not run on a QPU.

## Exact commands

```bash
python -m pytest -q
python experiments/radar_2026_11_06.py noise --out results/radar_2026_11_06/noise.json
python experiments/radar_2026_11_06.py depth --out results/radar_2026_11_06/depth.json
python experiments/radar_2026_11_06.py conditional --out results/radar_2026_11_06/conditional.json
python -m pytest -q
```

Seeds are `0,1,2`. Dataset sizes, loss traces, parameter counts, spectra, shot
budget, selected settings, and per-seed metrics are in the JSON files.

