# MP-QGNN radar experiment plan, 16 October 2026

Baseline: `43 passed` on the clean `main` branch at commit `8515375`.

## Project 1: Does effective dimension predict real MP-QGNN utility?

**Source idea.** Mia, Khandakar, and Ayari argue that the diagonal-QFIM participation ratio describes local circuit geometry, not trainability or model quality.

**Proposed test.** Train the real TSPQGNN at depths 1, 2, and 3 and data sizes 40, 80, and 160. For every run, compute a task-conditioned diagonal empirical-Fisher participation ratio from per-example gradients of the real balanced edge loss. Compare it with greedy tour excess, test loss, and the train-test gap across seeds.

**Go metric.** Treat the scalar as useful for MP-QGNN model selection only if its Spearman correlation with greedy tour excess has absolute value at least 0.7, has a stable sign in leave-one-seed-out checks, and beats parameter count as a predictor.

**Stop metric.** Stop using it as a model-quality headline if absolute correlation is below 0.4, changes sign across seeds, or is no better than parameter count. It may still be reported as a geometry diagnostic.

**Paper value.** A controlled negative result would prevent an attractive but invalid capacity/trainability claim in a graph-QML setting.

## Project 2: Is localized readout genuinely more trainable?

**Source idea.** Q-Capsule attributes stable gradients partly to register partitioning and local readout.

**Proposed test.** On randomly initialized real TSPQGNNs, measure gradient variance of the first compound-layer parameter for a uniformly selected single-edge loss and for the all-edge mean loss as city count grows. Repeat after multiplying the global mean gradient by the square root of the number of undirected edges, which removes the variance reduction expected from averaging independent terms. Also retain the edge sum as a scale check.

**Go metric.** Pursue a locality claim only if the local-loss variance decays at least ten times more slowly than the square-root-edge-rescaled global loss over the tested range and the slope difference is consistent across seeds.

**Stop metric.** If the advantage disappears after rescaling, classify it as a loss-normalization effect rather than barren-plateau mitigation.

**Paper value.** This is a mechanism-matched control for importing capsule-style claims into message-passing QGNNs.

## Project 3: Can training-set size select MP-QGNN depth?

**Source idea.** Nowmi, Kamol, and Rahman find no predictable data-size/circuit-size law for adaptively grown classifiers.

**Proposed test.** Use the depth and data-size grid from Project 1. Select depth by validation loss for each data size and compare the selected depth, active parameter count, and generalization gap. Then test a progressive 1-to-2-layer curriculum against a two-layer model trained from scratch under the same total epoch budget.

**Go metric.** A useful growth rule requires selected depth to be nondecreasing with data size in at least 80% of seed transitions and progressive growth to reduce greedy tour excess by at least 0.5 percentage points without increasing total epochs.

**Stop metric.** Stop the adaptive-depth story if selected depth is non-monotone or seed-dominated and the curriculum gain is below 0.2 percentage points.

**Paper value.** A positive result would give a practical small-data-to-deeper-model schedule; a negative result would justify cross-size transfer and validation-based stopping rather than data-size heuristics.
