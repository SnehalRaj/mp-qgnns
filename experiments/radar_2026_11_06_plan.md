# MP-QGNN radar experiments — 2026-11-06 pre-registration

All three MVPs use the repository's real `TSPQGNN`, exact TSP tours, reduced-basis
RBS/compound layers, and the circuit-equivalence tests already in the repository.
The primary endpoint is the **tour gap**, `100 * (tour_length / optimum - 1)`;
zero is optimal and lower is better. Results are preliminary until repeated on
larger benchmark splits and quantum hardware.

## 1. Residual-noise selection (GENIE-inspired)

- Source: *Noise-regularized training of quantum circuit Born machines under a
  finite shot budget* (arXiv:2611.06630, 2026-11-04).
- Mechanism: train the MP-QGNN while applying zero-mean over-rotation noise to
  every graph-conditioned SingleExcitation angle. Select the training noise on a
  held-out validation set, then evaluate with a fixed 512-shot budget.
- Baseline: clean training, evaluated through the same noisy, finite-shot path.
- Go: at least 10% relative reduction in median noisy tour gap over three seeds,
  with no more than two percentage points of clean-gap degradation.
- Stop: less than 3% median improvement, or all seeds worsen.
- Paper case: an observable, fixed-budget noise-selection rule for an equivariant
  message-passing quantum model, with cross-size deployment as the next stage.

## 2. Data-informed depth selection

- Source: *Data informed Ansatz Selection for Variational Quantum Circuits*
  (arXiv:2611.05079, 2026-11-03).
- Mechanism: for depths 1–3, compute the empirical pure-state QFIM of normalized
  MP-QGNN node states with respect to the compound-layer parameters, then compare
  its numerical/effective rank with trained TSP performance.
- Baseline: exhaustive depth sweep under the same data and optimizer budget.
- Go: the smallest depth reaching 95% of the maximum effective QFIM rank is
  within 0.02 absolute tour ratio of the best depth while using at most 70% of
  its trainable parameters.
- Stop: the rank is saturated/misleading or selects a model more than 0.05 worse.
- Planned revision if raw rank saturates: use the stable/effective rank of the
  QFIM spectrum and report the failure of the raw-rank rule explicitly.
- Paper case: replaces architecture grid search with a pre-training structural
  diagnostic tied to the real quantum state map.

## 3. Train-classical / deploy-larger conditional MP-QGNN

- Source: *Classical training of conditional instantaneous quantum
  polynomial-time Born machines* (arXiv:2611.05483, 2026-11-03).
- Mechanism: share every learned MP-QGNN module across TSP-5 and TSP-6, encode
  graph size at the input, and deploy the same learned circuit parameters on
  unseen TSP-7. Compare raw-size conditioning with a bounded normalized size
  feature if the first attempt extrapolates poorly.
- Baselines: TSP-5-only transfer and unconditioned joint TSP-5/TSP-6 training.
- Go: at least 10% relative reduction in median TSP-7 tour gap versus both
  baselines over three seeds, without worsening seen-size gap by more than 0.02.
- Stop: less than 3% improvement over the stronger baseline, or inconsistent
  direction across seeds.
- Paper case: a direct train-small/deploy-larger test of shared, circuit-realizable
  message passing rather than a separate model at each graph size.

## Common protocol

- Seeds: 0, 1, 2.
- Exact uniform Euclidean TSP: fixed generated train/validation/test splits.
- Tests: run the full 43-test suite before and after changes.
- Raw outputs: JSON records with commands, package/git revision, data sizes,
  per-seed metrics, selected hyperparameters, and caveats.

