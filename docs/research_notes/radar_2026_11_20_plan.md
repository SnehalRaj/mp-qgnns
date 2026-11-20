# MP-QGNN radar MVP plan — 20 November 2026

All three studies run the repository's real `TSPQGNN`, compound/RBS message
layers, exact Euclidean TSP targets and greedy decoding. **Tour gap** is
`100 * (decoded route / exact optimum - 1)`; lower is better.

## 1. Can an initialization-time gradient diagnostic predict trainability?

- Source: *Structure Makes Quantum Circuits Trainable* (13 Nov 2026).
- Mechanism: measure the variance of a real adjacency-layer gradient across
  independent initializations for five MP-QGNN structures (particle order and
  depth), then ask whether it predicts the median trained TSP-6 gap. Also measure
  the same gradient variance from TSP-4 through TSP-8.
- Go: Spearman correlation between log gradient variance and final gap is at
  most -0.6; the diagnostic-selected architecture is within one tour-gap point
  of the best; and log10 variance has size slope above -0.5 per added city.
- Stop: the correlation has the wrong sign, the selected model is more than two
  points from best, or the slope is below -1.
- Revision: if raw component variance is dominated by parameter count, rank by
  a normalized gradient signal (median norm divided by parameter count) and
  report both without selecting post hoc between them.
- Paper case: converts a representation-theoretic trainability question into a
  pre-training architecture-selection audit for WL/particle-order QGNNs.

## 2. Soft routing over fixed MP-QGNN experts

- Source: *Soft Mixture of Quantum Experts for Hybrid Quantum-Classical
  Learning* (18 Nov 2026).
- Mechanism: freeze three heterogeneous MP-QGNN message backbones (orders
  `k=1,2,3`), train only their classical edge heads and a size-free classical
  router on TSP-5, then deploy unchanged on TSP-6. Compare against every single
  expert and a uniform expert average using the same cached backbone outputs.
- Go: the revised soft router lowers median TSP-6 transfer gap by at least 10%
  versus both the best single expert and uniform averaging, wins in at least
  four of five seeds, and does not lose more than 0.5 point on TSP-5.
- Stop: median transfer does not improve, fewer than three seeds improve, or
  routing collapses to one expert while performing no better than that expert.
- Revision: the first router uses a sharp temperature; if its effective expert
  count falls below 1.5 or results are unstable, increase the temperature and
  add a small entropy bonus, keeping data and seeds fixed.
- Paper case: a train-classical/deploy-quantum graph model with input-dependent
  choice of WL/particle order and no quantum-parameter updates.

## 3. Reusable calibration of MP-QGNN readout features

- Source: *Restoring quantum neural network trainability through reusable
  calibration* (16 Nov 2026).
- Mechanism: train the real MP-QGNN ideally, apply a fixed ill-conditioned local
  response to its final node features, estimate that response once from finite
  calibration probes, and reuse the estimate for all TSP instances. Compare raw
  noisy, naive inverse and regularized inverse correction.
- Go: regularized calibration recovers at least 50% of the noise-induced tour-gap
  increase in four of five seeds and is never more than 0.5 point worse than raw
  noisy inference.
- Stop: it worsens at least three seeds or recovers under 10% of the median loss.
- Revision: replace the unstable direct pseudoinverse with a ridge inverse whose
  regularization is chosen only from held-out synthetic calibration probes.
- Paper case: tests whether calibration cost can stay fixed while one shared
  correction is reused over graphs and eventually across graph sizes.

## Common protocol

- Seeds: 0–4 for task results; additional initialization seeds for diagnostics.
- Exact TSP solutions and greedy decoding to avoid beam-search saturation.
- Existing test suite before and after changes.
- Raw JSON stores commands, splits, epochs, parameters, per-seed values,
  revisions, calibration budget and limitations.
