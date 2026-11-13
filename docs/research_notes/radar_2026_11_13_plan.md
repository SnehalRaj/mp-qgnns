# MP-QGNN radar MVP plan — 13 November 2026

All experiments use the repository's real `TSPQGNN`, exact TSP tours, shared
RBS/compound layers and the standard greedy-tour metric. The reported **tour
gap** is `100 * (decoded length / exact optimum - 1)`; lower is better.

## 1. Permutation invariance versus shortcut learning

- Source: *Invariance can prevent overfitting and shortcut learning in Quantum
  Neural Networks* (arXiv:2611.14368, 10 Nov 2026).
- Mechanism: compare the standard coordinate-canonical RBS edge ordering with a
  label-lexicographic ordering. The models have the same embedding, RBS gates,
  trainable parameter count and edge-angle network. Train on graphs whose labels
  are sorted by x-coordinate, then test both sorted and randomly relabelled copies.
- Go: on randomly relabelled TSP-6, the invariant model has at least 20% lower
  median tour gap than the broken model; its relabelling penalty is <=1 percentage
  point while the broken model's is >=3 points; sorted-test gaps differ by <=2 points.
- Stop: the relative advantage is below 5%, or the invariant model itself suffers
  a >2-point relabelling penalty.
- Paper case: isolates the benefit of S_n equivariance on a controlled graph
  shortcut without changing parameter count or the quantum message primitive.

## 2. Neutral-atom quantum graph features inside MP-QGNN

- Source: *Neutral-atom quantum features as complementary structural encodings
  for graph learning* (arXiv:2611.08821, 6 Nov 2026; NeurIPS 2026).
- Mechanism: exactly simulate the paper's graph-dependent Rydberg Hamiltonian on
  small Euclidean TSP graphs, extract connected occupation correlations under
  AFM-like and FM-like quenches, and inject node/edge features into the real
  MP-QGNN encoder and RBS-angle network.
- Baseline: unchanged coordinate-only `TSPQGNN` at the same train/test split.
- Go: with 30 training graphs, the hybrid reduces median greedy TSP-6 tour gap by
  >=15% (and >=2 percentage points) over three seeds with <50% extra parameters.
- Stop: improvement is <5%, reverses in two seeds, or parameter growth exceeds 50%.
- Evidence-driven revision: if the full time series overfits or is unstable,
  replace it by per-node/per-edge mean and standard deviation, matching the paper's
  compressed-statistics direction, and rerun the same seeds.
- Paper case: a direct comparison between hardware-derived structural features
  and learned quantum message passing, including whether they are complementary.

## 3. V-usable-information sample selection

- Source: *Which data samples matter when training a quantum model?*
  (arXiv:2611.13103, 9 Nov 2026).
- Mechanism: train a short probe and a null-input control, score each graph by
  pointwise usable information (null loss minus input-conditioned loss), and train
  the real MP-QGNN on half of the available TSP-5 graphs.
- Baselines: uniform half-data selection and the full-data model.
- Go: the selected half-data model is within 1 percentage point of the full-data
  TSP-5 gap and improves median TSP-6 transfer gap by >=10% relative to uniform.
- Stop: transfer improvement is <3%, inconsistent in direction, or in-size gap
  worsens by >2 points.
- Evidence-driven revision: if top-score selection concentrates on outliers, use
  a coverage-preserving mixture with half high-score and half sampled uniformly
  across the remaining score quantiles.
- Paper case: a data-centric route to reducing QPU training calls while testing
  whether the selected small graphs preserve cross-size transfer.

## Common protocol

- Seeds: 0, 1, 2.
- Exact Euclidean TSP targets; greedy decoding avoids beam-search saturation.
- Full test suite before and after changes.
- Raw JSON records exact commands, data sizes, seeds, parameters, metrics,
  per-seed outcomes, revisions and limitations.

