# MP-QGNN radar plan, 23 October 2026

All projects use the repository's real `TSPQGNN`, exact small-TSP labels, and
the standard greedy or beam decoder.  Reported tour excess is
`predicted_tour_length / optimal_tour_length - 1`, so 0.05 means a tour that is
5% longer than the exact optimum.

## Project 1: graph-size Fourier query

- **Mechanism.** Encode the known graph size with fixed sine/cosine frequencies
  and use the resulting query to FiLM-modulate the shared node embeddings. This
  is a graph analogue of a task-defined spectral query: it changes how existing
  features are combined without adding instance labels or per-edge answers.
- **Baseline.** The unmodified `TSPQGNN`, trained jointly on small graph sizes.
- **Go metric.** At least 0.05 lower mean greedy-tour excess on unseen graph
  sizes, while training-size excess changes by no more than 0.01.
- **Stop metric.** Less than 0.01 unseen-size improvement, or more than 0.02
  degradation on the training sizes, after one evidence-driven revision.
- **Paper case.** A positive result would give a concrete mechanism for
  cross-size transfer and connect MP-QGNN equivariance with Fourier inductive
  bias. A negative result would delimit what a size-only query can achieve.

## Project 2: margin-based cross-size model selection

- **Mechanism.** Compute graph-normalized edge classification margins from the
  real model logits and test whether they select models that transfer across
  graph size better than validation BCE.
- **Baseline.** Validation BCE and raw, unnormalized logit margin.
- **Go metric.** Absolute Spearman correlation at least 0.60 between the chosen
  validation margin statistic and unseen-size greedy-tour excess, and at least
  0.03 lower excess for margin-selected checkpoints than BCE-selected ones.
- **Stop metric.** Absolute correlation at most 0.30 or no seed-selection gain
  after revising the statistic to remove graph-size and class-imbalance effects.
- **Paper case.** It would supply an inexpensive generalization diagnostic for
  the central small-to-large transfer claim and a bridge to current margin
  bounds for trainable quantum encodings.

## Project 3: confidence-stopped joint readout

- **Mechanism.** Treat each row of edge probabilities as one commuting
  categorical readout, simulate finite-shot counts, and stop only when the
  best unvisited next-city probability is separated from the runner-up by a
  confidence interval. This allocates shots to ambiguous decisions instead of
  giving every graph the same budget.
- **Baseline.** A fixed number of joint shots for every decoding step.
- **Go metric.** At least 4x fewer mean shots than the fixed policy with no more
  than 0.01 additional greedy-tour excess and at least 95% agreement with the
  exact-probability decoder.
- **Stop metric.** Less than 2x shot reduction or more than 0.02 added excess
  after revising confidence calibration or the stopping batch size.
- **Paper case.** A positive result would convert low-cost readout from an
  observable-count argument into an end-to-end task guarantee at a measured
  shot budget.

## Reproducibility contract

- Run `pytest` before and after changes.
- Store every seed, sample count, epoch count, threshold, exact command, and
  raw row in JSON.
- Use only exact small-TSP optima from `mp_qgnns.data_io.tsp.random_instances`.
- Keep `main` unchanged and publish only a dedicated research branch.
