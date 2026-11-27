# MP-QGNN radar MVP plan — 27 November 2026

All three studies use the repository's real `TSPQGNN`, exact reduced-basis
forward path, exact small-TSP labels, and greedy tour decoding.  A **tour gap**
of 0% is optimal; 5% means that the decoded route is 5% longer than the exact
optimum.  Decisions are made from paired random seeds, not a best run.

## A. Free-fermion / Slater dequantization boundary

**Mechanism.**  The new FermiML paper makes a single Slater determinant the
natural efficiently simulable fermionic control.  MP-QGNN's RBS evolution is a
compound of a one-particle orthogonal matrix, but its learned data re-upload can
produce arbitrary states in the fixed-particle sector.  Measure the best-Slater
fidelity of trained MP-QGNN embeddings, project them to that manifold, and
separate representation loss from readout mismatch by refitting only the edge
head.

**Go metric.**  Median best-Slater infidelity is at least 0.10, and the
Slater-projected representation is at least 1.0 tour-gap point worse than the
full model after an equal-budget head refit in at least 4/5 seeds.

**Stop metric.**  Median infidelity is below 0.01, or the refitted surrogate is
within 0.5 tour-gap point in at least 4/5 seeds.

**Paper case.**  A positive result identifies the non-Gaussian loader/re-upload,
not the RBS dynamics, as the empirically useful resource; a negative result is
an explicit dequantization of the learned MP-QGNN representation.

**Planned revision.**  If direct projection gives a saturated failure, refit the
classical edge head on projected training features before judging information
loss.

## B. Training-only MP-QGNN prior, classical deployment

**Mechanism.**  Adapt QPAT's training-only quantum prior to graphs: train a real
MP-QGNN teacher on TSP-5, use its soft edge logits to regularize a size-free
classical edge network, then remove the MP-QGNN entirely at TSP-6 inference.

**Go metric.**  The classical-only deployed student improves the baseline
TSP-6 median tour gap by at least 10% relatively, wins at least 4/5 paired
seeds, and beats an untrained-MP-QGNN teacher control by at least 0.5 point.

**Stop metric.**  No positive relative gain, fewer than 3/5 paired wins, or the
random-teacher control is within 0.25 point of the trained-teacher result.

**Paper case.**  This would be a graph-structured quantum-assisted-training,
classical-deployment result with zero inference quantum calls and a cross-size
test rather than another in-distribution accuracy claim.

**Planned revision.**  Start with stochastic additive prior logits.  If this
couples or destabilizes the student, keep supervised BCE on the classical
student and use temperature-scaled soft-logit distillation as a separate loss.

## C. Symmetry-compatible response to the contractive-layer proposal

**Mechanism.**  The paper's literal layer,
`exp(i alpha Z_1 ... Z_D)`, is only a global phase in a fixed-Hamming-weight
sector.  Verify this no-go exactly, then test the smallest symmetry-compatible
replacement: a one-parameter dense number-conserving orbital rotation whose
`k`-particle action is its compound matrix.

**Go metric.**  First verify literal-sector invariance below 1e-12.  The revised
one-parameter collective mixer must improve median TSP-6 gap by at least 1.0
point or 10% relatively and win at least 4/5 seeds, with no more than two added
parameters.

**Stop metric.**  The collective layer wins fewer than 3/5 seeds or changes the
median by less than 0.25 point.

**Paper case.**  The exact no-go is a useful architecture warning for
number-conserving QML; a successful replacement would turn it into a constructive
symmetry-aware layer design for MP-QGNNs.

**Planned revision.**  Compare a randomly initialized collective angle with an
identity initialization; retain the latter only if the first version disturbs
the pretrained fixed-particle geometry.

