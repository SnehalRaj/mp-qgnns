# Three real MP-QGNN follow-up experiments

**Date:** 9 October 2026  
**Status:** five-seed fast loop on the public MP-QGNN implementation  
**Code:** `experiments/radar_real_mpqgnn.py`  
**Results:** `results/radar_real_full.json`

## Executive decision

The three ideas are not equally promising.

1. **Do not lead with the Fourier angle-map extension.** It runs and transfers across graph sizes, but it does not beat the original MLP consistently.
2. **Prioritize operational WL distinguishability under finite measurements.** The current noiseless CFI result is mathematically correct, but the normalized class signal is around one part in a billion. This creates a clear and important follow-up question.
3. **Develop the two-body phase extension in parallel.** One shared non-Gaussian phase changes the accessible fixed-particle algebra from a small mode algebra to the full sector algebra. The next test is whether this extra access produces a larger, measurable graph signal without destroying trainability.

## What the metrics mean

For TSP, a tour ratio of 1.00 is optimal. A ratio of 1.02 means the predicted tour is, on average, 2% longer than the true optimum. Lower is better.

For CFI, 50% classification accuracy is random guessing. The reported finite-shot curve is optimistic: it assumes that the exact best separating direction is already known, measurements are independent, and there is no hardware or training noise.

For the Lie algebra, 100% means that the generators can access every anti-Hermitian direction in the chosen fixed-particle subspace. It does not mean that a shallow circuit reaches every direction efficiently.

## Project 1: neural Fourier edge-angle map

### Proposed idea

Replace the small classical MLP that maps an edge's distance and displacement to an RBS angle with a learned Fourier map. Keep the RBS gates, compound layer, edge ordering, and quantum deployment path unchanged. This is a train-classical, deploy-quantum extension, not a classical surrogate of the whole model.

The Fourier map has 80 parameters per adjacency layer. The original MLP has 81. This makes the comparison close to parameter matched.

### Paper threshold

Train on five-city TSP and require all of the following:

- no worse five-city performance;
- at least 0.01 lower tour ratio on six cities;
- at least 0.01 lower tour ratio on seven cities;
- stability across five seeds.

A 0.01 improvement means reducing the predicted tour length by one percentage point relative to the optimum. That would be noticeable and worth investigating.

### What was run

- Actual `TSPQGNN` model and repository training loop.
- 160 five-city training instances per seed.
- 80 test instances at each of five, six, and seven cities.
- Five seeds and 100 training epochs.
- All 7,143 or 7,145 trainable parameters transferred without refitting when graph size changed.
- A first 40-epoch run failed. A frequency-scale sweep found that wider initialization improved optimization, so the final run used that initialization.

### Result

| Model | Five cities | Six cities | Seven cities |
| --- | ---: | ---: | ---: |
| Original MLP | 0.097% above optimum | 0.587% | 2.204% |
| Fourier map | 0.077% above optimum | 0.480% | 2.637% |

The Fourier map improves six-city transfer by only 0.107 percentage points and is worse at seven cities by 0.433 percentage points. It fails the paper threshold.

### Decision and score

- **Paper likelihood: 3/10.** A negative benchmark could be a section in a broader dequantization paper, but it is not a paper by itself.
- **Novelty: 5/10.** Applying neural Fourier parameterization to graph-conditioned RBS angles is new enough, but close to the IonQ Neural Fourier Surrogates direction.
- **Potential field impact: 4/10.** It directly touches perhaps 3-8 papers on Fourier dequantization, quantum baselines, and graph QNNs. Citation impact cannot be forecast reliably.

The source paper is strong: all authors are at IonQ and the paper explicitly frames NFS as a classical baseline for locating the quantum-advantage boundary. However, it does not appear to cite Hsin-Yuan (Robert) Huang in the arXiv HTML, and our current extension does not yet prove or disprove dequantization.

### Next action only if retained

Change the question from “does a Fourier angle map improve TSP?” to “can a support-matched Fourier surrogate reproduce a frozen MP-QGNN circuit?” Measure spectral overlap and inference cost, and compare with MLP, random Fourier features, and tensor-train surrogates. Stop if the surrogate is neither faithful nor cheaper.

## Project 2: operational WL expressivity under finite shots

### Proposed idea

Separate mathematical distinguishability from operational distinguishability. The present CFI experiment gives the frozen model exact float64 embeddings, whitens them, and can classify at the correct WL level. A hardware experiment must estimate bounded observables from shots. The question is whether the distinguishing direction is large enough to measure.

### Paper threshold

Call the issue material if an optimistic measurement model still needs more than \(10^{12}\) shots per bounded feature for a five-sigma single-graph distinction. A successful architecture repair should later reduce this requirement by at least four orders of magnitude while preserving equivariance and the WL threshold.

### What was run

- Actual CFI(K3) gadgets and `EquivariantQGNN` embeddings.
- \(n=18\), with \(j=1,2,3\).
- Fifty independent relabellings per class.
- Five random model seeds.
- Pooled features divided by \(\binom{n}{j}\), so every component is a bounded mean in \([-1,1]\).
- An optimistic oracle classifier with the exact class-mean direction and independent Pauli-mean noise only at inference.

The original fixed-budget TSP allocation experiment was attempted first. It was rejected because five-city beam decoding was almost always optimal, making the metric insensitive to measurement noise.

### Result

At \(j=1\), the class gap is exactly zero. At \(j=2\), it is at the float64 noise scale, consistent with the WL threshold. At \(j=3\), the five-seed bounded class-mean gap is roughly \(0.8\times10^{-9}\) to \(2.4\times10^{-9}\).

The median optimistic five-sigma requirement is approximately \(2.1\times10^{19}\) shots per bounded feature. At \(10^{16}\) shots, the mean oracle accuracy is only around 53%. It approaches useful accuracy only near \(10^{18}\) shots.

This is not yet a formal hardware resource lower bound. Observable grouping, collective measurements, amplitude estimation, a different readout, or training the quantum body could change the scale. It is nevertheless a serious operational gap in the present frozen-readout implementation.

### Decision and score

- **Paper likelihood: 8/10.** The question is direct, falsifiable, and exposes a gap between WL expressivity and measurement-realistic learnability.
- **Novelty: 8/10.** Many papers study expressivity or shot noise separately. Applying a finite-shot operational criterion to WL hierarchy separation is a sharper combination.
- **Potential field impact: 8/10.** The addressable literature is roughly 15-40 papers across QGNN expressivity, equivariant circuits, finite-shot QML, shadows, and hardware benchmarking.

The surrounding community is strong. The finite-shot linear-regression paper gives a rigorous graph-versus-shots framework. A Los Alamos paper by Chang, Larocca, and Cerezo gives an efficient classical simulation framework for permutation-equivariant circuits and cites Hsin-Yuan Huang's work on classical observable estimation and shadows. The finite-shot paper also cites Huang, Kueng, and Preskill on classical shadows. This places the idea directly inside the current advantage, dequantization, and measurement-efficiency conversation.

### Next experiment

1. Derive the exact observables and operator norms for the current CFI readout.
2. Compile Pauli groups or a shadow estimator and replace the independent-feature proxy.
3. Train the encoders and mixing angles with a normalized-margin or shot-aware loss.
4. Require at least a \(10^4\) reduction in the inferred shot requirement.
5. Compare with Johnson-GIN and the Los Alamos equivariant classical simulator.

A positive paper can say that shot-aware training repairs operational WL separation. A negative paper can prove that the current architecture's exact WL signal is exponentially or prohibitively small under a specified measurement model.

## Project 3: one two-body phase beyond fermionic linear optics

### Proposed idea

The existing compound layer is a fixed-particle representation of a mode-orthogonal transformation. Add one shared two-body occupation phase, such as \(\exp(i\phi n_0n_1)\), between compound layers. Because the same gate acts on every graph message, graph-node permutation equivariance is preserved.

### Paper threshold

The algebraic fast loop requires:

- extra hopping connectivity does not enlarge the existing algebra;
- a one-body phase remains inside the mode-unitary representation;
- one two-body phase reaches the full unitary algebra on the fixed-particle sector.

The empirical slow loop will require either a large increase in CFI measurement margin or at least a 5% reduction in QM9/TSP error at matched depth and parameter count, without more than a twofold degradation in gradient variance.

### What was run

- The actual embedding sizes used in the repository, including \(D=6,k=3\), whose reduced dimension is 20.
- Jordan-Wigner fermionic signs.
- Nearest-neighbour hopping, one extra long-range hopping, one one-body phase, and one two-body phase.
- Numerical commutator closure with double Gram-Schmidt orthogonalization.
- Additional checks at \((D,k)=(4,2)\), \((5,2)\), and the particle-hole-related sectors of \(D=5,6\).

### Result

For the actual \(D=6,k=3\) layer:

| Generator family | Lie dimension | Fraction of full \(\mathfrak u(20)\) |
| --- | ---: | ---: |
| Existing path hopping | 15 | 3.75% |
| Plus long-range hopping | 15 | 3.75% |
| Plus one-body phase | 36 | 9% |
| Plus one two-body phase | 400 | 100% |

The same pattern appears at \((4,2)\) and \((5,2)\). The two-body phase reaches the full fixed-sector algebra for all tested interior particle sectors. It does not do so in the one-particle sector, where the interaction is zero, which is the expected control.

### Decision and score

- **Paper likelihood: 7/10.** The algebra gives a clean architecture-design result, but the machine-learning benefit is not yet shown.
- **Novelty: 6/10.** Fermionic linear optics plus non-Gaussian resources is established theory. The potentially new part is the minimal shared interaction as an MP-QGNN expressivity and measurement-margin repair.
- **Potential field impact: 7/10.** The addressable literature is roughly 10-25 papers across symmetry-preserving ansatz design, fermionic QML, Lie simulation, and QGNNs.

The closest current communities include Reja's connectivity paper, the Augsburg and Quantum Motion work on Lie-algebraic simulation beyond free fermions, and Los Alamos work on equivariant simulation. This project becomes strong only if it states the known universality boundary honestly and proves a graph-learning consequence.

### Next experiment

Prove or locate the theorem that connected hopping plus one occupation interaction generates \(\mathfrak u\!\left(\binom{D}{k}\right)\) for \(2\le k\le D-2\). Then implement one complex phase per compound layer and compare it with extra RBS depth and a long-range RBS gate on tangent rank, gradient variance, CFI margin, QM9 error, and TSP transfer.

## Ranking

| Rank | Project | Paper likelihood | Novelty | Potential impact | Current verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Operational WL expressivity under finite shots | 8/10 | 8/10 | 8/10 | Main project |
| 2 | Minimal two-body phase extension | 7/10 | 6/10 | 7/10 | Parallel theory and model track |
| 3 | Fourier edge-angle extension | 3/10 | 5/10 | 4/10 | Stop as performance paper |

## Collaborators

- **Su Yeon Chang, Martin Larocca, M. Cerezo:** permutation-equivariant simulation, observable estimation, and the dequantization boundary.
- **Gabriele Lo Monaco, Salvatore Lorenzo, Luca Innocenti:** finite-shot feature geometry and resource allocation.
- **Jakob Kottmann and Adelina Bärligea:** bounded-Hamming-weight Lie algebras and practical algebra simulation.
- **Sahinur Reja:** minimal generator connectivity and variational accessibility.
- **Oliver Knitter and Martin Roetteler:** Fourier baselines and hardware-facing QML evaluation.
- **Brian Coyle and André J. Ferreira-Martins:** internal trainability, compound representation, and proof development.

Do not contact external collaborators until the exact measurement model or the complex two-body layer has one additional verified result. No email has been sent.
