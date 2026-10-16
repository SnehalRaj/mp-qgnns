# MP-QGNN radar results, 16 October 2026

## Executive decision

All three MVPs stop as standalone paper stories. They are useful negative controls for the broader operational-expressivity project.

| MVP | Paper likelihood | Novelty | Potential impact | Decision |
|---|---:|---:|---:|---|
| Task-conditioned Fisher dimension | 4/10 | 5/10 | 6/10 | **STOP** as a selector; retain as geometry diagnostic |
| Local-loss trainability | 5/10 | 6/10 | 6/10 | **STOP**; no scaling advantage after normalization control |
| Data-size-driven depth growth | 3/10 | 4/10 | 5/10 | **STOP**; selection is mostly seed-driven and growth hurts |

The scores estimate the chance that the tested idea, in its current form, supports a defensible standalone paper, its difference from nearby work, and the size of the adjacent literature that would need to respond. They are judgments, not statistical outputs.

## What actually ran

- Clean baseline: 43 repository tests passed before changes.
- Grid: 27 real `TSPQGNN` models, from 3 data sizes x 3 depths x 3 seeds, each trained for 100 full-batch epochs on exact five-city TSP targets.
- Locality audit: 60 independent model initializations for each city count from 4 through 8, for 300 real model probes.
- Curriculum: 5 seeds, 120 training graphs, 80 test graphs, and 120 total epochs per method.
- Main task metric: greedy decoded tour length divided by the exact optimum. A ratio of 1 is optimal; a ratio of 1.02 means the predicted tour is 2% longer than optimal. Beam-5 results are retained but not used as the headline because they nearly saturate at five cities.
- Raw machine-readable output: `results/radar_2026_10_16_full.json`.

## MVP 1: task-conditioned Fisher dimension

The source paper correctly distinguishes circuit geometry from model quality. Our graph-specific test used the participation ratio of per-example compound-parameter gradient energy, conditioned on the real supervised edge loss.

- Effective fraction versus greedy tour excess: Spearman rho = 0.402, p = 0.038.
- Per-seed correlations: 0.367, 0.500, and 0.367.
- Parameter count versus greedy excess: rho = -0.623, stronger than the new scalar.

**Decision: STOP.** The result misses the predefined |rho| >= 0.7 threshold and does not beat parameter count. It should not be used to claim trainability, expected accuracy, or useful depth. It remains a descriptive diagnostic of how gradient energy is distributed across quantum parameters.

## MVP 2: local versus global loss

The first quick run selected the hardest edge, which introduced an extreme-value bias that grew with graph size. The final run selected one edge independently of the loss and compared it with the all-edge mean after multiplying the global gradient by the square root of the edge count. This is the variance correction expected for averaging independent contributions.

- At 8 cities, local variance / controlled-global variance = 0.393, not greater than 10.
- Log-log slope versus city count: local 0.392; controlled global 3.126.
- Neither curve decays over the tested range, so there is no barren-plateau signature here.

**Decision: STOP.** Local readout is not more trainable under this mechanism-matched MP-QGNN control. The uncorrected global mean looks smaller partly because averaging changes scale. Q-Capsule's broader claims may still hold for its architecture, but they do not transfer automatically to this message-passing model.

## MVP 3: data size and progressive depth

Initializations were held fixed across nested data sizes. Validation selected depths `[2,2,2]`, `[2,3,3]`, and `[3,3,3]` across the three seeds. All transitions are nondecreasing, but only 1 of 6 transitions actually grows, so the apparent monotonicity is mostly constant, seed-specific selection.

The first growth implementation perturbed the shallow model with a random new layer and lost 1.59 percentage points of greedy tour quality on average. We then made growth exactly function-preserving by zeroing the new re-upload and edge angles and setting the new compound transform to the identity. That fairer revision lost 2.20 percentage points on average and was worse in all five seeds.

**Decision: STOP.** Training-set size is not a useful depth rule in this range, and naive 1-to-2-layer growth is worse than training the two-layer model from scratch under the same 120-epoch budget.

## Papers that motivated the MVPs

1. Mia, Khandakar & Ayari, [Can the Quantum Fisher Effective Dimension Judge a Hybrid Quantum Neural Network?](https://arxiv.org/abs/2610.14070), 7 October 2026.
2. Fime et al., [Q-Capsule: A Localized Capsule-Based Quantum Neural Architecture for Barren Plateau Mitigation](https://arxiv.org/abs/2610.11261), 8 October 2026.
3. Nowmi, Kamol & Rahman, [Toward Joint Optimization of Circuit Depth and Training Data Size in Adaptively Grown Quantum Classifiers](https://arxiv.org/abs/2610.12428), 8 October 2026.
4. Svetlichnyy, [Optimal Lower Bound on the Sample Complexity of Coherent Classical Shadows for Mixed States](https://arxiv.org/abs/2610.18031), 13 October 2026.
5. Ding, Huang & Xing, [Federated Unlearning of Quantum Data via Classical Shadows](https://arxiv.org/abs/2610.19077), 13 October 2026.

No stable official code repository was found from the arXiv records for these five first-version preprints.

## Best next paper direction

Return to operational WL expressivity. Compile an explicit bounded-observable measurement contract for the CFI readout, then determine whether the tiny exact separation can be amplified by a trainable observable without increasing backward-light-cone size. This connects the current MP-QGNN finite-shot gap to both the new coherent-shadow lower bound and SHADE's light-cone-dependent shadow complexity.

## Collaboration targets, only after one positive result

- **Pavel Svetlichnyy:** ask which rank/Frobenius-norm regime applies to a structured CFI observable family and whether the coherent lower bound can be specialized to permutation-equivariant readouts.
- **Zihao Ding, Jun Huang, Cong-cong Xing:** ask whether the SHADE backward-light-cone analysis can bound reusable gradient and curvature estimates for shared MP-QGNN layers.
- **Md Masum Mia, Amith Khandakar, Mohamed Arselene Ayari:** share the controlled graph-specific failure of effective dimension as a selector and ask about a joint diagnostic protocol.

Do not email yet. First produce the physically explicit CFI observable family or a positive amplification result, then attach a two-page note with one precise technical question.
