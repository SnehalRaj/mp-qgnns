# Quantum ML radar MVP plan — 30 October 2026

All experiments use the real `SnehalRaj/mp-qgnns` models, data generators, compound layers, message-passing layers, and public decoders. The original test suite is run before any change and again after the final revision.

## Project 1 — Is the quantum evolution necessary?

**Source idea.** *Generative quantum extreme learning machines* finds that an identity reservoir can match entangling reservoirs because the encoding fixes the available function space.

**Proposed mechanism.** Compare the full MP-QGNN with controlled identity variants: zero Johnson mixing on CFI and no compound evolution / no adjacency message passing on TSP. Keep encoders and readouts unchanged, and retain unused parameters so the comparison isolates the forward mechanism rather than parameter count.

**Go metric.** At the certified CFI separation order, full Johnson mixing improves test accuracy by at least 0.15 over zero mixing; on TSP, the full model lowers mean greedy tour excess by at least 0.02 versus each identity ablation, with the direction reproduced in at least four of five seeds.

**Stop metric.** Zero mixing matches CFI accuracy within 0.02, or either TSP ablation matches the full model within 0.01 tour excess after the revised multi-size test.

**Paper case.** A positive result identifies the quantum evolution as a functional rather than decorative component. A reproducible negative result is still important: it dequantizes a claimed mechanism and defines the missing ablation required before stronger advantage language.

## Project 2 — Train in the reduced basis, deploy as an RBS circuit

**Source idea.** *Classical Pre-Training of Large Shallow Parametrized Quantum Circuits through Pauli Propagation and deployment on Quantum Hardware* trains a classical observable surrogate and deploys the learned circuit without an in-loop QPU.

**Proposed mechanism.** Train the real TSP MP-QGNN in its exact reduced-basis backend, compile every learned compound orthogonal matrix into the repository's RBS-pyramid gate order, verify the PennyLane weight-k block, and then prune low-sensitivity gates before recompilation.

**Go metric.** Unpruned compilation has minimum state fidelity at least 0.999999 on held-out node embeddings; pruning 50% of compiled RBS gates followed by a short fine-tune increases greedy tour excess by at most 0.01 while retaining circuit fidelity at least 0.999.

**Stop metric.** Compilation fidelity remains below 0.999 after angle-convention correction, or 25% pruning increases tour excess by more than 0.02 after the evidence-driven fine-tuning revision.

**Paper case.** This would turn the current circuit-equivalence unit tests into an end-to-end train-classical/deploy-quantum result with explicit gate compression, task loss, and simulator-to-circuit discrepancy.

## Project 3 — Sequential retirement for MP-QGNN readout

**Source idea.** *Shot Allocation Strategies in Debugging Quantum Circuits* allocates shots adaptively using entropy routing and sequential retirement rather than a fixed split.

**Proposed mechanism.** Apply sequential retirement to the real TSP MP-QGNN's candidate-edge decisions. Compare fixed shots, the 23 October confidence-stopped policy, entropy routing, and a revised top-two-gap retirement policy under identical trained probabilities and random seeds.

**Go metric.** At a 2,048-shot cap, the revised policy uses at least 1.20 times fewer shots than confidence stopping and at least 4 times fewer than fixed allocation, while adding no more than 0.001 mean tour excess and preserving at least 95% of the confidence policy's decoder agreement.

**Stop metric.** Additional saving is below 1.05 times or added tour excess exceeds 0.005 after the task-aware revision.

**Paper case.** A positive result provides a concrete measurement-efficiency contribution tied to end-task regret. The limitation is explicit: this MVP samples independent Bernoulli edge observables; a hardware paper still requires compiled observables and covariance-aware grouping.

## Planned commands

```bash
python -m pytest -q
python experiments/radar_2026_10_30.py pilot --output results/radar_2026_10_30_pilot.json
python experiments/radar_2026_10_30.py full --output results/radar_2026_10_30_full.json
python experiments/radar_2026_10_30_robustness.py --output results/radar_2026_10_30_robustness.json
python -m pytest -q
```
