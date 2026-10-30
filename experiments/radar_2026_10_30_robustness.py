"""Evidence-driven revisions and robustness controls for the 30 October radar."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from mp_qgnns.core.circuits import hw_block, pyramid_unitary
from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.training.tsp import greedy_tour

from radar_2026_10_30 import (
    AblatedTSPQGNN,
    adaptive_choice,
    compile_model,
    greedy_excess,
    optimal_lengths,
    prune_compiled,
    train_tsp_variant,
    transfer_parameters,
)

torch.set_default_dtype(torch.float64)


def retrain_full_models():
    trained = {}
    for seed in range(5):
        trained[seed] = train_tsp_variant(seed, "full", samples=120, epochs=120)
    return trained


def counterfactual_gate_neutralization(trained):
    rows = []
    for seed, (model, _, _) in trained.items():
        for n in (5, 6, 7):
            coords, tours, _ = random_instances(n, 64, seed=40_000 + 100 * seed + n)
            for variant, flags in (("full", (True, True)), ("neutralize_compound", (True, False)),
                                   ("neutralize_adjacency", (False, True)),
                                   ("neutralize_both", (False, False))):
                target = AblatedTSPQGNN(n, D=6, k=3, num_layers=2,
                                        use_adjacency=flags[0], use_compound=flags[1])
                transfer_parameters(model, target)
                rows.append({"seed": seed, "n_cities": n, "variant": variant,
                             "greedy_excess": greedy_excess(target, coords, tours)})
    analysis = {}
    for variant in ("neutralize_compound", "neutralize_adjacency", "neutralize_both"):
        analysis[variant] = {}
        for n in (5, 6, 7):
            full = np.mean([r["greedy_excess"] for r in rows if r["variant"] == "full" and r["n_cities"] == n])
            ablated = np.mean([r["greedy_excess"] for r in rows if r["variant"] == variant and r["n_cities"] == n])
            analysis[variant][f"n{n}_added_excess"] = float(ablated - full)
    return {"configuration": {"seeds": list(trained), "test_sizes": [5, 6, 7],
                               "samples_per_size": 64,
                               "control": "same trained weights; gates neutralized only at inference"},
            "rows": rows, "analysis": analysis}


def pruning_curve(trained):
    rows = []
    for seed in range(3):
        model, coords, heat = trained[seed]
        compiled, compile_rows = compile_model(model)
        test_coords, test_tours, _ = random_instances(5, 64, seed=50_000 + seed)
        base = greedy_excess(model, test_coords, test_tours)
        # Verify the learned compiler against the actual PennyLane circuit block.
        circuit_errors = []
        for layer in compiled.layers:
            full = pyramid_unitary(layer.angles.detach().numpy(), layer.D)
            circuit_block = torch.tensor(hw_block(full, layer.D, layer.k).real)
            from mp_qgnns.core.compound import compound_matrix
            expected = compound_matrix(layer.orthogonal(), layer.k).detach()
            circuit_errors.append(float(torch.linalg.matrix_norm(circuit_block - expected)))
        for fraction in (0.25, 0.5, 0.75):
            raw, masks = prune_compiled(compiled, coords, heat, fraction, finetune_epochs=0)
            revised, _ = prune_compiled(compiled, coords, heat, fraction, finetune_epochs=60)
            rows.append({"seed": seed, "prune_fraction": fraction, "baseline_excess": base,
                         "raw_added_excess": greedy_excess(raw, test_coords, test_tours) - base,
                         "revised_added_excess": greedy_excess(revised, test_coords, test_tours) - base,
                         "kept_gates": [int(m.sum()) for m in masks],
                         "compile_rows": compile_rows,
                         "pennylane_block_errors": circuit_errors})
    analysis = {}
    for fraction in (0.25, 0.5, 0.75):
        selected = [r for r in rows if r["prune_fraction"] == fraction]
        analysis[str(fraction)] = {
            "raw_added_excess": float(np.mean([r["raw_added_excess"] for r in selected])),
            "revised_added_excess": float(np.mean([r["revised_added_excess"] for r in selected]))}
    analysis["maximum_pennylane_block_error"] = float(max(e for r in rows for e in r["pennylane_block_errors"]))
    return {"configuration": {"seeds": [0, 1, 2], "fractions": [0.25, 0.5, 0.75],
                               "finetune_epochs": 60}, "rows": rows, "analysis": analysis}


def revised_choice(probs, candidates, rng, *, batch=32, max_shots=2048):
    success = {c: int(rng.binomial(batch, probs[c])) for c in candidates}
    count = {c: batch for c in candidates}; active = list(candidates)
    from radar_2026_10_30 import bounds
    while len(active) > 1 and max(count.values()) < max_shots:
        stats = {c: bounds(success[c], count[c]) for c in active}
        best = max(active, key=lambda c: stats[c][0]); lower = stats[best][1]
        active = [c for c in active if c == best or stats[c][2] >= lower]
        if len(active) == 1: break
        best = max(active, key=lambda c: stats[c][0])
        challenger = max((c for c in active if c != best), key=lambda c: stats[c][2])
        # Revision: sample only the less certain member of the top-two comparison.
        chosen = max((best, challenger), key=lambda c: stats[c][2] - stats[c][1])
        if count[chosen] < max_shots:
            success[chosen] += int(rng.binomial(batch, probs[chosen])); count[chosen] += batch
    winner = max(active, key=lambda c: success[c] / count[c])
    return winner, sum(count.values())


def noisy_tour(probs, rng, policy, shots=2048):
    path, used = [0], 0
    while len(path) < len(probs):
        candidates = [c for c in range(len(probs)) if c not in path]; row = probs[path[-1]]
        if policy == "fixed":
            means = {c: rng.binomial(shots, row[c]) / shots for c in candidates}
            nxt = max(candidates, key=lambda c: means[c]); cost = shots * len(candidates)
        elif policy == "gap_single":
            nxt, cost = revised_choice(row, candidates, rng, max_shots=shots)
        else:
            nxt, cost = adaptive_choice(row, candidates, rng, policy=policy, max_shots=shots)
        path.append(nxt); used += cost
    return np.asarray(path), used


def larger_graph_readout(trained):
    rows = []
    for seed, (source, _, _) in trained.items():
        model = AblatedTSPQGNN(7, D=6, k=3, num_layers=2)
        transfer_parameters(source, model)
        coords, tours, _ = random_instances(7, 40, seed=60_000 + seed)
        optimum = optimal_lengths(coords, tours)
        with torch.no_grad(): probs = torch.sigmoid(model(coords)).numpy()
        for policy in ("fixed", "confidence", "gap_retirement", "gap_single"):
            excess, agreement, costs = [], [], []
            for i in range(len(coords)):
                exact = greedy_tour(probs[i])
                for repeat in range(24):
                    rng = np.random.default_rng(61_000 + seed * 1_000_000 + i * 1000 + repeat)
                    pred, cost = noisy_tour(probs[i], rng, policy)
                    excess.append(tour_length(coords[i].numpy(), pred) / optimum[i] - 1)
                    agreement.append(np.array_equal(pred, exact)); costs.append(cost)
            rows.append({"seed": seed, "policy": policy, "mean_excess": float(np.mean(excess)),
                         "agreement": float(np.mean(agreement)), "mean_binary_shots": float(np.mean(costs))})
    def avg(policy, key): return float(np.mean([r[key] for r in rows if r["policy"] == policy]))
    analysis = {}
    for policy in ("confidence", "gap_retirement", "gap_single"):
        analysis[policy] = {"reduction_vs_fixed": avg("fixed", "mean_binary_shots") / avg(policy, "mean_binary_shots"),
                            "reduction_vs_confidence": avg("confidence", "mean_binary_shots") / avg(policy, "mean_binary_shots"),
                            "added_excess_vs_confidence": avg(policy, "mean_excess") - avg("confidence", "mean_excess"),
                            "agreement": avg(policy, "agreement")}
    return {"configuration": {"seeds": list(trained), "n_cities": 7, "samples": 40,
                               "repeats": 24, "max_shots": 2048}, "rows": rows, "analysis": analysis}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True); start = time.time()
    trained = retrain_full_models()
    payload = {"date": "2026-10-30",
               "commands": [f"python experiments/radar_2026_10_30_robustness.py --output {args.output}"],
               "counterfactual_neutralization": counterfactual_gate_neutralization(trained),
               "pruning_curve": pruning_curve(trained),
               "larger_graph_readout": larger_graph_readout(trained)}
    payload["runtime_seconds"] = time.time() - start
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"neutralization": payload["counterfactual_neutralization"]["analysis"],
                      "pruning": payload["pruning_curve"]["analysis"],
                      "readout": payload["larger_graph_readout"]["analysis"]}, indent=2))


if __name__ == "__main__": main()
