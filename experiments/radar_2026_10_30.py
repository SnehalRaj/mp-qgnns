"""Real MP-QGNN MVPs for the 30 October 2026 research radar."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mp_qgnns.core.compound import RBSPyramidLayer, compound_matrix, rbs_pyramid_matrix
from mp_qgnns.data_io.cfi import load_cfi, prism_vs_k33
from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.cfi import EquivariantQGNN
from mp_qgnns.models.tsp import TSPQGNN
from mp_qgnns.training.cfi import evaluate
from mp_qgnns.training.tsp import balanced_bce, greedy_tour, train

torch.set_default_dtype(torch.float64)


def transfer_parameters(source, target):
    src, dst = source.state_dict(), target.state_dict()
    for name, value in src.items():
        if name in dst and dst[name].shape == value.shape:
            dst[name].copy_(value)
    target.load_state_dict(dst)


def optimal_lengths(coords, tours):
    return np.array([tour_length(coords[i].numpy(), tours[i].numpy()) for i in range(len(coords))])


def greedy_excess(model, coords, tours):
    with torch.no_grad():
        probs = torch.sigmoid(model(coords)).numpy()
    optimum = optimal_lengths(coords, tours)
    return float(np.mean([tour_length(coords[i].numpy(), greedy_tour(probs[i])) / optimum[i] - 1
                          for i in range(len(coords))]))


class AblatedTSPQGNN(TSPQGNN):
    def __init__(self, *args, use_adjacency=True, use_compound=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_adjacency, self.use_compound = use_adjacency, use_compound

    def node_embeddings(self, coords):
        x = None
        for r in range(self.num_layers):
            z = self.embed[r](coords)
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            x = z if x is None else x + z
            if self.use_adjacency:
                x = self.adjacency[r](x, coords)
            if self.use_compound:
                x = self.layers[r](x)
        return x


def train_tsp_variant(seed, variant, *, samples, epochs):
    flags = {"full": (True, True), "no_compound": (True, False),
             "no_adjacency": (False, True), "encoder_only": (False, False)}[variant]
    torch.manual_seed(10_000 + seed)
    model = AblatedTSPQGNN(5, D=6, k=3, num_layers=2,
                           use_adjacency=flags[0], use_compound=flags[1])
    coords, _, heat = random_instances(5, samples, seed=11_000 + seed)
    train(model, coords, heat, epochs=epochs, lr=3e-3)
    return model, coords, heat


def evaluate_tsp_sizes(model, seed, test_samples):
    rows = []
    for n in (5, 6, 7):
        coords, tours, _ = random_instances(n, test_samples, seed=12_000 + 100 * seed + n)
        target = model
        if n != 5:
            target = AblatedTSPQGNN(n, D=6, k=3, num_layers=2,
                                    use_adjacency=model.use_adjacency,
                                    use_compound=model.use_compound)
            transfer_parameters(model, target)
        rows.append({"seed": seed, "n_cities": n, "split": "seen" if n == 5 else "unseen",
                     "greedy_excess": greedy_excess(target, coords, tours)})
    return rows


def run_evolution_ablation(*, pilot):
    seeds = list(range(2 if pilot else 5))
    samples, epochs, test_samples = (60, 55, 32) if pilot else (120, 120, 64)
    rows, trained, start = [], {}, time.time()
    for seed in seeds:
        for variant in ("full", "no_compound", "no_adjacency", "encoder_only"):
            model, coords, heat = train_tsp_variant(seed, variant, samples=samples, epochs=epochs)
            if variant == "full":
                trained[seed] = (model, coords, heat)
            for row in evaluate_tsp_sizes(model, seed, test_samples):
                rows.append({"variant": variant, **row})

    cfi_rows = []
    for family, loader in (("prism", prism_vs_k33), ("k3", lambda: load_cfi("k3"))):
        a0, a1, n, j = loader()
        for label, zero in (("full", False), ("zero_mixing", True)):
            def build(n=n, j=j, zero=zero):
                model = EquivariantQGNN(n, j)
                if zero:
                    with torch.no_grad(): model.alphas.zero_()
                return model
            result = evaluate(build, a0, a1, n, j, seeds=tuple(seeds),
                              n_relabel=40 if pilot else 80,
                              epochs=80 if pilot else 160, inv_relabel=20)
            cfi_rows.append({"family": family, "variant": label, **result})

    def mean(variant, split):
        return float(np.mean([r["greedy_excess"] for r in rows
                              if r["variant"] == variant and r["split"] == split]))
    analysis = {"tsp": {}, "cfi": {}}
    for variant in ("no_compound", "no_adjacency", "encoder_only"):
        analysis["tsp"][variant] = {
            "seen_full_advantage": mean(variant, "seen") - mean("full", "seen"),
            "unseen_full_advantage": mean(variant, "unseen") - mean("full", "unseen")}
    for family in ("prism", "k3"):
        full = next(r for r in cfi_rows if r["family"] == family and r["variant"] == "full")
        zero = next(r for r in cfi_rows if r["family"] == family and r["variant"] == "zero_mixing")
        analysis["cfi"][family] = {"full_accuracy": full["test_acc_mean"],
                                    "zero_mixing_accuracy": zero["test_acc_mean"],
                                    "full_advantage": full["test_acc_mean"] - zero["test_acc_mean"]}
    return {"configuration": {"seeds": seeds, "train_samples": samples, "epochs": epochs,
                               "test_samples_per_size": test_samples, "test_sizes": [5, 6, 7]},
            "tsp_rows": rows, "cfi_rows": cfi_rows, "analysis": analysis,
            "runtime_seconds": time.time() - start}, trained


def fit_rbs_angles(target, *, steps=350, restarts=3):
    best = None
    n_angles = target.shape[0] * (target.shape[0] - 1) // 2
    for restart in range(restarts):
        angles = torch.nn.Parameter(torch.randn(n_angles, dtype=target.dtype) * (0.1 + restart))
        opt = torch.optim.LBFGS([angles], lr=0.8, max_iter=steps,
                                tolerance_grad=1e-13, tolerance_change=1e-15,
                                line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = (rbs_pyramid_matrix(angles, target.shape[0]) - target).square().mean()
            loss.backward()
            return loss
        opt.step(closure)
        error = float(torch.linalg.matrix_norm(
            rbs_pyramid_matrix(angles, target.shape[0]) - target).detach())
        if best is None or error < best[0]:
            best = (error, angles.detach().clone())
    return best[1], best[0]


def compile_model(model):
    compiled, rows = copy.deepcopy(model), []
    for index, layer in enumerate(model.layers):
        target = layer.orthogonal().detach()
        angles, error = fit_rbs_angles(target)
        replacement = RBSPyramidLayer(layer.D, layer.k, angles).double()
        compiled.layers[index] = replacement
        block_error = torch.linalg.matrix_norm(compound_matrix(target, layer.k) -
                                                compound_matrix(replacement.orthogonal(), layer.k))
        rows.append({"layer": index, "orthogonal_frobenius_error": error,
                     "compound_frobenius_error": float(block_error)})
    return compiled, rows


def state_fidelity_rows(source, compiled, coords):
    rows, x = [], None
    with torch.no_grad():
        for r in range(source.num_layers):
            z = source.embed[r](coords)
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            x = z if x is None else x + z
            x = source.adjacency[r](x, coords)
            a, b = source.layers[r](x), compiled.layers[r](x)
            af, bf = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
            overlap = (af * bf).sum(-1).abs()
            fidelity = overlap.square() / (af.square().sum(-1) * bf.square().sum(-1) + 1e-15)
            rows.append({"layer": r, "minimum_state_fidelity": float(fidelity.min()),
                         "mean_state_fidelity": float(fidelity.mean())})
            x = a
    return rows


def prune_compiled(compiled, coords, heat, fraction, *, finetune_epochs):
    model = copy.deepcopy(compiled)
    loss = balanced_bce(model(coords), heat); loss.backward()
    masks = []
    for layer in model.layers:
        importance = (layer.angles.detach() * layer.angles.grad.detach()).abs()
        keep = max(1, int(round((1 - fraction) * len(importance))))
        idx = torch.topk(importance, keep).indices
        mask = torch.zeros_like(importance); mask[idx] = 1
        layer.gate_mask.copy_(mask); masks.append(mask)
    model.zero_grad(set_to_none=True)
    if finetune_epochs:
        train(model, coords, heat, epochs=finetune_epochs, lr=1e-3)
    return model, masks


def run_deploy_compile(trained, *, pilot):
    rows, start = [], time.time()
    for seed, (model, coords, heat) in trained.items():
        compiled, compile_rows = compile_model(model)
        test_coords, test_tours, _ = random_instances(5, 32 if pilot else 64, seed=20_000 + seed)
        base_excess = greedy_excess(model, test_coords, test_tours)
        raw, masks = prune_compiled(compiled, coords, heat, 0.5, finetune_epochs=0)
        revised, _ = prune_compiled(compiled, coords, heat, 0.5,
                                    finetune_epochs=30 if pilot else 60)
        rows.append({"seed": seed, "baseline_excess": base_excess,
                     "compiled_excess": greedy_excess(compiled, test_coords, test_tours),
                     "raw_pruned_excess": greedy_excess(raw, test_coords, test_tours),
                     "revised_pruned_excess": greedy_excess(revised, test_coords, test_tours),
                     "kept_gates": [int(m.sum()) for m in masks],
                     "total_gates": [len(m) for m in masks], "compile_rows": compile_rows,
                     "fidelity_rows": state_fidelity_rows(model, compiled, test_coords[:8])})
    return {"configuration": {"seeds": list(trained), "prune_fraction": 0.5,
                               "revision": "fine-tune surviving gates and remaining model"},
            "rows": rows, "analysis": {
                "minimum_compile_fidelity": float(min(f["minimum_state_fidelity"] for r in rows for f in r["fidelity_rows"])),
                "compiled_added_excess": float(np.mean([r["compiled_excess"] - r["baseline_excess"] for r in rows])),
                "raw_pruned_added_excess": float(np.mean([r["raw_pruned_excess"] - r["baseline_excess"] for r in rows])),
                "revised_pruned_added_excess": float(np.mean([r["revised_pruned_excess"] - r["baseline_excess"] for r in rows]))},
            "runtime_seconds": time.time() - start}


def bounds(success, count):
    mean = success / count
    radius = 1.96 * math.sqrt(max(mean * (1 - mean), 1e-6) / count) + 1 / count
    return mean, mean - radius, mean + radius


def adaptive_choice(probs, candidates, rng, *, policy, batch=32, max_shots=2048):
    success = {c: int(rng.binomial(batch, probs[c])) for c in candidates}
    count = {c: batch for c in candidates}; active = list(candidates)
    while len(active) > 1 and max(count.values()) < max_shots:
        stats = {c: bounds(success[c], count[c]) for c in active}
        best = max(active, key=lambda c: stats[c][0]); lower = stats[best][1]
        active = [c for c in active if c == best or stats[c][2] >= lower]
        if len(active) == 1: break
        if policy == "confidence":
            selected = active
        elif policy == "entropy":
            def entropy(c):
                p = min(max(stats[c][0], 1e-12), 1 - 1e-12)
                return -p * math.log(p) - (1 - p) * math.log(1 - p)
            selected = [max(active, key=entropy)]
        else:
            best = max(active, key=lambda c: stats[c][0])
            challenger = max((c for c in active if c != best), key=lambda c: stats[c][2])
            selected = [best, challenger]
        for c in selected:
            if count[c] < max_shots:
                success[c] += int(rng.binomial(batch, probs[c])); count[c] += batch
    winner = max(active, key=lambda c: success[c] / count[c])
    return winner, sum(count.values())


def noisy_tour(probs, rng, *, policy, shots=2048):
    path, used = [0], 0
    while len(path) < len(probs):
        candidates = [c for c in range(len(probs)) if c not in path]; row = probs[path[-1]]
        if policy == "fixed":
            means = {c: rng.binomial(shots, row[c]) / shots for c in candidates}
            nxt = max(candidates, key=lambda c: means[c]); used += shots * len(candidates)
        else:
            nxt, cost = adaptive_choice(row, candidates, rng, policy=policy, max_shots=shots); used += cost
        path.append(nxt)
    return np.asarray(path), used


def run_readout(trained, *, pilot):
    repeats, test_samples = (12, 24) if pilot else (32, 48)
    rows, start = [], time.time()
    for seed, (model, _, _) in trained.items():
        coords, tours, _ = random_instances(5, test_samples, seed=30_000 + seed)
        optimum = optimal_lengths(coords, tours)
        with torch.no_grad(): probs = torch.sigmoid(model(coords)).numpy()
        for policy in ("fixed", "confidence", "entropy", "gap_retirement"):
            excess, agreement, shots = [], [], []
            for i in range(test_samples):
                exact = greedy_tour(probs[i])
                for repeat in range(repeats):
                    rng = np.random.default_rng(31_000 + seed * 1_000_000 + i * 1000 + repeat)
                    pred, cost = noisy_tour(probs[i], rng, policy=policy)
                    excess.append(tour_length(coords[i].numpy(), pred) / optimum[i] - 1)
                    agreement.append(np.array_equal(pred, exact)); shots.append(cost)
            rows.append({"seed": seed, "policy": policy, "mean_excess": float(np.mean(excess)),
                         "decoder_agreement": float(np.mean(agreement)), "mean_binary_shots": float(np.mean(shots))})
    def avg(policy, key): return float(np.mean([r[key] for r in rows if r["policy"] == policy]))
    fixed_shots, conf_shots = avg("fixed", "mean_binary_shots"), avg("confidence", "mean_binary_shots")
    analysis = {}
    for policy in ("confidence", "entropy", "gap_retirement"):
        analysis[policy] = {"reduction_vs_fixed": fixed_shots / avg(policy, "mean_binary_shots"),
                            "reduction_vs_confidence": conf_shots / avg(policy, "mean_binary_shots"),
                            "added_excess_vs_confidence": avg(policy, "mean_excess") - avg("confidence", "mean_excess"),
                            "agreement": avg(policy, "decoder_agreement")}
    return {"configuration": {"seeds": list(trained), "test_samples": test_samples,
                               "repeats": repeats, "max_shots_per_candidate": 2048, "batch": 32},
            "rows": rows, "analysis": analysis, "runtime_seconds": time.time() - start}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["pilot", "full"])
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True); pilot = args.mode == "pilot"; start = time.time()
    ablation, trained = run_evolution_ablation(pilot=pilot)
    payload = {"date": "2026-10-30", "mode": args.mode,
               "commands": [f"python experiments/radar_2026_10_30.py {args.mode} --output {args.output}"],
               "thresholds": {"evolution_go": "CFI advantage >= 0.15 and TSP advantage >= 0.02",
                   "evolution_stop": "identity within 0.02 CFI accuracy or 0.01 TSP excess after revision",
                   "deploy_go": "fidelity >= 0.999999 and 50% prune added excess <= 0.01 after fine-tune",
                   "deploy_stop": "fidelity < 0.999 or 25% prune added excess > 0.02 after revision",
                   "readout_go": ">=1.2x vs confidence, >=4x vs fixed, added excess <=0.001",
                   "readout_stop": "<1.05x vs confidence or added excess >0.005 after revision"},
               "evolution_ablation": ablation,
               "deploy_compile": run_deploy_compile(trained, pilot=pilot),
               "readout": run_readout(trained, pilot=pilot)}
    payload["runtime_seconds"] = time.time() - start
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"evolution": payload["evolution_ablation"]["analysis"],
                      "deploy": payload["deploy_compile"]["analysis"],
                      "readout": payload["readout"]["analysis"]}, indent=2))


if __name__ == "__main__": main()
