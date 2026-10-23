"""Real MP-QGNN MVPs for the 23 October 2026 research radar.

Commands
--------
python experiments/radar_2026_10_23.py pilot --output results/radar_2026_10_23_pilot.json
python experiments/radar_2026_10_23.py full --output results/radar_2026_10_23_full.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import FourierSizeTSPQGNN, TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, greedy_tour, train


torch.set_default_dtype(torch.float64)


def optimal_lengths(coords: torch.Tensor, tours: torch.Tensor) -> np.ndarray:
    return np.array([tour_length(coords[i].numpy(), tours[i].numpy()) for i in range(len(coords))])


def transfer_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy same-shaped trainable tensors while retaining size-specific buffers."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)
    target.load_state_dict(target_state)


def instantiate_for_size(model, n_cities: int):
    cls = type(model)
    target = cls(n_cities, D=6, k=3, num_layers=2)
    transfer_parameters(model, target)
    return target


def greedy_excess(model, coords, tours) -> float:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(coords)).cpu().numpy()
    opt = optimal_lengths(coords, tours)
    values = []
    for i in range(len(coords)):
        pred = greedy_tour(probs[i])
        values.append(tour_length(coords[i].numpy(), pred) / opt[i] - 1.0)
    return float(np.mean(values))


def train_size_curriculum(cls, seed: int, *, train_samples: int, epochs: int):
    torch.manual_seed(10_000 + seed)
    model = cls(4, D=6, k=3, num_layers=2)
    for stage, n in enumerate((4, 5)):
        coords, _, heat = random_instances(n, train_samples, seed=20_000 + 100 * seed + n)
        if stage:
            next_model = cls(n, D=6, k=3, num_layers=2)
            transfer_parameters(model, next_model)
            model = next_model
        train(model, coords, heat, epochs=epochs, lr=3e-3)
    return model


def run_size_query(*, pilot: bool):
    seeds = [0, 1] if pilot else [0, 1, 2, 3]
    train_samples = 40 if pilot else 80
    epochs = 35 if pilot else 80
    test_samples = 24 if pilot else 48
    rows = []
    trained = {}
    start = time.time()
    for seed in seeds:
        for label, cls in (("baseline", TSPQGNN), ("fourier_query", FourierSizeTSPQGNN)):
            model = train_size_curriculum(cls, seed, train_samples=train_samples, epochs=epochs)
            trained[(seed, label)] = model
            for n in (4, 5, 6, 7):
                coords, tours, _ = random_instances(n, test_samples, seed=30_000 + 1000 * seed + n)
                target = instantiate_for_size(model, n)
                rows.append({
                    "seed": seed,
                    "model": label,
                    "n_cities": n,
                    "split": "seen" if n in (4, 5) else "unseen",
                    "greedy_excess": greedy_excess(target, coords, tours),
                })
    def mean(label, split):
        return float(np.mean([r["greedy_excess"] for r in rows if r["model"] == label and r["split"] == split]))
    seen_gain = mean("baseline", "seen") - mean("fourier_query", "seen")
    unseen_gain = mean("baseline", "unseen") - mean("fourier_query", "unseen")
    return ({
        "configuration": {"seeds": seeds, "train_sizes": [4, 5], "test_sizes": [4, 5, 6, 7],
                          "train_samples_per_size": train_samples, "test_samples_per_size": test_samples,
                          "epochs_per_size": epochs},
        "rows": rows,
        "analysis": {"seen_excess_improvement": seen_gain, "unseen_excess_improvement": unseen_gain},
        "runtime_seconds": time.time() - start,
    }, trained)


def margin_statistics(logits: torch.Tensor, target: torch.Tensor, tours: torch.Tensor) -> dict:
    n = logits.shape[-1]
    iu = torch.triu_indices(n, n, offset=1)
    z = logits[:, iu[0], iu[1]]
    y = target[:, iu[0], iu[1]]
    signed = torch.where(y > 0.5, z, -z)
    raw = float(signed.mean())
    pos = signed[y > 0.5]
    neg = signed[y <= 0.5]
    balanced = 0.5 * (pos.mean() + neg.mean())
    scale = z.square().mean().sqrt().clamp_min(1e-9)
    normalized_q10 = torch.quantile(signed / scale, 0.10)
    decision_margins = []
    for b in range(len(tours)):
        tour = tours[b].tolist()
        for step in range(len(tour) - 1):
            current = tour[step]
            remaining = tour[step + 1 :]
            correct = tour[step + 1]
            incorrect = [c for c in remaining if c != correct]
            if incorrect:
                decision_margins.append(logits[b, current, correct] - logits[b, current, incorrect].max())
    structured = torch.stack(decision_margins)
    structured_scale = logits.square().mean().sqrt().clamp_min(1e-9)
    structured_q10 = torch.quantile(structured / structured_scale, 0.10)
    return {"raw_mean_margin": raw, "balanced_mean_margin": float(balanced),
            "normalized_q10_margin": float(normalized_q10),
            "structured_q10_margin": float(structured_q10)}


def run_margin(*, pilot: bool):
    seeds = [0, 1] if pilot else [0, 1, 2, 3, 4]
    depths = [1, 2] if pilot else [1, 2, 3]
    train_samples = 50 if pilot else 100
    val_samples = 30 if pilot else 60
    test_samples = 24 if pilot else 48
    epochs = 45 if pilot else 90
    rows = []
    start = time.time()
    for seed in seeds:
        train_coords, _, train_heat = random_instances(5, train_samples, seed=40_000 + seed)
        val_coords, val_tours, val_heat = random_instances(5, val_samples, seed=41_000 + seed)
        test_coords, test_tours, _ = random_instances(7, test_samples, seed=42_000 + seed)
        for depth in depths:
            torch.manual_seed(43_000 + 100 * seed + depth)
            model = TSPQGNN(5, D=6, k=3, num_layers=depth)
            train(model, train_coords, train_heat, epochs=epochs, lr=3e-3)
            model.eval()
            with torch.no_grad():
                val_logits = model(val_coords)
                val_loss = float(balanced_bce(val_logits, val_heat))
            target = TSPQGNN(7, D=6, k=3, num_layers=depth)
            transfer_parameters(model, target)
            rows.append({"seed": seed, "depth": depth, "validation_loss": val_loss,
                         "unseen_greedy_excess": greedy_excess(target, test_coords, test_tours),
                         **margin_statistics(val_logits, val_heat, val_tours)})

    def corr(metric):
        rho, p = spearmanr([r[metric] for r in rows], [r["unseen_greedy_excess"] for r in rows])
        return {"rho": float(rho), "pvalue": float(p)}

    selections = []
    for seed in seeds:
        subset = [r for r in rows if r["seed"] == seed]
        bce = min(subset, key=lambda r: r["validation_loss"])
        margin = max(subset, key=lambda r: r["structured_q10_margin"])
        selections.append({"seed": seed, "bce_depth": bce["depth"], "margin_depth": margin["depth"],
                           "bce_excess": bce["unseen_greedy_excess"],
                           "margin_excess": margin["unseen_greedy_excess"],
                           "margin_selection_gain": bce["unseen_greedy_excess"] - margin["unseen_greedy_excess"]})
    return {"configuration": {"seeds": seeds, "depths": depths, "train_size": 5, "test_size": 7,
                              "train_samples": train_samples, "validation_samples": val_samples,
                              "test_samples": test_samples, "epochs": epochs},
            "rows": rows,
            "analysis": {m: corr(m) for m in ("raw_mean_margin", "balanced_mean_margin", "normalized_q10_margin", "structured_q10_margin")}
            | {"selection_rows": selections,
               "mean_margin_selection_gain": float(np.mean([r["margin_selection_gain"] for r in selections]))},
            "runtime_seconds": time.time() - start}


def _sample_edge_means(probs, candidates, shots, rng):
    return {c: float(rng.binomial(shots, probs[c]) / shots) for c in candidates}


def _adaptive_choice(probs, candidates, rng, *, batch=32, max_shots=2048, delta=0.05, method="normal"):
    successes = {c: 0 for c in candidates}
    counts = {c: 0 for c in candidates}
    active = list(candidates)
    while len(active) > 1 and max(counts[c] for c in active) < max_shots:
        for c in active:
            draw = int(rng.binomial(batch, probs[c]))
            successes[c] += draw
            counts[c] += batch
        means = {c: successes[c] / counts[c] for c in active}
        bounds = {}
        for c in active:
            n = counts[c]
            if method == "hoeffding":
                rad = math.sqrt(math.log(2 * len(candidates) / delta) / (2 * n))
            elif method == "empirical_bernstein":
                p = means[c]
                logterm = math.log(3 * len(candidates) / delta)
                rad = math.sqrt(2 * max(p * (1 - p), 1e-6) * logterm / n) + 3 * logterm / n
            else:
                p = means[c]
                rad = 1.96 * math.sqrt(max(p * (1 - p), 1e-6) / n) + 1.0 / n
            bounds[c] = (means[c] - rad, means[c] + rad)
        best = max(active, key=lambda c: means[c])
        best_lower = bounds[best][0]
        active = [c for c in active if c == best or bounds[c][1] >= best_lower]
    winner = max(active, key=lambda c: successes[c] / max(counts[c], 1))
    return winner, sum(counts.values())


def noisy_greedy(probs, rng, *, policy, shots=2048, method="normal"):
    n = probs.shape[0]
    path = [0]
    used = 0
    while len(path) < n:
        candidates = [c for c in range(n) if c not in path]
        row = probs[path[-1]]
        if policy == "fixed":
            means = _sample_edge_means(row, candidates, shots, rng)
            nxt = max(candidates, key=lambda c: means[c])
            used += shots * len(candidates)
        else:
            nxt, cost = _adaptive_choice(row, candidates, rng, max_shots=shots, method=method)
            used += cost
        path.append(nxt)
    return np.array(path), used


def run_readout(trained, *, pilot: bool):
    seeds = sorted({k[0] for k in trained})
    repeats = 8 if pilot else 24
    test_samples = 16 if pilot else 40
    rows = []
    start = time.time()
    for seed in seeds:
        model = instantiate_for_size(trained[(seed, "baseline")], 7)
        coords, tours, _ = random_instances(7, test_samples, seed=50_000 + seed)
        opt = optimal_lengths(coords, tours)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(coords)).numpy()
        for method in ("hoeffding", "normal"):
            for policy in ("fixed", "adaptive"):
                if policy == "fixed" and method == "normal":
                    continue
                excesses, agreements, costs = [], [], []
                for i in range(test_samples):
                    exact = greedy_tour(probs[i])
                    for rep in range(repeats):
                        rng = np.random.default_rng(60_000 + 100_000 * seed + 1000 * i + rep)
                        pred, cost = noisy_greedy(probs[i], rng, policy=policy, shots=2048, method=method)
                        excesses.append(tour_length(coords[i].numpy(), pred) / opt[i] - 1.0)
                        agreements.append(bool(np.array_equal(pred, exact)))
                        costs.append(cost)
                rows.append({"seed": seed, "policy": policy, "method": method,
                             "mean_greedy_excess": float(np.mean(excesses)),
                             "exact_decoder_agreement": float(np.mean(agreements)),
                             "mean_binary_shots": float(np.mean(costs))})
    fixed = [r for r in rows if r["policy"] == "fixed"]
    analyses = {}
    fixed_agreement = float(np.mean([r["exact_decoder_agreement"] for r in fixed]))
    for method in ("hoeffding", "normal"):
        adaptive = [r for r in rows if r["policy"] == "adaptive" and r["method"] == method]
        analyses[method] = {
            "shot_reduction_factor": float(np.mean([r["mean_binary_shots"] for r in fixed]) /
                                           np.mean([r["mean_binary_shots"] for r in adaptive])),
            "added_excess": float(np.mean([r["mean_greedy_excess"] for r in adaptive]) -
                                  np.mean([r["mean_greedy_excess"] for r in fixed])),
            "agreement": float(np.mean([r["exact_decoder_agreement"] for r in adaptive])),
            "fixed_policy_agreement": fixed_agreement,
        }
    return {"configuration": {"seeds": seeds, "test_size": 7, "test_samples": test_samples,
                              "measurement_repeats": repeats, "max_shots_per_edge": 2048,
                              "confidence_delta": 0.05, "batch_size": 32},
            "rows": rows, "analysis": analyses, "runtime_seconds": time.time() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "full"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = args.mode == "pilot"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_start = time.time()
    size_query, trained = run_size_query(pilot=pilot)
    payload = {
        "date": "2026-10-23",
        "mode": args.mode,
        "commands": [f"python experiments/radar_2026_10_23.py {args.mode} --output {args.output}"],
        "thresholds": {
            "size_query_go": "unseen excess improvement >= 0.05 and seen degradation <= 0.01",
            "size_query_stop": "unseen improvement < 0.01 or seen degradation > 0.02 after revision",
            "margin_go": "|rho| >= 0.60 and selection gain >= 0.03",
            "margin_stop": "|rho| <= 0.30 or no selection gain after normalization revision",
            "readout_go": "shot reduction >= 4, added excess <= 0.01, agreement >= 0.95",
            "readout_stop": "shot reduction < 2 or added excess > 0.02 after confidence revision",
        },
        "size_query": size_query,
        "margin": run_margin(pilot=pilot),
        "readout": run_readout(trained, pilot=pilot),
    }
    payload["runtime_seconds"] = time.time() - total_start
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["size_query"]["analysis"], indent=2))
    print(json.dumps(payload["margin"]["analysis"], indent=2))
    print(json.dumps(payload["readout"]["analysis"], indent=2))


if __name__ == "__main__":
    main()
