"""Robustness and falsification controls for the 23 October radar MVPs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from mp_qgnns.data_io.tsp import random_instances
from mp_qgnns.models.tsp import FourierSizeTSPQGNN, TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, train

from radar_2026_10_23 import (
    greedy_excess,
    instantiate_for_size,
    margin_statistics,
    noisy_greedy,
    optimal_lengths,
    tour_length,
    train_size_curriculum,
    transfer_parameters,
)


torch.set_default_dtype(torch.float64)


def query_alignment_control():
    seeds = list(range(6))
    rows = []
    start = time.time()
    trained_baselines = {}
    for seed in seeds:
        baseline = train_size_curriculum(TSPQGNN, seed, train_samples=100, epochs=100)
        query = train_size_curriculum(FourierSizeTSPQGNN, seed, train_samples=100, epochs=100)
        trained_baselines[seed] = baseline
        for n in (6, 7, 8):
            coords, tours, _ = random_instances(n, 64, seed=70_000 + 1000 * seed + n)
            b = instantiate_for_size(baseline, n)
            q = instantiate_for_size(query, n)
            correct = greedy_excess(q, coords, tours)
            q.query_size_override = 5
            mismatch = greedy_excess(q, coords, tours)
            rows.append({"seed": seed, "n_cities": n,
                         "baseline_excess": greedy_excess(b, coords, tours),
                         "correct_query_excess": correct,
                         "mismatched_query_excess": mismatch,
                         "alignment_gain": mismatch - correct})
    return {"configuration": {"seeds": seeds, "train_sizes": [4, 5], "test_sizes": [6, 7, 8],
                              "train_samples_per_size": 100, "epochs_per_size": 100,
                              "test_samples_per_size": 64, "mismatch_query_size": 5},
            "rows": rows,
            "analysis": {
                "query_vs_baseline_gain": float(np.mean([r["baseline_excess"] - r["correct_query_excess"] for r in rows])),
                "alignment_gain": float(np.mean([r["alignment_gain"] for r in rows])),
                "alignment_win_fraction": float(np.mean([r["alignment_gain"] > 0 for r in rows])),
            }, "runtime_seconds": time.time() - start}, trained_baselines


def _bootstrap_rho(x, y, seed=0, draws=2000):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        ix = rng.integers(0, len(x), len(x))
        rho = spearmanr(np.asarray(x)[ix], np.asarray(y)[ix]).statistic
        if np.isfinite(rho):
            vals.append(rho)
    return {"rho": float(spearmanr(x, y).statistic),
            "bootstrap_95pct": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}


def margin_target_size_control():
    seeds = list(range(6))
    depths = [1, 2, 3]
    rows = []
    start = time.time()
    for seed in seeds:
        train_coords, _, train_heat = random_instances(5, 120, seed=80_000 + seed)
        val_coords, val_tours, val_heat = random_instances(5, 72, seed=81_000 + seed)
        targets = {n: random_instances(n, 56, seed=82_000 + 1000 * seed + n) for n in (6, 7, 8)}
        for depth in depths:
            torch.manual_seed(83_000 + 100 * seed + depth)
            model = TSPQGNN(5, D=6, k=3, num_layers=depth)
            train(model, train_coords, train_heat, epochs=110, lr=3e-3)
            with torch.no_grad():
                logits = model(val_coords)
            row = {"seed": seed, "depth": depth,
                   "validation_loss": float(balanced_bce(logits, val_heat)),
                   **margin_statistics(logits, val_heat, val_tours)}
            for n, (coords, tours, _) in targets.items():
                target = TSPQGNN(n, D=6, k=3, num_layers=depth)
                transfer_parameters(model, target)
                row[f"excess_n{n}"] = greedy_excess(target, coords, tours)
            rows.append(row)
    analysis = {}
    for metric in ("raw_mean_margin", "normalized_q10_margin", "structured_q10_margin"):
        analysis[metric] = {}
        for n in (6, 7, 8):
            analysis[metric][f"n{n}"] = _bootstrap_rho(
                [r[metric] for r in rows], [r[f"excess_n{n}"] for r in rows], seed=90_000 + n)
    return {"configuration": {"seeds": seeds, "depths": depths, "train_size": 5,
                              "test_sizes": [6, 7, 8], "train_samples": 120,
                              "validation_samples": 72, "test_samples_per_size": 56,
                              "epochs": 110, "bootstrap_draws": 2000},
            "rows": rows, "analysis": analysis, "runtime_seconds": time.time() - start}


def readout_budget_curve(trained_baselines):
    budgets = [128, 512, 2048, 8192]
    repeats = 32
    rows = []
    start = time.time()
    for seed, source in trained_baselines.items():
        model = instantiate_for_size(source, 7)
        coords, tours, _ = random_instances(7, 48, seed=95_000 + seed)
        opt = optimal_lengths(coords, tours)
        with torch.no_grad():
            probs = torch.sigmoid(model(coords)).numpy()
        for budget in budgets:
            for policy in ("fixed", "adaptive"):
                excesses, agreements, costs = [], [], []
                for i in range(len(coords)):
                    exact = None
                    # Import avoided here so every path is generated by the same public decoder logic.
                    from mp_qgnns.training.tsp import greedy_tour
                    exact = greedy_tour(probs[i])
                    for rep in range(repeats):
                        rng = np.random.default_rng(96_000 + seed * 1_000_000 + budget * 10 + i * 100 + rep)
                        pred, cost = noisy_greedy(probs[i], rng, policy=policy, shots=budget, method="normal")
                        excesses.append(tour_length(coords[i].numpy(), pred) / opt[i] - 1.0)
                        agreements.append(bool(np.array_equal(pred, exact)))
                        costs.append(cost)
                rows.append({"seed": seed, "budget_per_edge": budget, "policy": policy,
                             "mean_excess": float(np.mean(excesses)),
                             "agreement": float(np.mean(agreements)),
                             "mean_binary_shots": float(np.mean(costs))})
    pareto = []
    for budget in budgets:
        fixed = [r for r in rows if r["budget_per_edge"] == budget and r["policy"] == "fixed"]
        adaptive = [r for r in rows if r["budget_per_edge"] == budget and r["policy"] == "adaptive"]
        pareto.append({"budget_per_edge": budget,
                       "shot_reduction": float(np.mean([r["mean_binary_shots"] for r in fixed]) /
                                               np.mean([r["mean_binary_shots"] for r in adaptive])),
                       "added_excess": float(np.mean([r["mean_excess"] for r in adaptive]) -
                                             np.mean([r["mean_excess"] for r in fixed])),
                       "adaptive_agreement": float(np.mean([r["agreement"] for r in adaptive])),
                       "fixed_agreement": float(np.mean([r["agreement"] for r in fixed]))})
    return {"configuration": {"seeds": list(trained_baselines), "test_size": 7,
                              "test_samples": 48, "repeats": repeats, "budgets": budgets,
                              "adaptive_method": "normal confidence, batch 32"},
            "rows": rows, "analysis": {"pareto": pareto}, "runtime_seconds": time.time() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    query, trained = query_alignment_control()
    payload = {"date": "2026-10-23", "query_alignment": query,
               "margin_target_size": margin_target_size_control(),
               "readout_budget_curve": readout_budget_curve(trained)}
    payload["runtime_seconds"] = time.time() - start
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"query": query["analysis"],
                      "margin": payload["margin_target_size"]["analysis"],
                      "readout": payload["readout_budget_curve"]["analysis"]}, indent=2))


if __name__ == "__main__":
    main()
