"""Real MP-QGNN MVPs for the 16 October 2026 research radar.

The three experiments use the repository's actual TSPQGNN and exact small-TSP
data generator. They test:

1. whether a diagonal empirical-Fisher participation ratio predicts task quality;
2. whether local losses retain a gradient advantage after aggregation controls;
3. whether data size predicts useful circuit depth, including a progressive-depth
   curriculum against an equal-epoch scratch baseline.

Examples
--------
    python experiments/radar_2026_10_16.py all --quick --output results/radar_2026_10_16_quick.json
    python experiments/radar_2026_10_16.py all --output results/radar_2026_10_16_full.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, tour_ratio, train


torch.set_default_dtype(torch.float64)


def _loss(model: TSPQGNN, coords: torch.Tensor, target: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(balanced_bce(model(coords), target))


def _optimal_lengths(coords: torch.Tensor, tours: torch.Tensor) -> np.ndarray:
    return np.array([
        tour_length(coords[i].numpy(), tours[i].numpy()) for i in range(len(coords))
    ])


def _parameter_count(model: TSPQGNN) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def empirical_fisher_dimension(
    model: TSPQGNN,
    coords: torch.Tensor,
    heatmaps: torch.Tensor,
    max_examples: int = 16,
) -> dict:
    """Participation ratio of per-example compound-parameter gradient energy.

    This is deliberately called an empirical-Fisher diagnostic rather than a
    QFIM: it is conditioned on the actual supervised edge loss and therefore
    probes task sensitivity, not pure-state geometry.
    """
    quantum_params = [layer.theta for layer in model.layers]
    rows = []
    model.eval()
    for i in range(min(max_examples, len(coords))):
        logits = model(coords[i : i + 1])
        loss = balanced_bce(logits, heatmaps[i : i + 1])
        grads = torch.autograd.grad(loss, quantum_params, retain_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]).detach())
    G = torch.stack(rows)
    diagonal = G.square().mean(dim=0)
    total = diagonal.sum()
    denom = diagonal.square().sum()
    deff = float(total.square() / denom) if float(denom) > 0 else 0.0
    active = int((diagonal > 1e-14 * max(float(diagonal.max()), 1e-300)).sum())
    return {
        "effective_dimension": deff,
        "effective_fraction": deff / diagonal.numel(),
        "active_directions": active,
        "quantum_parameter_count": diagonal.numel(),
        "mean_gradient_energy": float(diagonal.mean()),
        "max_gradient_energy": float(diagonal.max()),
    }


def run_grid(*, quick: bool) -> dict:
    sizes = [40, 80] if quick else [40, 80, 160]
    depths = [1, 2] if quick else [1, 2, 3]
    seeds = [0, 1] if quick else [0, 1, 2]
    epochs = 30 if quick else 100
    test_samples = 40 if quick else 80
    val_samples = 30 if quick else 60
    rows = []
    start = time.time()

    for seed in seeds:
        max_train = max(sizes)
        train_coords, _, train_heat = random_instances(5, max_train, seed=10_000 + seed)
        val_coords, _, val_heat = random_instances(5, val_samples, seed=20_000 + seed)
        test_coords, test_tours, test_heat = random_instances(5, test_samples, seed=30_000 + seed)
        opt = _optimal_lengths(test_coords, test_tours)
        for n_train in sizes:
            for depth in depths:
                # Hold initialization fixed across nested training-set sizes so
                # that a data-size trend is not an initialization trend.
                torch.manual_seed(40_000 + 100 * seed + depth)
                model = TSPQGNN(5, num_layers=depth)
                train(model, train_coords[:n_train], train_heat[:n_train], epochs=epochs, lr=3e-3)
                fisher = empirical_fisher_dimension(
                    model, train_coords[:n_train], train_heat[:n_train], max_examples=12 if quick else 20
                )
                row = {
                    "seed": seed,
                    "n_train": n_train,
                    "depth": depth,
                    "epochs": epochs,
                    "parameter_count": _parameter_count(model),
                    "train_loss": _loss(model, train_coords[:n_train], train_heat[:n_train]),
                    "validation_loss": _loss(model, val_coords, val_heat),
                    "test_loss": _loss(model, test_coords, test_heat),
                    "greedy_tour_ratio": tour_ratio(model, test_coords, opt, beam_width=1),
                    "beam5_tour_ratio": tour_ratio(model, test_coords, opt, beam_width=5),
                    **fisher,
                }
                row["generalization_gap"] = row["test_loss"] - row["train_loss"]
                row["greedy_excess"] = row["greedy_tour_ratio"] - 1.0
                rows.append(row)

    deff = np.array([r["effective_fraction"] for r in rows])
    excess = np.array([r["greedy_excess"] for r in rows])
    params = np.array([r["parameter_count"] for r in rows])
    test_loss = np.array([r["test_loss"] for r in rows])

    def corr(a, b):
        value, p = spearmanr(a, b)
        return {"rho": float(value), "pvalue": float(p)}

    per_seed = {}
    for seed in seeds:
        seed_rows = [r for r in rows if r["seed"] == seed]
        per_seed[str(seed)] = corr(
            [r["effective_fraction"] for r in seed_rows],
            [r["greedy_excess"] for r in seed_rows],
        )

    selected = []
    monotone = []
    strict_growth = []
    for seed in seeds:
        chosen = []
        for n_train in sizes:
            candidates = [r for r in rows if r["seed"] == seed and r["n_train"] == n_train]
            best = min(candidates, key=lambda r: r["validation_loss"])
            chosen.append(best["depth"])
            selected.append({"seed": seed, "n_train": n_train, "depth": best["depth"]})
        monotone.extend([chosen[i + 1] >= chosen[i] for i in range(len(chosen) - 1)])
        strict_growth.extend([chosen[i + 1] > chosen[i] for i in range(len(chosen) - 1)])

    return {
        "configuration": {
            "sizes": sizes,
            "depths": depths,
            "seeds": seeds,
            "epochs": epochs,
            "test_samples": test_samples,
            "validation_samples": val_samples,
        },
        "rows": rows,
        "analysis": {
            "effective_fraction_vs_greedy_excess": corr(deff, excess),
            "effective_fraction_vs_test_loss": corr(deff, test_loss),
            "parameter_count_vs_greedy_excess": corr(params, excess),
            "per_seed_effective_fraction_vs_greedy_excess": per_seed,
            "selected_depths": selected,
            "nondecreasing_transition_fraction": float(np.mean(monotone)) if monotone else 1.0,
            "strict_growth_transition_fraction": float(np.mean(strict_growth)) if strict_growth else 0.0,
        },
        "runtime_seconds": time.time() - start,
    }


def _upper_edge_losses(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = logits.shape[-1]
    iu = torch.triu_indices(n, n, offset=1)
    return F.binary_cross_entropy_with_logits(
        logits[:, iu[0], iu[1]], target[:, iu[0], iu[1]], reduction="none"
    )


def run_locality(*, quick: bool) -> dict:
    city_sizes = [4, 5, 6] if quick else [4, 5, 6, 7, 8]
    seeds = list(range(12 if quick else 60))
    rows = []
    start = time.time()
    for n in city_sizes:
        local_grads, mean_grads, rms_grads, sum_grads = [], [], [], []
        edge_count = n * (n - 1) // 2
        for seed in seeds:
            coords, _, heat = random_instances(n, 1, seed=50_000 + 100 * n + seed)
            torch.manual_seed(60_000 + 100 * n + seed)
            model = TSPQGNN(n, num_layers=2)
            losses = _upper_edge_losses(model(coords), heat)
            # Select an edge independently of the loss. The first diagnostic
            # used the hardest edge and therefore introduced an extreme-value
            # bias that grew with edge count.
            edge_index = np.random.default_rng(70_000 + 100 * n + seed).integers(edge_count)
            local_loss = losses[0, int(edge_index)]
            mean_loss = losses.mean()
            theta = model.layers[0].theta
            local_grads.append(float(torch.autograd.grad(local_loss, theta, retain_graph=True)[0][0]))
            mean_grad = float(torch.autograd.grad(mean_loss, theta, retain_graph=True)[0][0])
            mean_grads.append(mean_grad)
            rms_grads.append(math.sqrt(edge_count) * mean_grad)
            sum_grads.append(edge_count * mean_grad)
        rows.append({
            "n_cities": n,
            "edge_count": edge_count,
            "samples": len(seeds),
            "local_variance": float(np.var(local_grads)),
            "global_mean_variance": float(np.var(mean_grads)),
            "global_sqrt_edges_variance": float(np.var(rms_grads)),
            "global_sum_variance": float(np.var(sum_grads)),
        })

    x = np.log(np.array(city_sizes, dtype=float))
    slopes = {}
    for key in ["local_variance", "global_mean_variance", "global_sqrt_edges_variance", "global_sum_variance"]:
        y = np.log(np.maximum([r[key] for r in rows], 1e-300))
        slopes[key] = float(np.polyfit(x, y, 1)[0])
    end_ratio = rows[-1]["local_variance"] / max(rows[-1]["global_sqrt_edges_variance"], 1e-300)
    return {
        "configuration": {"city_sizes": city_sizes, "initializations_per_size": len(seeds), "depth": 2},
        "rows": rows,
        "analysis": {
            "log_log_slopes": slopes,
            "largest_size_local_to_sqrt_edge_global_ratio": end_ratio,
            "slope_advantage_after_sqrt_edge_control": (
                slopes["local_variance"] - slopes["global_sqrt_edges_variance"]
            ),
        },
        "runtime_seconds": time.time() - start,
    }


def _copy_first_layer(one: TSPQGNN, two: TSPQGNN) -> None:
    source = one.state_dict()
    target = two.state_dict()
    prefixes = ("embed.0.", "adjacency.0.", "layers.0.", "head.")
    with torch.no_grad():
        for name, value in source.items():
            if name.startswith(prefixes) and name in target and target[name].shape == value.shape:
                target[name].copy_(value)
    two.load_state_dict(target)
    # Function-preserving growth: the new re-upload emits zero, its edge angles
    # are zero, and its compound transformation is the identity. The two-layer
    # model therefore starts at exactly the trained one-layer function.
    with torch.no_grad():
        two.embed[1][-1].weight.zero_()
        two.embed[1][-1].bias.zero_()
        two.adjacency[1].angle_mlp[-1].weight.zero_()
        two.adjacency[1].angle_mlp[-1].bias.zero_()
        two.layers[1].theta.zero_()


def run_curriculum(*, quick: bool) -> dict:
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    n_train = 60 if quick else 120
    test_samples = 40 if quick else 80
    total_epochs = 40 if quick else 120
    stage_epochs = total_epochs // 2
    rows = []
    start = time.time()
    for seed in seeds:
        tr_coords, _, tr_heat = random_instances(5, n_train, seed=70_000 + seed)
        te_coords, te_tours, te_heat = random_instances(5, test_samples, seed=80_000 + seed)
        opt = _optimal_lengths(te_coords, te_tours)

        torch.manual_seed(90_000 + seed)
        scratch = TSPQGNN(5, num_layers=2)
        train(scratch, tr_coords, tr_heat, epochs=total_epochs, lr=3e-3)

        torch.manual_seed(90_000 + seed)
        shallow = TSPQGNN(5, num_layers=1)
        train(shallow, tr_coords, tr_heat, epochs=stage_epochs, lr=3e-3)
        torch.manual_seed(95_000 + seed)
        grown = TSPQGNN(5, num_layers=2)
        _copy_first_layer(shallow, grown)
        train(grown, tr_coords, tr_heat, epochs=stage_epochs, lr=3e-3)

        scratch_ratio = tour_ratio(scratch, te_coords, opt, beam_width=1)
        grown_ratio = tour_ratio(grown, te_coords, opt, beam_width=1)
        rows.append({
            "seed": seed,
            "scratch_greedy_ratio": scratch_ratio,
            "grown_greedy_ratio": grown_ratio,
            "greedy_excess_improvement": scratch_ratio - grown_ratio,
            "scratch_test_loss": _loss(scratch, te_coords, te_heat),
            "grown_test_loss": _loss(grown, te_coords, te_heat),
        })
    gains = np.array([r["greedy_excess_improvement"] for r in rows])
    return {
        "configuration": {
            "seeds": seeds,
            "n_train": n_train,
            "test_samples": test_samples,
            "total_epochs": total_epochs,
            "stage_epochs": stage_epochs,
        },
        "rows": rows,
        "analysis": {
            "mean_greedy_excess_improvement": float(gains.mean()),
            "std_greedy_excess_improvement": float(gains.std()),
            "fraction_improved": float(np.mean(gains > 0)),
        },
        "runtime_seconds": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["grid", "locality", "curriculum", "all"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "date": "2026-10-16",
        "repository_commit": "8515375",
        "quick": args.quick,
        "thresholds": {
            "effective_dimension_go": "|rho| >= 0.7, stable sign by seed, stronger than parameter count",
            "effective_dimension_stop": "|rho| < 0.4, sign instability, or no gain over parameter count",
            "locality_go": "at least 10x advantage that persists after sqrt(edge-count) variance control",
            "locality_stop": "advantage vanishes after normalization control",
            "depth_go": "at least 80% nondecreasing depth transitions and >= 0.005 curriculum gain",
            "depth_stop": "seed-dominated depth or curriculum gain < 0.002",
        },
    }
    if args.experiment in ("grid", "all"):
        payload["grid"] = run_grid(quick=args.quick)
    if args.experiment in ("locality", "all"):
        payload["locality"] = run_locality(quick=args.quick)
    if args.experiment in ("curriculum", "all"):
        payload["curriculum"] = run_curriculum(quick=args.quick)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
