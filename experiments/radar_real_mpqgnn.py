"""Three radar experiments executed against the real MP-QGNN implementation.

The experiments are deliberately small enough for a laptop, but use the actual
TSPQGNN, EquivariantAdjacencyLayer, CompoundPyramidLayer, CFI data, and decoding
code from this repository.

Examples
--------
python experiments/radar_real_mpqgnn.py fourier --seeds 0 1 2
python experiments/radar_real_mpqgnn.py shots --seeds 0 1 2
python experiments/radar_real_mpqgnn.py lie
python experiments/radar_real_mpqgnn.py all --output results/radar_real.json
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns import EquivariantQGNN, load_cfi, relabelled_dataset
from mp_qgnns.models.tsp import EdgeHead, TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, beam_search, train


torch.set_default_dtype(torch.float64)


class LearnedFourierAngle(nn.Module):
    """Parameter-matched Fourier replacement for the edge-angle MLP.

    With 15 frequencies this module has 80 trainable parameters, versus 81 for
    the repository's default 3 -> 16 -> 1 MLP. Its scalar output is still passed
    through tanh and used as the angle of the unchanged RBS gate sequence.
    """

    def __init__(self, in_dim: int = 3, frequencies: int = 15):
        super().__init__()
        # A wider initial spectrum was materially more stable than 0.7 in the
        # first two-seed pilot (recorded in the research note).
        self.omega = nn.Parameter(torch.randn(frequencies, in_dim) * 1.2)
        self.out = nn.Linear(2 * frequencies, 1)
        self.skip = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = x @ self.omega.T
        return self.out(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)) + self.skip(x)


def make_tsp_model(n: int, variant: str) -> TSPQGNN:
    model = TSPQGNN(n)
    if variant == "fourier":
        for layer in model.adjacency:
            layer.angle_mlp = LearnedFourierAngle()
    elif variant != "mlp":
        raise ValueError(f"unknown variant: {variant}")
    return model


@torch.no_grad()
def transfer_shared_parameters(source: nn.Module, target: nn.Module) -> int:
    """Copy same-shaped named parameters, excluding n-dependent buffers."""
    src = dict(source.named_parameters())
    copied = 0
    for name, parameter in target.named_parameters():
        if name in src and src[name].shape == parameter.shape:
            parameter.copy_(src[name])
            copied += parameter.numel()
    return copied


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimal_lengths(coords: torch.Tensor, tours: torch.Tensor) -> np.ndarray:
    return np.array([
        tour_length(coords[i].numpy(), tours[i].numpy()) for i in range(len(coords))
    ])


@torch.no_grad()
def ratio_from_embeddings(head: EdgeHead, embeddings: torch.Tensor, coords: torch.Tensor,
                          opt_lengths: np.ndarray, beam_width: int = 5) -> float:
    probs = torch.sigmoid(head(embeddings)).cpu().numpy()
    coords_np = coords.cpu().numpy()
    ratios = []
    for i in range(len(coords_np)):
        tours = beam_search(probs[i], beam_width)
        length = min(tour_length(coords_np[i], candidate) for candidate in tours)
        ratios.append(length / float(opt_lengths[i]))
    return float(np.mean(ratios))


def run_fourier(seeds: list[int], train_size: int = 160, test_size: int = 80,
                epochs: int = 100) -> dict:
    """Train at n=5 and transfer all shared parameters to n=6."""
    rows = []
    for seed in seeds:
        tr_x, _, tr_h = random_instances(5, train_size, seed=1000 + seed)
        te5_x, te5_t, _ = random_instances(5, test_size, seed=2000 + seed)
        te6_x, te6_t, _ = random_instances(6, test_size, seed=3000 + seed)
        te7_x, te7_t, _ = random_instances(7, test_size, seed=3500 + seed)
        opt5 = optimal_lengths(te5_x, te5_t)
        opt6 = optimal_lengths(te6_x, te6_t)
        opt7 = optimal_lengths(te7_x, te7_t)
        for variant in ("mlp", "fourier"):
            torch.manual_seed(seed)
            model5 = make_tsp_model(5, variant)
            train(model5, tr_x, tr_h, epochs=epochs, lr=3e-3)
            ratio5 = ratio_from_embeddings(
                model5.head, model5.node_embeddings(te5_x), te5_x, opt5
            )
            model6 = make_tsp_model(6, variant)
            copied = transfer_shared_parameters(model5, model6)
            ratio6 = ratio_from_embeddings(
                model6.head, model6.node_embeddings(te6_x), te6_x, opt6
            )
            model7 = make_tsp_model(7, variant)
            transfer_shared_parameters(model5, model7)
            ratio7 = ratio_from_embeddings(
                model7.head, model7.node_embeddings(te7_x), te7_x, opt7
            )
            rows.append({
                "seed": seed,
                "variant": variant,
                "parameters": trainable_parameters(model5),
                "copied_parameters": copied,
                "n5_tour_ratio": ratio5,
                "n6_transfer_tour_ratio": ratio6,
                "n7_transfer_tour_ratio": ratio7,
            })
    summary = {}
    for variant in ("mlp", "fourier"):
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "n5_mean": float(np.mean([r["n5_tour_ratio"] for r in selected])),
            "n5_std": float(np.std([r["n5_tour_ratio"] for r in selected])),
            "n6_mean": float(np.mean([r["n6_transfer_tour_ratio"] for r in selected])),
            "n6_std": float(np.std([r["n6_transfer_tour_ratio"] for r in selected])),
            "n7_mean": float(np.mean([r["n7_transfer_tour_ratio"] for r in selected])),
            "n7_std": float(np.std([r["n7_transfer_tour_ratio"] for r in selected])),
            "parameters": selected[0]["parameters"],
        }
    delta6 = summary["mlp"]["n6_mean"] - summary["fourier"]["n6_mean"]
    delta7 = summary["mlp"]["n7_mean"] - summary["fourier"]["n7_mean"]
    return {
        "experiment": "parameter-matched Fourier edge-angle map",
        "success_threshold": "at least 0.01 lower tour ratio at both n=6 and n=7, without worse n=5 ratio",
        "configuration": {"train_n": 5, "test_n": [5, 6, 7], "train_size": train_size,
                          "test_size": test_size, "epochs": epochs, "seeds": seeds},
        "rows": rows,
        "summary": summary,
        "n6_ratio_improvement": delta6,
        "n7_ratio_improvement": delta7,
        "passed": bool(delta6 >= 0.01 and delta7 >= 0.01
                       and summary["fourier"]["n5_mean"] <= summary["mlp"]["n5_mean"]),
    }


def sample_bounded_features(features: torch.Tensor, shots: int, rng: np.random.Generator,
                            bound: float = 2.0) -> torch.Tensor:
    """Estimate bounded real features as independent +/-1 sample means.

    TSP node embeddings after two re-uploading rounds have norm at most two.
    Dividing by two therefore produces numbers in [-1,1]. This is a measurement
    proxy, not a compiled observable schedule.
    """
    scaled = torch.clamp(features / bound, -1.0, 1.0).cpu().numpy()
    prob = 0.5 * (scaled + 1.0)
    counts = rng.binomial(shots, prob)
    estimate = bound * (2.0 * counts / shots - 1.0)
    return torch.tensor(estimate, dtype=features.dtype)


def fit_edge_head(embeddings: torch.Tensor, heatmaps: torch.Tensor, *, epochs: int,
                  seed: int) -> EdgeHead:
    torch.manual_seed(seed)
    head = EdgeHead(embeddings.shape[-1])
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = balanced_bce(head(embeddings), heatmaps)
        loss.backward()
        opt.step()
    return head


def run_shots(seeds: list[int], pool_size: int = 192, test_size: int = 80,
              body_epochs: int = 100, head_epochs: int = 120) -> dict:
    """Allocate a fixed graph-shot budget using actual TSPQGNN embeddings."""
    allocations = [(16, 512), (32, 256), (64, 128), (128, 64)]
    rows = []
    for seed in seeds:
        pool_x, _, pool_h = random_instances(5, pool_size, seed=4000 + seed)
        test_x, test_t, _ = random_instances(5, test_size, seed=5000 + seed)
        opt_test = optimal_lengths(test_x, test_t)
        torch.manual_seed(seed)
        body = TSPQGNN(5)
        train(body, pool_x, pool_h, epochs=body_epochs, lr=3e-3)
        body.eval()
        with torch.no_grad():
            pool_z = body.node_embeddings(pool_x)
            test_z = body.node_embeddings(test_x)
        for n_graphs, shots in allocations:
            rng = np.random.default_rng(10_000 * seed + n_graphs)
            noisy_z = sample_bounded_features(pool_z[:n_graphs], shots, rng)
            noisy_head = fit_edge_head(noisy_z, pool_h[:n_graphs], epochs=head_epochs,
                                       seed=20_000 + seed + n_graphs)
            ideal_head = fit_edge_head(pool_z[:n_graphs], pool_h[:n_graphs], epochs=head_epochs,
                                       seed=20_000 + seed + n_graphs)
            rows.append({
                "seed": seed,
                "training_graphs": n_graphs,
                "shots_per_feature": shots,
                "graph_shot_product": n_graphs * shots,
                "total_feature_shots": n_graphs * shots * pool_z.shape[1] * pool_z.shape[2],
                "noisy_train_tour_ratio": ratio_from_embeddings(
                    noisy_head, test_z, test_x, opt_test
                ),
                "ideal_train_tour_ratio": ratio_from_embeddings(
                    ideal_head, test_z, test_x, opt_test
                ),
            })
    summary = []
    for n_graphs, shots in allocations:
        selected = [r for r in rows if r["training_graphs"] == n_graphs]
        summary.append({
            "training_graphs": n_graphs,
            "shots_per_feature": shots,
            "noisy_mean": float(np.mean([r["noisy_train_tour_ratio"] for r in selected])),
            "noisy_std": float(np.std([r["noisy_train_tour_ratio"] for r in selected])),
            "ideal_mean": float(np.mean([r["ideal_train_tour_ratio"] for r in selected])),
            "ideal_std": float(np.std([r["ideal_train_tour_ratio"] for r in selected])),
        })
    noisy = [r["noisy_mean"] for r in summary]
    endpoint_best = min(noisy[0], noisy[-1])
    interior_best = min(noisy[1:-1])
    improvement = endpoint_best - interior_best
    return {
        "experiment": "fixed-budget graph diversity versus feature shots",
        "success_threshold": "an interior allocation lowers tour ratio by at least 0.01 versus both endpoints",
        "configuration": {"n": 5, "pool_size": pool_size, "test_size": test_size,
                          "body_epochs": body_epochs, "head_epochs": head_epochs,
                          "seeds": seeds, "embedding_bound": 2.0},
        "rows": rows,
        "summary": summary,
        "interior_improvement": improvement,
        "passed": bool(improvement >= 0.01),
        "warning": "Finite-shot embeddings are an independent bounded-observable proxy; no commuting groups or device noise are compiled.",
    }


def run_cfi_measurement_audit(seeds: list[int], relabels: int = 50) -> dict:
    """Best-case finite-shot audit of the real CFI(K3) MP-QGNN embeddings.

    The pooled embedding is divided by the number of j-subsets, making every
    component a bounded mean in [-1, 1]. We then evaluate an optimistic oracle
    classifier that knows the exact class-mean direction and suffers independent
    Pauli-mean shot noise only at inference. This omits training noise, grouping,
    state-preparation noise, and device noise, so the resulting shot scale is a
    lower-bound-style diagnostic rather than a hardware resource estimate.
    """
    a0, a1, n, threshold = load_cfi("k3")
    shot_grid = [10**p for p in (10, 12, 14, 16, 18, 20)]
    rows = []
    for seed in seeds:
        A, labels = relabelled_dataset(a0, a1, relabels, 7000 + seed)
        for j in (1, 2, 3):
            torch.manual_seed(seed)
            model = EquivariantQGNN(n, j)
            model.eval()
            with torch.no_grad():
                z = model.embed(A) / comb(n, j)
            mean0, mean1 = z[labels == 0].mean(0), z[labels == 1].mean(0)
            delta = mean1 - mean0
            gap = float(torch.linalg.vector_norm(delta))
            max_component_gap = float(delta.abs().max())
            within = float(max(z[labels == 0].std(0).max(), z[labels == 1].std(0).max()))
            if gap <= 1e-30:
                required = None
                accuracies = {str(shots): 0.5 for shots in shot_grid}
            else:
                direction = delta / gap
                midpoint = 0.5 * (mean0 + mean1)
                projection = (z - midpoint) @ direction
                variance_coeff = ((direction**2) * (1.0 - z**2).clamp_min(0.0)).sum(1)
                accuracies = {}
                for shots in shot_grid:
                    sigma = torch.sqrt(variance_coeff / float(shots)).clamp_min(1e-300)
                    signed_margin = torch.where(labels == 1, projection, -projection)
                    q = signed_margin / sigma
                    probability = 0.5 * (1.0 + torch.erf(q / math.sqrt(2.0)))
                    accuracies[str(shots)] = float(probability.mean())
                # Difference of two independently estimated class means at 5 sigma.
                required = 50.0 / (gap * gap)
            rows.append({
                "seed": seed,
                "j": j,
                "bounded_mean_gap_l2": gap,
                "largest_component_gap": max_component_gap,
                "max_relabelling_std": within,
                "optimistic_single_graph_shots_for_5sigma": required,
                "oracle_accuracy_by_shots": accuracies,
            })
    j3 = [row for row in rows if row["j"] == threshold]
    finite_requirements = [r["optimistic_single_graph_shots_for_5sigma"] for r in j3]
    median_required = float(np.median(finite_requirements))
    return {
        "experiment": "finite-shot audit of the actual CFI(K3) WL signal",
        "success_threshold": "identify an operational gap if the optimistic median 5-sigma requirement exceeds 1e12 shots per bounded feature",
        "configuration": {"family": "k3", "n": n, "wl_threshold": threshold,
                          "relabels_per_class": relabels, "seeds": seeds,
                          "shot_grid": shot_grid},
        "rows": rows,
        "median_j3_shots_for_5sigma": median_required,
        "passed": bool(median_required > 1e12),
        "warning": "Optimistic independent-Pauli and oracle-readout model; this is not a compiled measurement protocol.",
    }


def fermionic_annihilate(state: tuple[int, ...], mode: int):
    if mode not in state:
        return None
    sign = -1 if sum(v < mode for v in state) % 2 else 1
    return tuple(v for v in state if v != mode), sign


def fermionic_create(state: tuple[int, ...], mode: int):
    if mode in state:
        return None
    sign = -1 if sum(v < mode for v in state) % 2 else 1
    return tuple(sorted(state + (mode,))), sign


def hopping_generator(D: int, k: int, a: int, b: int) -> np.ndarray:
    basis = list(combinations(range(D), k))
    lookup = {state: i for i, state in enumerate(basis)}
    out = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for col, state in enumerate(basis):
        for create, annihilate, coefficient in ((a, b, 1.0), (b, a, -1.0)):
            first = fermionic_annihilate(state, annihilate)
            if first is None:
                continue
            mid, s1 = first
            second = fermionic_create(mid, create)
            if second is not None:
                final, s2 = second
                out[lookup[final], col] += coefficient * s1 * s2
    return out


def occupation_phase(D: int, k: int, modes: tuple[int, ...]) -> np.ndarray:
    basis = list(combinations(range(D), k))
    diag = [1j if all(mode in state for mode in modes) else 0j for state in basis]
    return np.diag(diag)


def lie_closure_dimension(generators: list[np.ndarray], tol: float = 1e-10) -> int:
    """Real dimension using nested commutators with the original generators."""
    shape = generators[0].shape
    q_vectors: list[np.ndarray] = []
    basis_matrices: list[np.ndarray] = []

    def add(matrix: np.ndarray) -> bool:
        vector = np.concatenate([matrix.real.ravel(), matrix.imag.ravel()])
        for _ in range(2):
            for q in q_vectors:
                vector -= np.dot(q, vector) * q
        norm = np.linalg.norm(vector)
        if norm <= tol:
            return False
        q_vectors.append(vector / norm)
        basis_matrices.append(matrix / norm)
        return True

    for generator in generators:
        add(generator)
    cursor = 0
    while cursor < len(basis_matrices):
        current = basis_matrices[cursor]
        cursor += 1
        for generator in generators:
            add(current @ generator - generator @ current)
        if len(basis_matrices) >= 2 * shape[0] * shape[1]:
            break
    return len(basis_matrices)


def run_lie() -> dict:
    rows = []
    for D, k in ((4, 2), (5, 2), (6, 3)):
        path = [hopping_generator(D, k, i, i + 1) for i in range(D - 1)]
        long_range = hopping_generator(D, k, 0, 2)
        one_body = occupation_phase(D, k, (0,))
        interaction = occupation_phase(D, k, (0, 1))
        dim = comb(D, k)
        path_dim = lie_closure_dimension(path)
        long_dim = lie_closure_dimension(path + [long_range])
        one_body_dim = lie_closure_dimension(path + [one_body])
        interaction_dim = lie_closure_dimension(path + [interaction])
        rows.append({
            "D": D,
            "k": k,
            "reduced_dimension": dim,
            "path_lie_dimension": path_dim,
            "path_plus_long_range_dimension": long_dim,
            "path_plus_one_body_phase_dimension": one_body_dim,
            "path_plus_two_body_phase_dimension": interaction_dim,
            "full_u_dimension": dim * dim,
        })
    target = rows[-1]
    return {
        "experiment": "Lie closure of the actual D=6, k=3 compound layer",
        "success_threshold": "one two-body phase reaches full u(20), while extra hopping and a one-body phase remain restricted",
        "rows": rows,
        "passed": bool(
            target["path_plus_two_body_phase_dimension"] == target["full_u_dimension"]
            and target["path_plus_long_range_dimension"] == target["path_lie_dimension"]
            and target["path_plus_one_body_phase_dimension"] < target["full_u_dimension"]
        ),
        "warning": "This is numerical evidence under Jordan-Wigner fermionic signs, not a general controllability proof.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["fourier", "shots", "lie", "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = {}
    if args.experiment in ("fourier", "all"):
        output["fourier"] = run_fourier(
            args.seeds,
            train_size=80 if args.quick else 160,
            test_size=40 if args.quick else 80,
            epochs=40 if args.quick else 100,
        )
    if args.experiment in ("shots", "all"):
        output["shots"] = run_cfi_measurement_audit(
            args.seeds, relabels=20 if args.quick else 50
        )
    if args.experiment in ("lie", "all"):
        output["lie"] = run_lie()

    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
