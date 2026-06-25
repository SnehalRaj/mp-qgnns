"""TSP instances and tour utilities.

Small instances are generated on the fly with exact optimal tours (brute force,
feasible for n <= 9). Larger benchmark splits are loaded from pickle files with
the fields ``coordinates`` [N, n, 2], ``optimal_tours`` [N, n], and
``optimal_lengths`` [N]; a generator for those is in
experiments/generate_tsp.py.
"""
from __future__ import annotations

import pickle
from itertools import permutations
from pathlib import Path

import numpy as np
import torch


def tour_length(coords: np.ndarray, tour: np.ndarray) -> float:
    pts = coords[tour]
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).sum())


def optimal_tour(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact shortest tour by brute force (fix city 0; n <= 9)."""
    n = len(coords)
    best, best_len = None, np.inf
    for rest in permutations(range(1, n)):
        tour = np.array((0,) + rest)
        length = tour_length(coords, tour)
        if length < best_len:
            best, best_len = tour, length
    return best, best_len


def tour_heatmap(tour: np.ndarray, n: int) -> torch.Tensor:
    """Symmetric 0/1 edge matrix of the tour."""
    h = torch.zeros(n, n)
    for i in range(n):
        u, v = int(tour[i]), int(tour[(i + 1) % n])
        h[u, v] = h[v, u] = 1.0
    return h


def random_instances(n_cities: int, n_samples: int, seed: int = 0):
    """Random uniform instances with exact optimal tours."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n_samples, n_cities, 2)).astype(np.float64)
    tours, heatmaps = [], []
    for c in coords:
        tour, _ = optimal_tour(c)
        tours.append(tour)
        heatmaps.append(tour_heatmap(tour, n_cities))
    return (torch.tensor(coords), torch.tensor(np.stack(tours)),
            torch.stack(heatmaps))


def load_split(path: str | Path):
    """Load a benchmark split. Returns (coords, tours, heatmaps)."""
    data = pickle.load(open(path, "rb"))
    coords = torch.tensor(np.asarray(data["coordinates"], dtype=np.float64))
    tours = torch.tensor(np.asarray(data["optimal_tours"]))
    n = coords.shape[1]
    heatmaps = torch.stack([tour_heatmap(t.numpy(), n) for t in tours])
    return coords, tours, heatmaps
