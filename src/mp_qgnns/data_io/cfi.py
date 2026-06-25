"""Graph pairs and relabelling utilities.

Provides the certified Cai-Furer-Immerman gadgets shipped under ``data/`` and a
small synthetic pair (triangular prism vs the complete bipartite graph K_{3,3})
that is 1-WL indistinguishable and first separated at j = 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_DATA = Path(__file__).resolve().parents[3] / "data" / "cfi_gadgets.json"


def load_cfi(name: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (adj0, adj1, n_vertices, first_separating_j) for 'k3' or 'k4'."""
    key = {"k3": "cfi_k3", "k4": "cfi_k4"}[name.lower()]
    entry = json.loads(_DATA.read_text())[key]
    a0 = np.array(entry["adj0"], dtype=np.float64)
    a1 = np.array(entry["adj1"], dtype=np.float64)
    return a0, a1, entry["n_vertices"], entry["first_separating_j"]


def prism_vs_k33() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Triangular prism vs K_{3,3}: 3-regular on 6 vertices, separate at j = 3."""
    prism = np.zeros((6, 6))
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]:
        prism[a, b] = prism[b, a] = 1.0
    k33 = np.zeros((6, 6))
    for a in (0, 1, 2):
        for b in (3, 4, 5):
            k33[a, b] = k33[b, a] = 1.0
    return prism, k33, 6, 3


def relabel(adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random vertex permutation: P A P^T."""
    p = rng.permutation(adj.shape[0])
    return adj[np.ix_(p, p)]


def relabelled_dataset(adj0, adj1, n_relabel: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``n_relabel`` random relabellings of each graph into a labelled set."""
    rng = np.random.default_rng(seed)
    adjs, labels = [], []
    for label, adj in [(0, adj0), (1, adj1)]:
        for _ in range(n_relabel):
            adjs.append(relabel(np.asarray(adj), rng))
            labels.append(label)
    A = torch.tensor(np.stack(adjs), dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    return A, y
