"""Exact integer set-based j-WL refinement.

This is the combinatorial ground truth, independent of any learned model. Each
graph is lifted to its j-subsets, coloured by the iso-type of the induced
subgraph, and refined over the Johnson (one-swap) neighbourhood until the colour
partition stabilises. Two graphs are distinguished at level j when their stable
colour histograms differ. All colours are integers in a shared space, so the
comparison is exact.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .subsets import enumerate_subsets, isotype_features, johnson_adjacency


def _canonicalise(signatures: list[list]) -> list[list[int]]:
    """Map per-graph signature lists to dense integers in one shared space."""
    order = {sig: idx for idx, sig in enumerate(sorted({s for g in signatures for s in g}))}
    return [[order[s] for s in g] for g in signatures]


def set_j_wl_histograms(adjs, n: int, j: int, max_rounds: int | None = None) -> list[Counter]:
    """Stable set-j-WL colour histogram for each adjacency matrix in ``adjs``."""
    subsets = enumerate_subsets(n, j)
    m = len(subsets)
    csr = johnson_adjacency(n, j)
    nbrs = [csr.indices[csr.indptr[i]:csr.indptr[i + 1]].tolist() for i in range(m)]

    init = [[tuple(row) for row in isotype_features(np.asarray(a), n, j)] for a in adjs]
    colours = _canonicalise(init)
    n_colours = len({c for g in colours for c in g})

    for _ in range(max_rounds if max_rounds is not None else m):
        refined = [
            [(colours[g][i], tuple(sorted(colours[g][k] for k in nbrs[i]))) for i in range(m)]
            for g in range(len(adjs))
        ]
        colours = _canonicalise(refined)
        new_n = len({c for g in colours for c in g})
        if new_n == n_colours:
            break
        n_colours = new_n

    return [Counter(g) for g in colours]


def distinguishes(adj0, adj1, n: int, j: int) -> bool:
    """True if set-j-WL tells the two graphs apart."""
    h0, h1 = set_j_wl_histograms([adj0, adj1], n, j)
    return h0 != h1


def first_separating_j(adj0, adj1, n: int, max_j: int) -> int | None:
    """Smallest j in 1..max_j at which set-j-WL separates the two graphs."""
    for j in range(1, max_j + 1):
        if distinguishes(adj0, adj1, n, j):
            return j
    return None
