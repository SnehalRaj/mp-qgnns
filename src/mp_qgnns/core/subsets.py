"""Johnson-graph combinatorics and iso-type features on j-subsets.

A graph on n vertices is lifted to the C(n, j) j-subsets of its vertex set. Two
subsets are Johnson-adjacent when they differ by a single element swap
(|S Δ S'| = 2); this is the neighbourhood that defines set-based j-WL. Each
subset carries the isomorphism type of its induced subgraph as an initial
feature.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import scipy.sparse as sp
import torch


def enumerate_subsets(n: int, j: int) -> list[tuple[int, ...]]:
    return list(combinations(range(n), j))


def johnson_adjacency(n: int, j: int) -> sp.csr_matrix:
    """Symmetric Johnson adjacency on the C(n, j) subsets (sparse, 0/1)."""
    subsets = enumerate_subsets(n, j)
    index = {s: i for i, s in enumerate(subsets)}
    rows, cols = [], []
    for i, s in enumerate(subsets):
        members = set(s)
        for out_v in s:
            for in_v in range(n):
                if in_v in members:
                    continue
                nbr = tuple(sorted((members - {out_v}) | {in_v}))
                rows.append(i)
                cols.append(index[nbr])
    m = len(subsets)
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, m))


def isotype_features(adj: np.ndarray, n: int, j: int) -> np.ndarray:
    """[C(n, j), 1 + j] iso-type matrix for one graph.

    For j = 1 the feature is the vertex degree. For j >= 2 it is the number of
    internal edges of the induced subgraph followed by its sorted internal
    degree sequence.
    """
    subsets = enumerate_subsets(n, j)
    out = np.zeros((len(subsets), 1 + j), dtype=np.float64)
    for i, s in enumerate(subsets):
        if j == 1:
            out[i, 1] = adj[s[0]].sum()
        else:
            block = adj[np.ix_(s, s)]
            out[i, 0] = block.sum() / 2.0
            out[i, 1:] = np.sort(block.sum(axis=1))
    return out


def isotype_features_batched(adj: torch.Tensor, n: int, j: int) -> torch.Tensor:
    """[B, C(n, j), 1 + j] iso-type features for a batch of adjacency matrices."""
    subsets = enumerate_subsets(n, j)
    out = torch.zeros(adj.shape[0], len(subsets), 1 + j, dtype=adj.dtype, device=adj.device)
    for i, s in enumerate(subsets):
        idx = list(s)
        if j == 1:
            out[:, i, 1] = adj[:, idx[0], :].sum(dim=1)
        else:
            block = adj[:, idx][:, :, idx]
            out[:, i, 0] = block.sum(dim=(1, 2)) / 2.0
            out[:, i, 1:] = torch.sort(block.sum(dim=2), dim=1).values
    return out
