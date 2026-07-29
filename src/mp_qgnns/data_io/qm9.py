"""QM9 molecular graphs and per-subset features.

Molecules are padded to a fixed atom count and described by an adjacency matrix,
per-atom types, per-bond types, and a node mask. Each j-subset of atoms carries
the iso-type of its induced subgraph together with histograms of the atom and
bond types it contains. The full QM9 set is loaded through PyTorch Geometric
(see experiments/run_qm9.py); a synthetic generator is provided so the package
can be exercised without the download.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

NUM_ATOM_TYPES = 5   # H, C, N, O, F
NUM_BOND_TYPES = 4   # single, double, triple, aromatic


_SUBSET_IDX_CACHE: dict = {}


def _subset_indices(n: int, j: int, device) -> torch.Tensor:
    key = (n, j, str(device))
    if key not in _SUBSET_IDX_CACHE:
        _SUBSET_IDX_CACHE[key] = torch.tensor(
            list(combinations(range(n), j)), dtype=torch.long, device=device
        )
    return _SUBSET_IDX_CACHE[key]


def molecular_features(adj, atom_types, bond_types, node_mask, n: int, j: int):
    """Per-subset iso-type features and a validity mask.

    Each of the C(n, j) subsets carries ``[slot0, sorted internal degrees (j),
    atom-type histogram, bond-type histogram]``. slot0 is the full-graph degree at
    j = 1 and the internal edge count at j >= 2; a single atom induces no edges, so
    an edge count at j = 1 would leave every structural slot zero.

    ``valid`` is one where all atoms of the subset are real and, at j >= 2, the
    induced subgraph has an internal edge. Invalid rows are zeroed.
    """
    subset_idx = _subset_indices(n, j, adj.device)
    m = subset_idx.shape[0]
    row_idx = subset_idx.unsqueeze(-1).expand(m, j, j)
    col_idx = subset_idx.unsqueeze(-2).expand(m, j, j)

    sub_adj = adj[:, row_idx, col_idx]                          # [B, m, j, j]
    int_edges = sub_adj.sum(dim=(-2, -1)) / 2.0                 # [B, m]
    if j == 1:
        slot0 = adj.sum(dim=-1)                                 # full-graph degree [B, n]
        sorted_degs = slot0.unsqueeze(-1)                       # [B, m, 1]
    else:
        slot0 = int_edges
        sorted_degs, _ = sub_adj.sum(dim=-1).sort(dim=-1)       # [B, m, j]

    atom_hist = (
        F.one_hot(atom_types[:, subset_idx], num_classes=NUM_ATOM_TYPES)
        .sum(dim=2)
        .to(adj.dtype)
    )
    bond_hist = (
        F.one_hot(bond_types[:, row_idx, col_idx].long(), num_classes=NUM_BOND_TYPES + 1)[..., 1:]
        .sum(dim=(2, 3))
        .to(adj.dtype)
        / 2.0
    )

    feats = torch.cat(
        [slot0.unsqueeze(-1), sorted_degs.to(adj.dtype), atom_hist, bond_hist], dim=-1
    )
    valid = node_mask[:, subset_idx].bool().all(dim=-1)
    if j >= 2:
        valid = valid & (int_edges > 0)
    return feats * valid.unsqueeze(-1).to(feats.dtype), valid


def synthetic_molecules(n_max: int, n_samples: int, seed: int = 0):
    """Random connected molecular graphs with a smooth synthetic target."""
    rng = np.random.default_rng(seed)
    adj = torch.zeros(n_samples, n_max, n_max)
    atom_types = torch.zeros(n_samples, n_max, dtype=torch.long)
    bond_types = torch.zeros(n_samples, n_max, n_max, dtype=torch.long)
    node_mask = torch.zeros(n_samples, n_max)
    targets = torch.zeros(n_samples)
    for i in range(n_samples):
        n = int(rng.integers(max(3, n_max - 4), n_max + 1))
        node_mask[i, :n] = 1.0
        atom_types[i, :n] = torch.tensor(rng.integers(0, NUM_ATOM_TYPES, n))
        order = rng.permutation(n)
        for a, b in zip(order[:-1], order[1:]):
            bt = int(rng.integers(1, NUM_BOND_TYPES + 1))
            adj[i, a, b] = adj[i, b, a] = 1.0
            bond_types[i, a, b] = bond_types[i, b, a] = bt
        targets[i] = (atom_types[i, :n].float().mean() + adj[i].sum() / n)
    return adj, atom_types, bond_types, node_mask, targets


def load_qm9(root: str, target_index: int = 4, n_max: int = 20, n_molecules: int | None = None):
    """Load QM9 through PyTorch Geometric and collate to padded tensors.

    target_index 4 is the HOMO-LUMO gap in eV. Requires ``torch_geometric``.
    """
    from torch_geometric.datasets import QM9

    dataset = QM9(root=root)
    items = []
    for data in dataset:
        n = data.x.shape[0]
        if n > n_max:
            continue
        items.append(data)
        if n_molecules is not None and len(items) >= n_molecules:
            break
    B = len(items)
    adj = torch.zeros(B, n_max, n_max)
    atom_types = torch.zeros(B, n_max, dtype=torch.long)
    bond_types = torch.zeros(B, n_max, n_max, dtype=torch.long)
    node_mask = torch.zeros(B, n_max)
    targets = torch.zeros(B)
    for i, data in enumerate(items):
        n = data.x.shape[0]
        node_mask[i, :n] = 1.0
        atom_types[i, :n] = data.x[:n, :NUM_ATOM_TYPES].argmax(dim=-1)
        src, dst = data.edge_index
        bt = data.edge_attr.argmax(dim=-1) + 1
        adj[i, src, dst] = 1.0
        bond_types[i, src, dst] = bt.to(bond_types.dtype)
        targets[i] = data.y[0, target_index]
    return adj, atom_types, bond_types, node_mask, targets
