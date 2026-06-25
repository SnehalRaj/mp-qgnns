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


def molecular_features(adj, atom_types, bond_types, node_mask, n: int, j: int):
    """Per-subset features and a validity mask.

    Returns features [B, C(n, j), 1 + j + atom_types + bond_types] and a mask
    [B, C(n, j)] that is one where every atom of the subset is present.
    """
    subsets = list(combinations(range(n), j))
    B = adj.shape[0]
    feat_dim = 1 + j + NUM_ATOM_TYPES + NUM_BOND_TYPES
    feats = torch.zeros(B, len(subsets), feat_dim, dtype=adj.dtype)
    valid = torch.zeros(B, len(subsets), dtype=adj.dtype)
    for s_idx, s in enumerate(subsets):
        s = list(s)
        block = adj[:, s][:, :, s]
        valid[:, s_idx] = node_mask[:, s].prod(dim=1)
        feats[:, s_idx, 0] = block.sum(dim=(1, 2)) / 2.0
        if j >= 2:
            feats[:, s_idx, 1:1 + j] = torch.sort(block.sum(dim=2), dim=1).values
        atoms = atom_types[:, s]
        feats[:, s_idx, 1 + j:1 + j + NUM_ATOM_TYPES] = \
            F.one_hot(atoms, NUM_ATOM_TYPES).sum(dim=1).to(adj.dtype)
        bonds = bond_types[:, s][:, :, s]
        bh = F.one_hot(bonds.long(), NUM_BOND_TYPES + 1).sum(dim=(1, 2)).to(adj.dtype)
        feats[:, s_idx, 1 + j + NUM_ATOM_TYPES:] = bh[:, 1:] / 2.0
    return feats, valid


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
