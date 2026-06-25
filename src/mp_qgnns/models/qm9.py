"""Quantum graph neural network for molecular property regression (QM9).

Each j-subset of atoms is encoded from its iso-type and its atom- and bond-type
histograms, loaded into the weight-k embedding register, and evolved by
HW-preserving compound-pyramid layers. A masked mean over the subsets that
contain only real atoms gives a permutation-invariant molecular embedding, which
a small head maps to the target property. Higher j exposes higher-order
substructure; the evolution is the same one realised as a circuit in
``core.circuits`` (RBS pyramid).
"""
from __future__ import annotations

from math import comb

import torch
import torch.nn as nn

from ..core.compound import CompoundPyramidLayer
from ..data_io.qm9 import NUM_ATOM_TYPES, NUM_BOND_TYPES, molecular_features


class QM9QGNN(nn.Module):
    def __init__(self, n: int, j: int, D: int = 6, k: int = 3, num_layers: int = 2,
                 encoder_hidden: int = 32, readout_hidden: int = 64):
        super().__init__()
        self.n, self.j = n, j
        emb = comb(D, k)
        feat_dim = 1 + j + NUM_ATOM_TYPES + NUM_BOND_TYPES
        self.encoder = nn.Sequential(nn.Linear(feat_dim, encoder_hidden), nn.ReLU(),
                                     nn.Linear(encoder_hidden, emb))
        self.layers = nn.ModuleList(CompoundPyramidLayer(D, k) for _ in range(num_layers))
        self.head = nn.Sequential(nn.Linear(emb, readout_hidden), nn.ReLU(),
                                  nn.Linear(readout_hidden, 1))

    def forward(self, adj, atom_types, bond_types, node_mask) -> torch.Tensor:
        feats, valid = molecular_features(adj, atom_types, bond_types, node_mask, self.n, self.j)
        x = self.encoder(feats)
        x = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-12)
        for layer in self.layers:
            x = layer(x)
        w = valid.unsqueeze(-1)
        pooled = (x * w).sum(dim=1) / (w.sum(dim=1) + 1e-9)
        return self.head(pooled).squeeze(-1)
