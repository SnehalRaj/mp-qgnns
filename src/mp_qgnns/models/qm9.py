"""Quantum graph neural network for molecular property regression (QM9).

The architecture of ``models.cfi.EquivariantQGNN`` with molecular features and a
regression head: each round mixes the subset axis by U_r = exp(i*alpha_r*A_J),
re-uploads the iso-type features, masks and renormalises. The mixing is the
message-passing step on the Johnson lift and has the circuit form in
``core.circuits.mixing_unitary`` (tests/test_qm9.py). Run in float64.

The per-round renormalisation and the validity mask break exact unitarity, as in
the CFI model.
"""
from __future__ import annotations

from math import comb

import numpy as np
import torch
import torch.nn as nn

from ..core.subsets import johnson_adjacency
from ..data_io.qm9 import NUM_ATOM_TYPES, NUM_BOND_TYPES, molecular_features
from .cfi import SubsetEncoder


class QM9QGNN(nn.Module):
    """``n`` is the padded atom count; the prediction is invariant under any
    permutation of the n slots, real and padded alike."""

    def __init__(self, n: int, j: int, D: int = 6, k: int = 3, num_rounds: int = 2,
                 encoder_hidden: int = 32, readout_hidden: int = 32,
                 init_std: float = 0.3):
        super().__init__()
        self.n, self.j, self.num_rounds = n, j, num_rounds
        emb = comb(D, k)
        feat_dim = 1 + j + NUM_ATOM_TYPES + NUM_BOND_TYPES
        self.embed_dim = (num_rounds + 1) * 2 * emb

        w, V = np.linalg.eigh(johnson_adjacency(n, j).toarray())
        self.register_buffer("eigvals", torch.tensor(w, dtype=torch.float64))
        self.register_buffer("eigvecs", torch.tensor(V, dtype=torch.float64))
        self.alphas = nn.Parameter(torch.randn(num_rounds, dtype=torch.float64) * init_std)

        self.enc0 = SubsetEncoder(feat_dim, emb, encoder_hidden)
        self.encs = nn.ModuleList(
            SubsetEncoder(feat_dim, emb, encoder_hidden) for _ in range(num_rounds)
        )
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, readout_hidden), nn.ReLU(),
            nn.Linear(readout_hidden, 1),
        )

    def _mix(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # exp(i alpha A_J) x  =  V diag(exp(i alpha lambda)) V^T x
        V = self.eigvecs.to(torch.complex128)
        phase = torch.exp(1j * alpha.to(torch.complex128) * self.eigvals)
        y = torch.einsum("pq,bpd->bqd", V, x)      # V^T x
        y = phase[None, :, None] * y
        return torch.einsum("pq,bqd->bpd", V, y)   # V (phase * V^T x)

    def embed(self, adj, atom_types, bond_types, node_mask) -> torch.Tensor:
        feats, valid = molecular_features(
            adj.to(torch.float64), atom_types, bond_types, node_mask, self.n, self.j
        )
        mask_f = valid.unsqueeze(-1).to(torch.float64)
        mask_c = mask_f.to(torch.complex128)

        def pool(z: torch.Tensor) -> torch.Tensor:
            return (torch.cat([z.real, z.imag], dim=-1) * mask_f).sum(dim=1)

        x = (self.enc0(feats) * mask_f).to(torch.complex128)
        pools = [pool(x)]
        for r in range(self.num_rounds):
            x = self._mix(x, self.alphas[r]) * mask_c   # keep amplitude in the valid sector
            x = x + (self.encs[r](feats) * mask_f).to(torch.complex128)
            x = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-12)
            x = x * mask_c
            pools.append(pool(x))
        return torch.cat(pools, dim=-1)                # [B, embed_dim] float64

    def forward(self, adj, atom_types, bond_types, node_mask) -> torch.Tensor:
        """Dense molecule tensors -> prediction [B]."""
        return self.head(self.embed(adj, atom_types, bond_types, node_mask)).squeeze(-1)
