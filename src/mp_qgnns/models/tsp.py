"""Quantum graph neural network for the Euclidean travelling salesman problem.

Each city's coordinates are loaded into the weight-k embedding register as a
normalised amplitude vector; the register is evolved by HW-preserving
compound-pyramid layers, giving a per-city embedding. An edge head reads pairs of
city embeddings and predicts which edges belong to the optimal tour. The quantum
evolution has the circuit form in ``pennylane_compound`` (RBS pyramid); the two
agree to machine precision (tests/test_compound_backend.py).
"""
from __future__ import annotations

from math import comb

import torch
import torch.nn as nn

from ..core.compound import CompoundPyramidLayer


class EdgeHead(nn.Module):
    """Predict tour-edge logits from pairs of node embeddings."""

    def __init__(self, emb_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        B, N, _ = nodes.shape
        iu = torch.triu_indices(N, N, offset=1, device=nodes.device)
        a, b = nodes[:, iu[0]], nodes[:, iu[1]]
        feat = torch.cat([a, b, a - b, a * b], dim=-1)
        scores = self.net(feat).squeeze(-1)
        logits = torch.zeros(B, N, N, device=nodes.device, dtype=nodes.dtype)
        logits[:, iu[0], iu[1]] = scores
        logits[:, iu[1], iu[0]] = scores
        return logits


class TSPQGNN(nn.Module):
    def __init__(self, n_cities: int, D: int = 6, k: int = 3, num_layers: int = 2,
                 embed_hidden: int = 64, readout_hidden: int = 64):
        super().__init__()
        emb = comb(D, k)
        self.coord_embed = nn.Sequential(nn.Linear(2, embed_hidden), nn.ReLU(),
                                         nn.Linear(embed_hidden, emb))
        self.layers = nn.ModuleList(CompoundPyramidLayer(D, k) for _ in range(num_layers))
        self.head = EdgeHead(emb, readout_hidden)

    def node_embeddings(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.coord_embed(coords)
        x = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-12)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords [B, n, 2] -> edge logits [B, n, n]."""
        return self.head(self.node_embeddings(coords))
