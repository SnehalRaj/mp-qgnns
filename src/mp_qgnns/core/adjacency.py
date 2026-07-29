"""Graph-conditioned mixing on the node register.

The node register is the unary (weight-one) subspace of n qubits, one basis state
per city, so a state is [B, n, d]. One RBS gate per graph edge mixes the two rows
it joins, with the angle produced by an MLP on the edge geometry.

Gate order matters because RBS gates on overlapping pairs do not commute, and a
fixed lexicographic order is not permutation equivariant. Edges are therefore
oriented by node key and sorted by the edge key (arctan distance, then the two
endpoint coordinates), which depends on the geometry rather than on the labelling.

Circuit form: ``core.circuits.tsp_adjacency_ops`` (tests/test_tsp.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class EquivariantAdjacencyLayer(nn.Module):
    """Per-edge Givens mixing of the node register, in canonical edge order."""

    def __init__(self, n: int, mlp_hidden: int = 16):
        super().__init__()
        self.n = n
        self.num_edges = n * (n - 1) // 2
        self.angle_mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1)
        )
        pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]
        self.register_buffer("edge_pairs", torch.tensor(pairs, dtype=torch.long))
        self.register_buffer("_sort_weights", torch.tensor([1e8, 1e6, 1e4, 1e2, 1.0]))

    def orient_and_sort_edges(self, coords: torch.Tensor):
        """coords [B, n, 2] -> oriented, canonically sorted endpoints ([B, E], [B, E])."""
        B = coords.shape[0]
        E = self.num_edges
        ei = self.edge_pairs[:, 0].unsqueeze(0).expand(B, -1)
        ej = self.edge_pairs[:, 1].unsqueeze(0).expand(B, -1)

        key_i = torch.gather(coords, 1, ei.unsqueeze(-1).expand(-1, -1, 2))
        key_j = torch.gather(coords, 1, ej.unsqueeze(-1).expand(-1, -1, 2))
        i_less = (key_i[..., 0] < key_j[..., 0]) | (
            (torch.abs(key_i[..., 0] - key_j[..., 0]) < 1e-10) & (key_i[..., 1] < key_j[..., 1])
        )
        oi = torch.where(i_less, ei, ej)
        oj = torch.where(i_less, ej, ei)

        ci = torch.gather(coords, 1, oi.unsqueeze(-1).expand(-1, -1, 2))
        cj = torch.gather(coords, 1, oj.unsqueeze(-1).expand(-1, -1, 2))
        kappa = torch.arctan(torch.norm(ci - cj, dim=-1))
        edge_keys = torch.cat([kappa.unsqueeze(-1), ci, cj], dim=-1)
        order = torch.argsort((edge_keys * self._sort_weights).sum(dim=-1), dim=-1)

        batch = torch.arange(B, device=coords.device).unsqueeze(1).expand(-1, E)
        return oi[batch, order], oj[batch, order]

    def compute_angles(self, coords, sorted_i, sorted_j) -> torch.Tensor:
        """Angles from [distance, dx, dy], bounded to [-pi/2, pi/2]."""
        ci = torch.gather(coords, 1, sorted_i.unsqueeze(-1).expand(-1, -1, 2))
        cj = torch.gather(coords, 1, sorted_j.unsqueeze(-1).expand(-1, -1, 2))
        diff = cj - ci
        feats = torch.cat([torch.norm(diff, dim=-1, keepdim=True), diff], dim=-1)
        angles = self.angle_mlp(feats.view(-1, 3))
        return (torch.tanh(angles) * (np.pi / 2)).view(coords.shape[0], self.num_edges)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x [B, n, d], coords [B, n, 2] -> [B, n, d]."""
        sorted_i, sorted_j = self.orient_and_sort_edges(coords)
        angles = self.compute_angles(coords, sorted_i, sorted_j)
        batch = torch.arange(x.shape[0], device=x.device)

        for e in range(self.num_edges):
            i, j, theta = sorted_i[:, e], sorted_j[:, e], angles[:, e]
            c = torch.cos(theta / 2).unsqueeze(-1)
            s = torch.sin(theta / 2).unsqueeze(-1)
            xi, xj = x[batch, i, :], x[batch, j, :]
            i_exp = i.view(-1, 1, 1).expand(-1, 1, x.shape[2])
            j_exp = j.view(-1, 1, 1).expand(-1, 1, x.shape[2])
            x = x.scatter(1, i_exp, (c * xi - s * xj).unsqueeze(1))
            x = x.scatter(1, j_exp, (s * xi + c * xj).unsqueeze(1))
        return x
