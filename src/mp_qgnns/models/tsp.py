"""Quantum graph neural network for the Euclidean travelling salesman problem.

Two registers: the node register is the unary subspace of n qubits, one basis
state per city; the embedding register is the weight-k subspace of D qubits. A
state is therefore [B, n, C(D, k)].

Each layer re-uploads the coordinates into the embedding register, mixes the node
register with one RBS gate per graph edge in canonical order
(``core.adjacency.EquivariantAdjacencyLayer``), then evolves the embedding
register with a compound-pyramid layer. The edge mixing is the message-passing
step: without it the cities never exchange information. An edge head reads pairs
of city embeddings and predicts which edges lie on the optimal tour.

Both quantum operations have circuit forms in ``core.circuits``
(``tsp_adjacency_ops``, RBS pyramid), checked in tests/test_tsp.py and
tests/test_compound_backend.py.
"""
from __future__ import annotations

from math import comb

import torch
import torch.nn as nn

from ..core.adjacency import EquivariantAdjacencyLayer
from ..core.compound import CompoundPyramidLayer


class EdgeHead(nn.Module):
    """Tour-edge logits from unordered pairs of node embeddings.

    (a + b, |a - b|, a * b) is symmetric under exchanging the endpoints. The first
    two recover the pair componentwise: max = (sum + |diff|) / 2.
    """

    def __init__(self, emb_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        B, N, _ = nodes.shape
        iu = torch.triu_indices(N, N, offset=1, device=nodes.device)
        a, b = nodes[:, iu[0]], nodes[:, iu[1]]
        feat = torch.cat([a + b, (a - b).abs(), a * b], dim=-1)
        scores = self.net(feat).squeeze(-1)
        logits = torch.zeros(B, N, N, device=nodes.device, dtype=nodes.dtype)
        logits[:, iu[0], iu[1]] = scores
        logits[:, iu[1], iu[0]] = scores
        return logits


class TSPQGNN(nn.Module):
    def __init__(self, n_cities: int, D: int = 6, k: int = 3, num_layers: int = 2,
                 embed_hidden: int = 64, readout_hidden: int = 64, mlp_hidden: int = 16):
        super().__init__()
        emb = comb(D, k)
        self.num_layers = num_layers
        self.embed = nn.ModuleList(
            nn.Sequential(nn.Linear(2, embed_hidden), nn.ReLU(), nn.Linear(embed_hidden, emb))
            for _ in range(num_layers)
        )
        self.adjacency = nn.ModuleList(
            EquivariantAdjacencyLayer(n_cities, mlp_hidden) for _ in range(num_layers)
        )
        self.layers = nn.ModuleList(CompoundPyramidLayer(D, k) for _ in range(num_layers))
        self.head = EdgeHead(emb, readout_hidden)

    def node_embeddings(self, coords: torch.Tensor) -> torch.Tensor:
        x = None
        for r in range(self.num_layers):
            z = self.embed[r](coords)
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            x = z if x is None else x + z
            x = self.adjacency[r](x, coords)
            x = self.layers[r](x)
        return x

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords [B, n, 2] -> edge logits [B, n, n]."""
        return self.head(self.node_embeddings(coords))


class FourierSizeTSPQGNN(TSPQGNN):
    """TSPQGNN with a small, task-defined graph-size Fourier query.

    The query is shared by every graph of the same size and therefore carries no
    instance or label information.  It modulates the final node embeddings with
    a feature-wise affine map.  The final query layer is zero-initialised so the
    model begins exactly at the unconditioned ``TSPQGNN`` function.
    """

    def __init__(self, *args, query_frequencies=(0.25, 0.5, 1.0), **kwargs):
        super().__init__(*args, **kwargs)
        emb_dim = self.head.net[0].in_features // 3
        frequencies = torch.tensor(query_frequencies, dtype=torch.get_default_dtype())
        self.register_buffer("query_frequencies", frequencies)
        self.size_queries = nn.ModuleList(
            nn.Linear(2 * len(query_frequencies), 2 * emb_dim) for _ in range(self.num_layers)
        )
        for layer in self.size_queries:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.query_size_override = None

    def query_features(self, n_cities: int) -> torch.Tensor:
        # Scaling by 8 keeps the phase smooth over the small-TSP transfer range.
        phase = 2 * torch.pi * self.query_frequencies * (float(n_cities) / 8.0)
        return torch.cat([torch.sin(phase), torch.cos(phase)])

    def node_embeddings(self, coords: torch.Tensor) -> torch.Tensor:
        query_size = coords.shape[1] if self.query_size_override is None else self.query_size_override
        query = self.query_features(query_size).to(device=coords.device, dtype=coords.dtype)
        x = None
        for r in range(self.num_layers):
            z = self.embed[r](coords)
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            gamma, beta = self.size_queries[r](query).chunk(2)
            z = z * (1.0 + torch.tanh(gamma).view(1, 1, -1)) + beta.view(1, 1, -1)
            x = z if x is None else x + z
            x = self.adjacency[r](x, coords)
            x = self.layers[r](x)
        return x
