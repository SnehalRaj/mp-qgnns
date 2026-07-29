"""PyTorch graph-classification models on the j-subset (Johnson) lift.

All three models are permutation-equivariant by construction: the graph enters
only through the iso-type features of induced subgraphs, the subset-axis mixing
commutes with vertex relabelling, and the readout pools over subsets. Run in
float64; the round-two WL signal on genuine CFI gadgets sits at the float32
noise floor.
"""
from __future__ import annotations

from math import comb

import numpy as np
import torch
import torch.nn as nn

from ..core.subsets import isotype_features_batched, johnson_adjacency


def _sparse_johnson(n: int, j: int, dtype=torch.float64) -> torch.Tensor:
    csr = johnson_adjacency(n, j).tocoo()
    idx = torch.tensor(np.vstack([csr.row, csr.col]), dtype=torch.long)
    val = torch.tensor(csr.data, dtype=dtype)
    return torch.sparse_coo_tensor(idx, val, csr.shape).coalesce()


class SubsetEncoder(nn.Module):
    """Per-subset MLP mapping iso-type features to a unit-norm embedding.

    Shared by the CFI and QM9 models.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)


class EquivariantQGNN(nn.Module):
    """Reduced-basis quantum GNN with exactly S_N-equivariant node mixing.

    The node register is the Hamming-weight-j subspace, indexed by the C(n, j)
    subsets. Mixing is the unitary U_r = exp(i * alpha_r * A_J), where A_J is the
    symmetric Johnson adjacency; because A_J commutes with every induced subset
    permutation, U_r is equivariant at any parameter value. Iso-type features are
    re-uploaded between rounds and the subset embeddings are renormalised, which
    supplies the nonlinearity needed for multi-round refinement. The readout is a
    jumping-knowledge sum-pool of the real and imaginary parts.
    """

    def __init__(self, n: int, j: int, D: int = 6, k: int = 3, num_rounds: int = 3,
                 encoder_hidden: int = 32, init_std: float = 0.3):
        super().__init__()
        self.n, self.j, self.num_rounds = n, j, num_rounds
        emb = comb(D, k)
        self.embed_dim = (num_rounds + 1) * 2 * emb

        w, V = np.linalg.eigh(johnson_adjacency(n, j).toarray())
        self.register_buffer("eigvals", torch.tensor(w, dtype=torch.float64))
        self.register_buffer("eigvecs", torch.tensor(V, dtype=torch.float64))
        self.alphas = nn.Parameter(torch.randn(num_rounds, dtype=torch.float64) * init_std)
        self.enc0 = SubsetEncoder(1 + j, emb, encoder_hidden)
        self.encs = nn.ModuleList(SubsetEncoder(1 + j, emb, encoder_hidden) for _ in range(num_rounds))

    def _mix(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # exp(i alpha A_J) x  =  V diag(exp(i alpha lambda)) V^T x
        V = self.eigvecs.to(torch.complex128)
        phase = torch.exp(1j * alpha.to(torch.complex128) * self.eigvals)
        y = torch.einsum("pq,bpd->bqd", V, x)      # V^T x
        y = phase[None, :, None] * y
        return torch.einsum("pq,bqd->bpd", V, y)   # V (phase * V^T x)

    def embed(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        feats = isotype_features_batched(adj, self.n, self.j)
        x = self.enc0(feats).to(torch.complex128)
        pools = [torch.cat([x.real, x.imag], dim=-1).sum(dim=1)]
        for r in range(self.num_rounds):
            x = self._mix(x, self.alphas[r])
            x = x + self.encs[r](feats).to(torch.complex128)
            x = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-12)
            pools.append(torch.cat([x.real, x.imag], dim=-1).sum(dim=1))
        return torch.cat(pools, dim=-1)


class JohnsonGIN(nn.Module):
    """Classical message-passing baseline on the Johnson graph J(n, j).

    Mirrors the distinguishing power of the equivariant QGNN and stays tractable
    where direct circuit simulation does not (n = 40 has C(40, 4) = 91,390
    subsets). Sum aggregation, per-node renormalisation, jumping-knowledge pool.
    """

    def __init__(self, n: int, j: int, hidden: int = 128, depth: int = 3, init_gain: float = 8.0):
        super().__init__()
        self.n, self.j, self.depth = n, j, depth
        self.embed_dim = (depth + 1) * hidden
        self.register_buffer("_johnson", _sparse_johnson(n, j))
        self.input_enc = nn.Linear(1 + j, hidden)
        self.layers = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(depth))
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.mul_(init_gain)

    def _renorm(self, h: torch.Tensor) -> torch.Tensor:
        return h / (torch.linalg.vector_norm(h, dim=-1, keepdim=True) + 1e-12)

    def embed(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        feats = isotype_features_batched(adj, self.n, self.j)
        h = self._renorm(torch.tanh(self.input_enc(feats)))
        pools = [h.sum(dim=1)]
        m = h.shape[1]
        for layer in self.layers:
            flat = h.permute(1, 0, 2).reshape(m, -1)
            agg = torch.sparse.mm(self._johnson, flat).reshape(m, h.shape[0], -1).permute(1, 0, 2)
            h = self._renorm(torch.tanh(layer(torch.cat([h, agg], dim=-1))))
            pools.append(h.sum(dim=1))
        return torch.cat(pools, dim=-1)


class GIN(nn.Module):
    """Standard message-passing GNN on the original graph (1-WL ceiling).

    Provided as the reference that cannot separate 1-WL-indistinguishable graphs.
    Node features are degrees; sum aggregation over graph neighbours.
    """

    def __init__(self, n: int, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.n, self.depth = n, depth
        self.embed_dim = hidden
        self.input_enc = nn.Linear(1, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth))

    def embed(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        deg = adj.sum(dim=2, keepdim=True)
        h = torch.relu(self.input_enc(deg))
        for layer in self.layers:
            h = torch.relu(layer(h + torch.bmm(adj, h)))
        return h.sum(dim=1)
