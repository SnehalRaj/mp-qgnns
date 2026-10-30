"""Hamming-weight-preserving evolution on the embedding register.

A pyramid of two-qubit RBS (Givens) gates on D qubits realises an orthogonal
matrix O in the single-particle sector. Its action on the weight-k subspace is
the k-th compound matrix of O, with entries det(O[I, J]) over k-subsets I, J.
Working directly with the compound matrix lets the reduced-basis (PyTorch) model
apply the same evolution as the circuit (PennyLane) without leaving the
C(D, k)-dimensional subspace. The orthogonal matrix is parametrised through the
Cayley transform of a skew-symmetric matrix, so the layer is trainable and stays
exactly orthogonal.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import torch
import torch.nn as nn


def compound_matrix(orthogonal: torch.Tensor, k: int) -> torch.Tensor:
    """k-th compound matrix C(D,k) x C(D,k) of a [D, D] orthogonal matrix."""
    D = orthogonal.shape[-1]
    if k == 1:
        return orthogonal
    subsets = torch.tensor(list(combinations(range(D), k)), dtype=torch.long,
                           device=orthogonal.device)
    blocks = orthogonal[subsets][:, :, subsets].permute(0, 2, 1, 3)
    return torch.linalg.det(blocks)


class CompoundPyramidLayer(nn.Module):
    """Trainable HW-preserving orthogonal evolution on the weight-k subspace."""

    def __init__(self, D: int, k: int, init_std: float = 0.1):
        super().__init__()
        self.D, self.k = D, k
        self.dim = comb(D, k)
        self.theta = nn.Parameter(torch.randn(D * (D - 1) // 2) * init_std)
        idx = torch.triu_indices(D, D, offset=1)
        self.register_buffer("_row", idx[0])
        self.register_buffer("_col", idx[1])
        self.register_buffer("_eye", torch.eye(D))

    def orthogonal(self) -> torch.Tensor:
        D = self.D
        skew = torch.zeros(D, D, device=self.theta.device, dtype=self.theta.dtype)
        skew[self._row, self._col] = self.theta
        skew = skew - skew.T
        eye = self._eye.to(self.theta.dtype)
        return torch.linalg.solve(eye + skew, eye - skew)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer to a state ``x`` of shape [..., C(D, k)]."""
        return torch.matmul(x, compound_matrix(self.orthogonal(), self.k).T)


def rbs_pyramid_matrix(angles: torch.Tensor, D: int) -> torch.Tensor:
    """Single-particle matrix of the PennyLane RBS pyramid.

    The sign convention matches ``qml.SingleExcitation(theta)`` in
    :func:`mp_qgnns.core.circuits.rbs_pyramid`.
    """
    matrix = torch.eye(D, dtype=angles.dtype, device=angles.device)
    pairs = [(q, q + 1) for offset in range(D - 1) for q in range(D - 1 - offset)]
    for theta, (a, b) in zip(angles, pairs):
        c, s = torch.cos(theta / 2), torch.sin(theta / 2)
        givens = torch.eye(D, dtype=angles.dtype, device=angles.device)
        givens[a, a], givens[a, b] = c, s
        givens[b, a], givens[b, b] = -s, c
        matrix = givens @ matrix
    return matrix


class RBSPyramidLayer(nn.Module):
    """Trainable circuit-native RBS pyramid with an optional fixed gate mask."""

    def __init__(self, D: int, k: int, angles: torch.Tensor | None = None):
        super().__init__()
        self.D, self.k = D, k
        n_angles = D * (D - 1) // 2
        initial = torch.zeros(n_angles) if angles is None else angles.detach().clone()
        self.angles = nn.Parameter(initial)
        self.register_buffer("gate_mask", torch.ones(n_angles, dtype=initial.dtype))

    def orthogonal(self) -> torch.Tensor:
        return rbs_pyramid_matrix(self.angles * self.gate_mask, self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, compound_matrix(self.orthogonal(), self.k).T)
