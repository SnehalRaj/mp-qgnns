import numpy as np
import pytest
import torch
from scipy.linalg import expm

from mp_qgnns.core.circuits import mixing_unitary
from mp_qgnns.core.subsets import johnson_adjacency
from mp_qgnns.models.cfi import EquivariantQGNN


@pytest.mark.parametrize("n,j", [(5, 2), (6, 3), (7, 3), (8, 4)])
@pytest.mark.parametrize("alpha", [0.3, 1.1])
def test_pennylane_matches_reduced_basis(n, j, alpha):
    A = johnson_adjacency(n, j).toarray()
    assert np.abs(mixing_unitary(n, j, alpha) - expm(1j * alpha * A)).max() < 1e-12


@pytest.mark.parametrize("n,j", [(6, 3), (8, 4)])
def test_pennylane_matches_torch_mixing(n, j):
    torch.set_default_dtype(torch.float64)
    model = EquivariantQGNN(n, j)
    alpha = float(model.alphas[0])
    x = torch.randn(1, johnson_adjacency(n, j).shape[0], 1, dtype=torch.complex128)
    torch_out = model._mix(x, model.alphas[0])[0, :, 0].detach().numpy()
    pl_out = mixing_unitary(n, j, alpha) @ x[0, :, 0].numpy()
    assert np.abs(torch_out - pl_out).max() < 1e-10
