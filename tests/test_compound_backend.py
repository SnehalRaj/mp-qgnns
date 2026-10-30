import numpy as np
import pytest
import torch

from mp_qgnns.core.circuits import hw_block, pyramid_pairs, pyramid_unitary
from mp_qgnns.core.compound import compound_matrix, rbs_pyramid_matrix


@pytest.mark.parametrize("D,k", [(4, 2), (6, 3), (6, 2), (8, 4)])
def test_circuit_block_is_compound_of_single_particle(D, k):
    rng = np.random.default_rng(0)
    angles = rng.normal(size=len(pyramid_pairs(D))) * 0.7
    U = pyramid_unitary(angles, D)
    single = hw_block(U, D, 1)
    block_k = hw_block(U, D, k)
    compound = compound_matrix(torch.tensor(single), k).numpy()
    assert np.abs(block_k - compound).max() < 1e-10


@pytest.mark.parametrize("D", [3, 4, 6])
def test_differentiable_rbs_matrix_matches_pennylane(D):
    rng = np.random.default_rng(100 + D)
    angles = torch.tensor(rng.normal(size=len(pyramid_pairs(D))), dtype=torch.float64)
    expected = hw_block(pyramid_unitary(angles.numpy(), D), D, 1).real
    actual = rbs_pyramid_matrix(angles, D).detach().numpy()
    assert np.allclose(actual, expected, atol=1e-10)
