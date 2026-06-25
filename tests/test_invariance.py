"""Permutation-invariance guardrail.

Every model must give an identical graph-level embedding for all relabellings of
a fixed graph, at randomly initialised (non-zero) parameters. This is the test
that distinguishes an equivariant construction from one that leaks vertex-label
structure.
"""
import numpy as np
import pytest
import torch

from mp_qgnns import GIN, EquivariantQGNN, JohnsonGIN, prism_vs_k33, relabel

torch.set_default_dtype(torch.float64)


def _invariance_spread(model, adj, n_relabel=50, seed=1):
    rng = np.random.default_rng(seed)
    batch = torch.tensor(np.stack([relabel(np.asarray(adj), rng) for _ in range(n_relabel)]))
    with torch.no_grad():
        emb = model.embed(batch)
    return float((emb.amax(0) - emb.amin(0)).abs().max())


@pytest.fixture
def graph():
    a0, _, n, _ = prism_vs_k33()
    return a0, n


@pytest.mark.parametrize("j", [2, 3])
def test_equivariant_qgnn_is_permutation_invariant(graph, j):
    a0, n = graph
    torch.manual_seed(0)
    assert _invariance_spread(EquivariantQGNN(n, j), a0) < 1e-6


@pytest.mark.parametrize("j", [2, 3])
def test_johnson_gin_is_permutation_invariant(graph, j):
    a0, n = graph
    torch.manual_seed(0)
    assert _invariance_spread(JohnsonGIN(n, j), a0) < 1e-6


def test_gin_is_permutation_invariant(graph):
    a0, n = graph
    torch.manual_seed(0)
    assert _invariance_spread(GIN(n), a0) < 1e-6
