import numpy as np
import pytest
import torch

from mp_qgnns.core.adjacency import EquivariantAdjacencyLayer
from mp_qgnns.core.circuits import tsp_adjacency_unitary
from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import EdgeHead, TSPQGNN
from mp_qgnns.training.tsp import tour_ratio, train

torch.set_default_dtype(torch.float64)


def test_forward_is_symmetric():
    coords, _, _ = random_instances(5, 4, seed=0)
    logits = TSPQGNN(5)(coords)
    assert logits.shape == (4, 5, 5)
    assert torch.allclose(logits, logits.transpose(1, 2))


def test_training_improves_tour_ratio():
    torch.manual_seed(0)
    coords, _, heatmaps = random_instances(5, 120, seed=0)
    te_coords, te_tours, _ = random_instances(5, 60, seed=1)
    opt = np.array([tour_length(te_coords[i].numpy(), te_tours[i].numpy())
                    for i in range(len(te_coords))])
    model = TSPQGNN(5, num_layers=2)
    before = tour_ratio(model, te_coords, opt)
    train(model, coords, heatmaps, epochs=200, lr=3e-3)
    after = tour_ratio(model, te_coords, opt)
    assert after < before
    assert after < 1.10


def test_node_embeddings_are_equivariant():
    """Relabelling the cities permutes the node embeddings and nothing else.

    This is what the canonical edge ordering buys: RBS gates on overlapping pairs
    do not commute, so a fixed lexicographic gate order fails here.
    """
    coords, _, _ = random_instances(6, 1, seed=0)
    torch.manual_seed(0)
    model = TSPQGNN(6, num_layers=2).eval()
    base = model.node_embeddings(coords)[0]
    rng = np.random.default_rng(1)
    for _ in range(10):
        p = rng.permutation(6)
        out = model.node_embeddings(coords[:, p])[0]
        assert torch.allclose(out, base[p], atol=1e-12)


def test_edge_logits_are_equivariant():
    """Relabelling the cities permutes the logit matrix."""
    coords, _, _ = random_instances(6, 1, seed=0)
    torch.manual_seed(0)
    model = TSPQGNN(6, num_layers=2).eval()
    base = model(coords)[0]
    rng = np.random.default_rng(1)
    for _ in range(10):
        p = rng.permutation(6)
        assert torch.allclose(model(coords[:, p])[0], base[np.ix_(p, p)], atol=1e-12)


def test_edge_head_is_symmetric():
    """An edge scores the same however its endpoints are ordered."""
    torch.manual_seed(0)
    head = EdgeHead(8).eval()
    a, b = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    nodes = torch.cat([a, b], dim=1)
    assert torch.allclose(head(nodes)[0, 0, 1], head(nodes.flip(1))[0, 0, 1], atol=1e-12)


def test_adjacency_mixes_nodes():
    """The edge layer moves amplitude between cities: guards against a per-city encoder."""
    coords, _, _ = random_instances(5, 1, seed=0)
    torch.manual_seed(0)
    layer = EquivariantAdjacencyLayer(5)
    x = torch.zeros(1, 5, 4)
    x[0, 0] = 1.0
    assert layer(x, coords)[0, 1:].abs().max().item() > 1e-6


@pytest.mark.parametrize("n", [4, 5, 6])
def test_adjacency_matches_pennylane_circuit(n):
    """The node mixing equals its SingleExcitation circuit on the unary subspace."""
    coords, _, _ = random_instances(n, 1, seed=0)
    torch.manual_seed(0)
    layer = EquivariantAdjacencyLayer(n)
    si, sj = layer.orient_and_sort_edges(coords)
    angles = layer.compute_angles(coords, si, sj)

    torch_u = layer(torch.eye(n).unsqueeze(0), coords)[0].detach().numpy()
    pl_u = tsp_adjacency_unitary(si[0].tolist(), sj[0].tolist(),
                                 angles[0].detach().tolist(), n)
    assert np.abs(torch_u - pl_u).max() < 1e-10


def test_adjacency_is_orthogonal():
    coords, _, _ = random_instances(6, 1, seed=0)
    torch.manual_seed(0)
    layer = EquivariantAdjacencyLayer(6)
    u = layer(torch.eye(6).unsqueeze(0), coords)[0].detach()
    assert torch.allclose(u.T @ u, torch.eye(6), atol=1e-10)
