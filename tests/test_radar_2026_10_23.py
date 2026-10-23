import torch

from mp_qgnns.models.tsp import FourierSizeTSPQGNN, TSPQGNN


torch.set_default_dtype(torch.float64)


def test_fourier_size_model_is_identity_at_initialisation():
    torch.manual_seed(12)
    base = TSPQGNN(5, D=4, k=2, num_layers=1)
    torch.manual_seed(12)
    queried = FourierSizeTSPQGNN(5, D=4, k=2, num_layers=1)
    coords = torch.rand(3, 5, 2)
    assert torch.allclose(base(coords), queried(coords), atol=1e-12)


def test_fourier_query_changes_with_graph_size():
    model = FourierSizeTSPQGNN(5, D=4, k=2, num_layers=1)
    q5 = model.query_features(5)
    q7 = model.query_features(7)
    assert q5.shape == q7.shape == (6,)
    assert not torch.allclose(q5, q7)


def test_query_override_is_a_clean_alignment_control():
    torch.manual_seed(3)
    model = FourierSizeTSPQGNN(5, D=4, k=2, num_layers=1)
    with torch.no_grad():
        model.size_queries[0].weight.normal_(std=0.1)
    coords = torch.rand(2, 5, 2)
    normal = model(coords)
    model.query_size_override = 7
    mismatched = model(coords)
    assert not torch.allclose(normal, mismatched)
