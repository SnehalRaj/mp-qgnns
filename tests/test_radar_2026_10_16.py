import torch

from mp_qgnns.data_io.tsp import random_instances
from mp_qgnns.models.tsp import TSPQGNN


def test_real_tsp_quantum_gradient_has_expected_size():
    coords, _, heat = random_instances(5, 2, seed=123)
    torch.manual_seed(0)
    model = TSPQGNN(5, D=6, k=3, num_layers=2).double()
    logits = model(coords)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, heat)
    grads = torch.autograd.grad(loss, [layer.theta for layer in model.layers])
    assert len(grads) == 2
    assert all(g.numel() == 15 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_progressive_depth_copy_preserves_first_layer_parameters():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "experiments" / "radar_2026_10_16.py"
    spec = importlib.util.spec_from_file_location("radar_2026_10_16", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(1)
    one = TSPQGNN(5, num_layers=1)
    torch.manual_seed(2)
    two = TSPQGNN(5, num_layers=2)
    module._copy_first_layer(one, two)
    assert torch.allclose(one.layers[0].theta, two.layers[0].theta)
    assert torch.allclose(one.embed[0][0].weight, two.embed[0][0].weight)
    coords, _, _ = random_instances(5, 3, seed=7)
    assert torch.allclose(one(coords), two(coords), atol=1e-12)
