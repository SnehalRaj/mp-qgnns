import numpy as np
import torch

from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import TSPQGNN
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
