import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "radar_real_mpqgnn.py"
SPEC = importlib.util.spec_from_file_location("radar_real_mpqgnn", MODULE_PATH)
radar = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(radar)


def test_fourier_angle_is_parameter_matched():
    mlp = radar.make_tsp_model(5, "mlp")
    fourier = radar.make_tsp_model(5, "fourier")
    mlp_angle = sum(p.numel() for p in mlp.adjacency[0].angle_mlp.parameters())
    fourier_angle = sum(p.numel() for p in fourier.adjacency[0].angle_mlp.parameters())
    assert mlp_angle == 81
    assert fourier_angle == 80


def test_shared_parameters_transfer_across_graph_sizes():
    torch.manual_seed(0)
    source = radar.make_tsp_model(5, "fourier")
    target = radar.make_tsp_model(6, "fourier")
    copied = radar.transfer_shared_parameters(source, target)
    assert copied == radar.trainable_parameters(source)
    for name, parameter in target.named_parameters():
        assert torch.equal(parameter, dict(source.named_parameters())[name])


def test_bounded_feature_sampling_converges():
    features = torch.tensor([[[-1.0, 0.0, 1.0]]])
    estimate = radar.sample_bounded_features(
        features, shots=200_000, rng=np.random.default_rng(0), bound=2.0
    )
    assert torch.max(torch.abs(estimate - features)) < 0.02


def test_small_lie_closure_matches_expected_dimensions():
    D, k = 4, 2
    path = [radar.hopping_generator(D, k, i, i + 1) for i in range(D - 1)]
    assert radar.lie_closure_dimension(path) == 6
    assert radar.lie_closure_dimension(path + [radar.hopping_generator(D, k, 0, 2)]) == 6
    assert radar.lie_closure_dimension(path + [radar.occupation_phase(D, k, (0,))]) == 16
    assert radar.lie_closure_dimension(path + [radar.occupation_phase(D, k, (0, 1))]) == 36


def test_cfi_measurement_audit_exposes_tiny_operational_margin():
    result = radar.run_cfi_measurement_audit([0], relabels=5)
    j3 = next(row for row in result["rows"] if row["j"] == 3)
    assert j3["bounded_mean_gap_l2"] < 1e-8
    assert j3["optimistic_single_graph_shots_for_5sigma"] > 1e12
