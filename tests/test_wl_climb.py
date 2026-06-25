"""The climb: chance below the WL threshold, separation at it."""
import pytest
import torch

from mp_qgnns import EquivariantQGNN, JohnsonGIN, evaluate, prism_vs_k33

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("model_cls", [EquivariantQGNN, JohnsonGIN])
def test_climb_on_prism_vs_k33(model_cls):
    a0, a1, n, sep = prism_vs_k33()
    below = evaluate(lambda: model_cls(n, 2), a0, a1, n, 2,
                     seeds=(0, 1), n_relabel=50, epochs=120, inv_relabel=20)
    at = evaluate(lambda: model_cls(n, 3), a0, a1, n, 3,
                  seeds=(0, 1), n_relabel=50, epochs=120, inv_relabel=20)
    assert below["test_acc_mean"] < 0.7
    assert at["test_acc_mean"] > 0.95
    assert at["invariance_spread_max"] < 1e-4
