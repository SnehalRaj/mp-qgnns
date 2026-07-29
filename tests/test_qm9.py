import numpy as np
import pytest
import torch

from mp_qgnns.core.circuits import mixing_unitary
from mp_qgnns.data_io.qm9 import molecular_features, synthetic_molecules
from mp_qgnns.models.qm9 import QM9QGNN
from mp_qgnns.training.qm9 import mae, train

torch.set_default_dtype(torch.float64)
N_MAX = 7


def test_forward_shape():
    adj, atom, bond, mask, _ = synthetic_molecules(N_MAX, 4, seed=0)
    assert QM9QGNN(N_MAX, j=2)(adj, atom, bond, mask).shape == (4,)


def test_permutation_invariance():
    adj, atom, bond, mask, _ = synthetic_molecules(N_MAX, 1, seed=0)
    torch.manual_seed(0)
    model = QM9QGNN(N_MAX, j=2).eval()
    base = model(adj, atom, bond, mask).item()
    n_real = int(mask[0].sum())
    rng = np.random.default_rng(1)
    for _ in range(10):
        p = np.arange(N_MAX)
        p[:n_real] = rng.permutation(n_real)
        out = model(adj[:, p][:, :, p], atom[:, p], bond[:, p][:, :, p], mask[:, p]).item()
        assert abs(out - base) < 1e-9


def test_training_reduces_mae():
    adj, atom, bond, mask, t = synthetic_molecules(N_MAX, 80, seed=0)
    te = synthetic_molecules(N_MAX, 40, seed=1)
    te_inp, te_t = te[:4], te[4]
    torch.manual_seed(0)
    model = QM9QGNN(N_MAX, j=2, num_rounds=2)
    before = mae(model, te_inp, te_t)
    train(model, (adj, atom, bond, mask), t, epochs=300, lr=3e-3)
    assert mae(model, te_inp, te_t) < before


@pytest.mark.parametrize("j", [1, 2, 3])
def test_mixing_matches_pennylane_circuit(j):
    """Subset mixing equals exp(i*alpha*H) on the weight-j block."""
    model = QM9QGNN(N_MAX, j=j)
    alpha = float(model.alphas[0])
    m = model.eigvals.shape[0]
    x = torch.randn(1, m, 3, dtype=torch.complex128)
    torch_out = model._mix(x, model.alphas[0])[0].detach().numpy()
    pl_out = mixing_unitary(N_MAX, j, alpha) @ x[0].numpy()
    assert np.abs(torch_out - pl_out).max() < 1e-10


def test_j1_features_carry_graph_structure():
    """At j = 1 the structural slot is the full-graph degree, not an edge count."""
    adj, atom, bond, mask, _ = synthetic_molecules(N_MAX, 2, seed=0)
    feats, _ = molecular_features(adj, atom, bond, mask, N_MAX, 1)
    degrees = (adj.sum(dim=-1) * mask)[0]
    assert feats[..., :2].abs().sum() > 0
    assert torch.allclose(feats[0, :, 0], degrees)


def test_subsets_exchange_information():
    """Mixing moves amplitude between subsets: guards against a per-subset encoder."""
    model = QM9QGNN(N_MAX, j=2).eval()
    m = model.eigvals.shape[0]
    x = torch.zeros(1, m, 4, dtype=torch.complex128)
    x[0, 0] = 1.0
    mixed = model._mix(x, model.alphas[0])[0]
    assert mixed[1:].abs().max().item() > 1e-6
