import numpy as np
import torch

from mp_qgnns.data_io.qm9 import synthetic_molecules
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
    model = QM9QGNN(N_MAX, j=2, num_layers=2)
    before = mae(model, te_inp, te_t)
    train(model, (adj, atom, bond, mask), t, epochs=300, lr=3e-3)
    assert mae(model, te_inp, te_t) < before
