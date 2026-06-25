"""Train the QM9 regression model at j in {1, 2, 3} and report MAE.

    python experiments/run_qm9.py --qm9-root data/QM9 --molecules 3000 --target 4
    python experiments/run_qm9.py --synthetic

The real run loads QM9 through PyTorch Geometric (downloaded on first use);
target 4 is the HOMO-LUMO gap in eV. ``--synthetic`` runs on generated molecules
so the pipeline can be exercised without the download.
"""
import argparse

import torch

from mp_qgnns.data_io.qm9 import load_qm9, synthetic_molecules
from mp_qgnns.models.qm9 import QM9QGNN
from mp_qgnns.training.qm9 import mae, train

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qm9-root", default=None)
    ap.add_argument("--molecules", type=int, default=3000)
    ap.add_argument("--target", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    if args.synthetic:
        adj, atom, bond, mask, y = synthetic_molecules(args.n_max, args.molecules, seed=0)
    else:
        adj, atom, bond, mask, y = load_qm9(args.qm9_root, args.target, args.n_max, args.molecules)

    n_tr = int(0.8 * len(y))
    train_batch = (adj[:n_tr], atom[:n_tr], bond[:n_tr], mask[:n_tr])
    test_batch = (adj[n_tr:], atom[n_tr:], bond[n_tr:], mask[n_tr:])
    y_tr, y_te = y[:n_tr], y[n_tr:]

    print(f"{'j':>3} {'test MAE':>12}")
    for j in (1, 2, 3):
        torch.manual_seed(0)
        model = QM9QGNN(args.n_max, j=j, num_layers=2)
        train(model, train_batch, y_tr, epochs=args.epochs)
        print(f"{j:>3} {mae(model, test_batch, y_te):>12.4f}")


if __name__ == "__main__":
    main()
