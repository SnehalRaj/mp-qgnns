"""Reproduce the WL-separation climb for a graph family and model.

    python experiments/run_cfi_climb.py --family k3 --model qgnn
    python experiments/run_cfi_climb.py --family k4 --model johnson_gin
    python experiments/run_cfi_climb.py --family prism --model qgnn --quick

A model that ascends the hierarchy scores at chance for j below the family's WL
threshold and separates (accuracy 1.0) at it, with a near-zero invariance spread.
The quantum circuit is simulated in the reduced (Hamming-weight-j) basis; at
n = 40 that subspace has C(40, 4) = 91,390 states, so CFI(K4) is run with the
classical Johnson-GIN baseline, which has the same set-j-WL distinguishing power.
"""
import argparse

import torch

from mp_qgnns import EquivariantQGNN, GIN, JohnsonGIN, evaluate, load_cfi, prism_vs_k33

torch.set_default_dtype(torch.float64)

FAMILIES = {"prism": prism_vs_k33, "k3": lambda: load_cfi("k3"), "k4": lambda: load_cfi("k4")}


def _build(model: str, n: int, j: int):
    if model == "qgnn":
        if n >= 40:
            raise SystemExit("qgnn is infeasible to simulate at n=40; use --model johnson_gin")
        return EquivariantQGNN(n, j)
    if model == "johnson_gin":
        return JohnsonGIN(n, j)
    return GIN(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), default="k3")
    ap.add_argument("--model", choices=["qgnn", "johnson_gin", "gin"], default="qgnn")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 seeds, fewer relabellings")
    args = ap.parse_args()

    a0, a1, n, sep = FAMILIES[args.family]()
    seeds = tuple(range(2 if args.quick else args.seeds))
    n_relabel = 50 if args.quick else 100
    epochs = 120 if args.quick else 200

    print(f"family={args.family} (n={n}, WL threshold j={sep})  model={args.model}  seeds={len(seeds)}")
    print(f"{'j':>3} {'test acc':>14} {'invariance':>12}")
    for j in range(1, sep + 1):
        res = evaluate(lambda: _build(args.model, n, j), a0, a1, n, j,
                       seeds=seeds, n_relabel=n_relabel, epochs=epochs,
                       inv_relabel=20 if n >= 40 else 50)
        print(f"{j:>3} {res['test_acc_mean']:>7.3f} +/- {res['test_acc_std']:.3f}"
              f" {res['invariance_spread_max']:>12.1e}")


if __name__ == "__main__":
    main()
