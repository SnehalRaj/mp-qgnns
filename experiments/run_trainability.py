"""Gradient variance versus register size for the HW-preserving ansatz.

    python experiments/run_trainability.py

At fixed Hamming weight the subspace grows polynomially in the qubit count, so
the gradient variance decays polynomially rather than exponentially.
"""
from mp_qgnns.trainability import variance_sweep


def main():
    rows = variance_sweep([4, 6, 8, 10, 12, 14], k=2, num_layers=3, num_samples=300)
    print(f"{'D':>3} {'subspace dim':>13} {'grad variance':>15}")
    for r in rows:
        print(f"{r['D']:>3} {r['dim']:>13} {r['grad_var']:>15.3e}")


if __name__ == "__main__":
    main()
