"""Exact set-j-WL first-separating level for each graph pair.

    python experiments/run_wl_ground_truth.py
"""
from mp_qgnns import load_cfi, prism_vs_k33
from mp_qgnns.core.wl import first_separating_j

PAIRS = {
    "prism vs K_{3,3}  (n=6)": prism_vs_k33(),
    "CFI(K3)           (n=18)": load_cfi("k3"),
    "CFI(K4)           (n=40)": load_cfi("k4"),
}


def main():
    for name, (a0, a1, n, expected) in PAIRS.items():
        j = first_separating_j(a0, a1, n, expected + 1)
        mark = "ok" if j == expected else "MISMATCH"
        print(f"{name}: first separated by set-{j}-WL  [{mark}]")


if __name__ == "__main__":
    main()
