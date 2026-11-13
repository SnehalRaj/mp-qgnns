"""Render a plain-English summary of the 13 November radar experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(root: Path, name: str):
    return json.loads((root / f"{name}.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/radar_2026_11_13"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inv, qgf, vin = (load(args.results, n) for n in ("invariance", "qgf", "vinfo"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    fig.suptitle("MP-QGNN radar MVPs: lower tour gap is better", fontsize=15, weight="bold")

    # A test graph is geometrically identical in both columns; only node labels change.
    m = inv["summary"]["median"]
    x = np.arange(2)
    axes[0].bar(x - 0.18, [m["equivariant"]["sorted_gap_percent"], m["equivariant"]["random_gap_percent"]],
                0.36, label="MP-QGNN (equivariant)", color="#2563eb")
    axes[0].bar(x + 0.18, [m["label_ordered"]["sorted_gap_percent"], m["label_ordered"]["random_gap_percent"]],
                0.36, label="Label-ordered control", color="#ef4444")
    axes[0].set_xticks(x, ["Labels sorted", "Labels shuffled"])
    axes[0].set_title("A. Relabelling shortcut — PASS")
    axes[0].set_ylabel("Median tour gap (%)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].text(1, m["label_ordered"]["random_gap_percent"] + 0.5, "3.00% vs 11.49%", ha="center", fontsize=9)

    qm = qgf["summary"]["median_gap_percent"]
    qnames = ["Baseline", "Full time series", "Mean + std"]
    qvals = [qm["baseline"], qm["qgf_full"], qm["qgf_stat"]]
    axes[1].bar(qnames, qvals, color=["#64748b", "#f59e0b", "#10b981"])
    axes[1].set_title("B. Neutral-atom features — PIVOT")
    axes[1].tick_params(axis="x", labelrotation=18)
    axes[1].text(2, qvals[2] + 0.25, "0.94 points lower\n(2 of 3 seeds)", ha="center", fontsize=9)

    vm = vin["summary"]["median"]
    vkeys = ["full", "uniform", "top_pvi", "mixed_pvi"]
    vnames = ["Full data", "Random half", "Top-score half", "Coverage mix"]
    vvals = [vm[k]["gap_transfer_n6_percent"] for k in vkeys]
    axes[2].bar(vnames, vvals, color=["#64748b", "#94a3b8", "#8b5cf6", "#ec4899"])
    axes[2].set_title("C. Data selection — STOP")
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].text(2, vvals[2] + 0.18, "Better median, but\nonly 1 of 3 seeds", ha="center", fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)
    fig.text(0.5, -0.02, "Tour gap = extra route length versus the exact optimum. 0% is perfect; 10% means a route 10% too long.",
             ha="center", fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
