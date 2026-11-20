"""Plot plain-English results for the 20 November MP-QGNN radar MVPs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(root, name):
    return json.loads((root / f"{name}.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/radar_2026_11_20"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tr, moe, cal = (load(a.results, n) for n in ("trainability", "moe", "calibration"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    fig.suptitle("MP-QGNN radar MVPs — lower tour gap is better", fontsize=15, weight="bold")

    cfg = tr["configs"]
    x = np.log10([c["raw_gradient_variance"] for c in cfg])
    y = [c["median_gap_percent"] for c in cfg]
    axes[0].scatter(x, y, s=60, color="#2563eb")
    for xx, yy, c in zip(x, y, cfg):
        axes[0].annotate(c["name"], (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("log10 initial gradient variance")
    axes[0].set_ylabel("Median TSP-6 tour gap (%)")
    axes[0].set_title("A. Gradient ranking — STOP")
    axes[0].text(0.04, 0.94, "ρ = +0.30 (wrong sign)", transform=axes[0].transAxes, va="top")

    mm = moe["summary"]["median"]
    keys = ["single_k2", "uniform", "soft_sharp", "soft_regularized"]
    labels = ["Best single\n(global median)", "Uniform", "Sharp router", "Revised router"]
    vals = [mm[k]["gap_transfer_n6_percent"] for k in keys]
    axes[1].bar(labels, vals, color=["#64748b", "#94a3b8", "#f59e0b", "#8b5cf6"])
    axes[1].set_title("B. Fixed-expert routing — STOP")
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].text(0.5, 0.94, "0/5 paired-seed wins", transform=axes[1].transAxes, ha="center", va="top")

    cm = cal["summary"]["median_gap_percent"]
    keys = ["ideal", "noisy", "naive_inverse", "ridge_inverse"]
    labels = ["Ideal", "Noisy", "Naive inverse", "Ridge revision"]
    vals = [cm[k] for k in keys]
    axes[2].bar(labels, vals, color=["#10b981", "#ef4444", "#f59e0b", "#2563eb"])
    axes[2].set_title("C. Reusable calibration — PIVOT")
    axes[2].tick_params(axis="x", labelrotation=16)
    axes[2].text(0.73, 0.94, "74% median recovery\n3/5 seeds clear 50%", transform=axes[2].transAxes,
                 ha="center", va="top")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, -0.02, "Tour gap = extra route length relative to the exact optimum. 0% is perfect.", ha="center")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
