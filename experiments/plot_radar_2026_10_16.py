"""Plot the three 16 October radar MVP outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    rows = data["grid"]["rows"]
    for depth, marker in [(1, "o"), (2, "s"), (3, "^")]:
        selected = [r for r in rows if r["depth"] == depth]
        axes[0].scatter(
            [r["effective_fraction"] for r in selected],
            [100 * r["greedy_excess"] for r in selected],
            label=f"depth {depth}", marker=marker, s=48, alpha=0.85,
        )
    rho = data["grid"]["analysis"]["effective_fraction_vs_greedy_excess"]["rho"]
    axes[0].set_title(f"Fisher dimension is not a selector\nSpearman rho = {rho:.2f}")
    axes[0].set_xlabel("Effective fraction of quantum directions")
    axes[0].set_ylabel("Greedy tour excess (%)")
    axes[0].legend(frameon=False)

    loc = data["locality"]["rows"]
    n = [r["n_cities"] for r in loc]
    axes[1].plot(n, [r["local_variance"] for r in loc], "o-", label="one random edge")
    axes[1].plot(
        n, [r["global_sqrt_edges_variance"] for r in loc], "s-",
        label="global mean x sqrt(edges)",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("No local-gradient scaling gain\nafter averaging control")
    axes[1].set_xlabel("Number of cities")
    axes[1].set_ylabel("Gradient variance")
    axes[1].legend(frameon=False, fontsize=8)

    cur = data["curriculum"]["rows"]
    seeds = np.arange(len(cur))
    width = 0.36
    axes[2].bar(
        seeds - width / 2,
        [100 * (r["scratch_greedy_ratio"] - 1) for r in cur],
        width, label="2 layers from scratch",
    )
    axes[2].bar(
        seeds + width / 2,
        [100 * (r["grown_greedy_ratio"] - 1) for r in cur],
        width, label="1 to 2 layers",
    )
    axes[2].set_title("Function-preserving growth loses\non all five seeds")
    axes[2].set_xlabel("Seed")
    axes[2].set_ylabel("Greedy tour excess (%)")
    axes[2].set_xticks(seeds)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Real MP-QGNN radar MVPs, 16 October 2026", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
