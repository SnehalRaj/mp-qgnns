"""Create the figures used in the radar MVP research report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"mlp": "#4C78A8", "fourier": "#F58518", "accent": "#54A24B"}


def plot_fourier(data: dict, output: Path) -> None:
    summary = data["fourier"]["summary"]
    sizes = np.array([5, 6, 7])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for variant, label in (("mlp", "Original MLP angle map"),
                           ("fourier", "Neural Fourier angle map")):
        means = np.array([summary[variant][f"n{n}_mean"] for n in sizes])
        stds = np.array([summary[variant][f"n{n}_std"] for n in sizes])
        excess = 100.0 * (means - 1.0)
        ax.errorbar(sizes, excess, yerr=100.0 * stds, marker="o", linewidth=2.2,
                    capsize=4, color=COLORS[variant], label=label)
    ax.axvline(5, color="#777777", linestyle="--", linewidth=1)
    ax.text(5.05, ax.get_ylim()[1] * 0.88, "trained here", color="#666666", fontsize=9)
    ax.set_xticks(sizes)
    ax.set_xlabel("Number of cities")
    ax.set_ylabel("Tour length above optimum (%)")
    ax.set_title("Fourier edge map does not improve cross-size transfer consistently")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_cfi(data: dict, output: Path) -> None:
    rows = [row for row in data["shots"]["rows"] if row["j"] == 3]
    shots = np.array(data["shots"]["configuration"]["shot_grid"], dtype=float)
    curves = np.array([
        [row["oracle_accuracy_by_shots"][str(int(s))] for s in shots] for row in rows
    ])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    mean, std = curves.mean(0), curves.std(0)
    ax.semilogx(shots, 100 * mean, marker="o", linewidth=2.4, color=COLORS["accent"])
    ax.fill_between(shots, 100 * (mean - std), 100 * (mean + std),
                    color=COLORS["accent"], alpha=0.18, label="one standard deviation")
    ax.axhline(50, color="#777777", linestyle="--", linewidth=1.2, label="random guessing")
    ax.axhline(90, color="#B279A2", linestyle=":", linewidth=1.2, label="90% accuracy")
    ax.set_xlabel("Shots per bounded feature (log scale)")
    ax.set_ylabel("Optimistic oracle accuracy (%)")
    ax.set_ylim(48, 102)
    ax.set_title("Noiseless CFI separation is not operational at realistic shot counts")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="both", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_lie(data: dict, output: Path) -> None:
    rows = data["lie"]["rows"]
    labels = [f"D={r['D']}, k={r['k']}\ndim={r['reduced_dimension']}" for r in rows]
    keys = [
        ("path_lie_dimension", "Existing hopping"),
        ("path_plus_long_range_dimension", "+ long-range hopping"),
        ("path_plus_one_body_phase_dimension", "+ one-body phase"),
        ("path_plus_two_body_phase_dimension", "+ two-body phase"),
    ]
    x = np.arange(len(rows))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    palette = ["#4C78A8", "#9ECAE9", "#F2CF5B", "#E45756"]
    for idx, ((key, label), color) in enumerate(zip(keys, palette)):
        fractions = [100.0 * r[key] / r["full_u_dimension"] for r in rows]
        ax.bar(x + (idx - 1.5) * width, fractions, width, label=label, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fraction of full fixed-sector algebra (%)")
    ax.set_ylim(0, 108)
    ax.set_title("A single two-body phase changes the accessible model class")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fourier(data, args.output_dir / "fourier_transfer.png")
    plot_cfi(data, args.output_dir / "cfi_shot_curve.png")
    plot_lie(data, args.output_dir / "lie_algebra_fraction.png")


if __name__ == "__main__":
    main()
