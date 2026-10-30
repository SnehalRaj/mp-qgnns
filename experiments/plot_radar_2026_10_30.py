"""Make the human-readable summary figure for the 30 October radar MVPs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULL = json.loads((ROOT / "results/radar_2026_10_30_full.json").read_text())
ROBUST = json.loads((ROOT / "results/radar_2026_10_30_robustness.json").read_text())
OUT = ROOT / "docs/research_notes/radar_2026_10_30_summary.png"
OUT_SVG = ROOT / "docs/research_notes/radar_2026_10_30_summary.svg"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), constrained_layout=True)

# A: the ablation is deliberately shown on two CFI families to expose saturation.
cfi = FULL["evolution_ablation"]["analysis"]["cfi"]
x = np.arange(2)
axes[0].bar(x - 0.18, [cfi["prism"]["full_accuracy"], cfi["k3"]["full_accuracy"]],
            0.36, label="Full Johnson mixing", color="#2b6cb0")
axes[0].bar(x + 0.18, [cfi["prism"]["zero_mixing_accuracy"], cfi["k3"]["zero_mixing_accuracy"]],
            0.36, label="Mixing removed", color="#ed8936")
axes[0].set(xticks=x, xticklabels=["Prism/K3,3\n(easy)", "CFI K3\n(harder)"], ylim=(0, 1.08),
            ylabel="Graph-pair accuracy", title="A. Quantum evolution is task-dependent")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
axes[0].text(1, 0.50, "55-point gap", ha="center", va="bottom", fontweight="bold")

# B: task loss, rather than unitary distance, is the gate-pruning success criterion.
pruning = ROBUST["pruning_curve"]["analysis"]
fractions = np.array([25, 50, 75])
raw = np.array([100 * pruning[str(f / 100)]["raw_added_excess"] for f in fractions])
revised = np.array([100 * pruning[str(f / 100)]["revised_added_excess"] for f in fractions])
axes[1].plot(fractions, raw, "o-", color="#c53030", label="Prune only")
axes[1].plot(fractions, revised, "o-", color="#2f855a", label="Prune + 60-step fine-tune")
axes[1].axhline(1.0, color="#718096", linestyle="--", linewidth=1, label="Go limit (1 point)")
axes[1].set(xlabel="RBS gates removed", ylabel="Added mean tour excess (points)",
            xticks=fractions, title="B. Fine-tuning recovers heavy pruning")
axes[1].legend(frameon=False, fontsize=8)

# C: savings are relative to fixed allocation; labels retain the quality caveat.
readout = ROBUST["larger_graph_readout"]["analysis"]
names = ["Confidence", "Top-two retirement", "Sample one uncertain"]
keys = ["confidence", "gap_retirement", "gap_single"]
savings = [readout[k]["reduction_vs_fixed"] for k in keys]
added = [100 * readout[k]["added_excess_vs_confidence"] for k in keys]
bars = axes[2].bar(np.arange(3), savings, color=["#718096", "#805ad5", "#319795"])
axes[2].set(xticks=np.arange(3), xticklabels=names, ylabel="Fewer binary shots than fixed",
            title="C. Task-aware shot allocation (7 cities)")
axes[2].tick_params(axis="x", rotation=18)
for bar, delta in zip(bars, added):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                 f"{bar.get_height():.2f}×\nΔexcess {delta:+.02f} pt", ha="center", va="bottom", fontsize=8)
axes[2].set_ylim(0, max(savings) + 1.4)

fig.suptitle("Real MP-QGNN MVPs: mechanism, deployment, and measurement cost", fontsize=13, fontweight="bold")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180)
fig.savefig(OUT_SVG)
print(OUT)
