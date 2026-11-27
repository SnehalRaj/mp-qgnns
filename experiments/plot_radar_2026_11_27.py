"""Make reader-facing plots from the raw 27 Nov radar results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "radar_2026_11_27"
OUT = ROOT / "docs" / "research_notes" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"baseline": "#64748b", "revision": "#2563eb", "control": "#f59e0b"}


def load(name):
    return json.loads((RESULTS / name).read_text())


def paired(ax, base, other, labels, ylabel, title):
    x = np.arange(len(base))
    for i in x:
        ax.plot([i, i], [base[i], other[i]], color="#cbd5e1", lw=1.2, zorder=1)
    ax.scatter(x - .08, base, color=COLORS["baseline"], label=labels[0], zorder=2)
    ax.scatter(x + .08, other, color=COLORS["revision"], label=labels[1], zorder=2)
    ax.set_xticks(x, [f"seed {i}" for i in x])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False)


f = load("fermionic.json")
q = load("qpat.json")
c = load("contractive.json")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

inf = [100 * r["median_test_slater_infidelity"] for r in f["records"]]
pen = [r["projection_penalty_points"] for r in f["records"]]
axes[0].scatter(inf, pen, s=55, color=COLORS["revision"])
axes[0].axhspan(-.5, .5, color="#dcfce7", alpha=.7, label="within 0.5 points")
axes[0].axhline(0, color="#475569", lw=1)
axes[0].set(xlabel="Distance from best Slater state (%)",
            ylabel="Projected minus full TSP gap (points)")
axes[0].set_title("A. State distance ≠ task advantage", loc="left", fontweight="bold")
axes[0].legend(frameon=False, fontsize=8)

def vals(payload, method, key):
    return [next(r[key] for r in payload["records"] if r["seed"] == s and r["method"] == method)
            for s in payload["seeds"]]

paired(axes[1], vals(q, "classical_baseline", "gap_transfer_n6_percent"),
       vals(q, "distilled_trained_mpqgnn", "gap_transfer_n6_percent"),
       ["classical", "trained-teacher distill"], "TSP-6 optimality gap (%)",
       "B. Distillation gain lacks causality")
rand = np.median(vals(q, "distilled_random_mpqgnn", "gap_transfer_n6_percent"))
axes[1].axhline(rand, ls="--", color=COLORS["control"], label="random-teacher median")
axes[1].legend(frameon=False, fontsize=8)

paired(axes[2], vals(c, "baseline", "gap_transfer_n7_percent"),
       vals(c, "collective_near_identity_init", "gap_transfer_n7_percent"),
       ["MP-QGNN", "+ collective hopping"], "TSP-7 optimality gap (%)",
       "C. Symmetry-compatible layer does not transfer")

fig.suptitle("27 Nov 2026 MP-QGNN MVPs — lower TSP gap is better", fontsize=14, fontweight="bold")
fig.savefig(OUT / "radar_2026_11_27_mvp_results.png", dpi=180)
fig.savefig(OUT / "radar_2026_11_27_mvp_results.pdf")

