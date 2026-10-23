"""Plot the 23 October MP-QGNN radar results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
robust = json.loads((ROOT / "results/radar_2026_10_23_robustness.json").read_text())

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# A. Correct versus mismatched size query.
rows = robust["query_alignment"]["rows"]
sizes = [6, 7, 8]
for key, label, color, marker in [
    ("baseline_excess", "No size query", "#4c78a8", "o"),
    ("correct_query_excess", "Correct size query", "#f58518", "s"),
    ("mismatched_query_excess", "Fixed query = 5", "#54a24b", "^"),
]:
    means, errors = [], []
    for n in sizes:
        vals = np.array([r[key] for r in rows if r["n_cities"] == n]) * 100
        means.append(vals.mean())
        errors.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    axes[0].errorbar(sizes, means, yerr=errors, label=label, color=color, marker=marker, capsize=3)
axes[0].set_title("A. Size query fails alignment test")
axes[0].set_xlabel("Unseen number of cities")
axes[0].set_ylabel("Tour longer than optimum (%)")
axes[0].legend(frameon=False, fontsize=8)

# B. Structured margin correlations and bootstrap intervals.
margin = robust["margin_target_size"]["analysis"]["structured_q10_margin"]
rhos = [margin[f"n{n}"]["rho"] for n in sizes]
lo = [margin[f"n{n}"]["bootstrap_95pct"][0] for n in sizes]
hi = [margin[f"n{n}"]["bootstrap_95pct"][1] for n in sizes]
axes[1].errorbar(sizes, rhos, yerr=[np.array(rhos) - lo, np.array(hi) - rhos],
                 marker="o", color="#e45756", capsize=4)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].axhline(-0.6, color="gray", linewidth=0.8, linestyle="--")
axes[1].set_ylim(-1.02, 1.02)
axes[1].set_title("B. Margin does not transfer")
axes[1].set_xlabel("Unseen number of cities")
axes[1].set_ylabel("Spearman correlation\n(more margin should mean less excess)")

# C. Shot saving and task loss.
pareto = robust["readout_budget_curve"]["analysis"]["pareto"]
budgets = np.array([r["budget_per_edge"] for r in pareto])
reductions = np.array([r["shot_reduction"] for r in pareto])
added = np.array([r["added_excess"] for r in pareto]) * 100
ax = axes[2]
ax.plot(budgets, reductions, marker="o", color="#4c78a8", label="Shot saving")
ax.set_xscale("log", base=2)
ax.set_xlabel("Maximum shots per edge")
ax.set_ylabel("Fixed shots / adaptive shots", color="#4c78a8")
ax.tick_params(axis="y", labelcolor="#4c78a8")
ax.axhline(4, color="#4c78a8", linewidth=0.8, linestyle="--")
ax2 = ax.twinx()
ax2.plot(budgets, added, marker="s", color="#f58518", label="Added tour excess")
ax2.set_ylabel("Added tour excess (percentage points)", color="#f58518")
ax2.tick_params(axis="y", labelcolor="#f58518")
ax2.axhline(1.0, color="#f58518", linewidth=0.8, linestyle="--")
ax.set_title("C. Adaptive readout is useful")

fig.suptitle("MP-QGNN radar MVPs, 23 October 2026", fontsize=14)
fig.tight_layout()
out = ROOT / "figures/radar_2026_10_23_summary.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
