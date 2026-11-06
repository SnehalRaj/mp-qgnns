"""Build the visual PDF research note for the 2026-11-06 radar run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "radar_2026_11_06"
FIG = DATA / "figures"
OUT = ROOT / "output" / "pdf" / "quantum_ml_radar_2026_11_06.pdf"
FIG.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.loads((DATA / name).read_text())


noise, depth, conditional = load("noise.json"), load("depth.json"), load("conditional.json")


def save_experiment_chart():
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.4))
    blue, orange, red = "#315B7D", "#E29D45", "#B4473D"
    nsum = noise["summary"]
    ax[0].bar(["Clean-trained", "Noise-selected"],
              [nsum["median_baseline_noisy_gap_percent"], nsum["median_selected_noisy_gap_percent"]],
              color=[blue, red])
    ax[0].set_title("Residual-noise selection")
    ax[0].set_ylabel("Tour gap (%) - lower is better")
    ax[0].text(0.5, .96, "STOP", transform=ax[0].transAxes, ha="center", va="top",
               color=red, fontweight="bold")
    dmed = depth["summary"]["median_ratio_by_depth"]
    vals = [100 * (dmed[str(d)] - 1) for d in [1, 2, 3]]
    ax[1].bar(["1", "2", "3"], vals, color=[blue, orange, red])
    ax[1].set_title("Depth sweep")
    ax[1].set_xlabel("Message-passing depth")
    ax[1].text(.5, .96, "QFIM chose 3; task chose 2", transform=ax[1].transAxes,
               ha="center", va="top", fontsize=8)
    cmed = conditional["summary"]["median_unseen_gap_percent"]
    labels = ["5-only", "Joint", "Raw n", "2/n"]
    keys = ["tsp5_transfer", "joint_none", "joint_raw", "joint_normalized"]
    ax[2].bar(labels, [cmed[k] for k in keys], color=["#8A95A5", blue, red, orange])
    ax[2].set_title("Deploy on unseen TSP-8")
    ax[2].tick_params(axis="x", labelrotation=25)
    ax[2].text(.5, .96, "Size tokens did not help", transform=ax[2].transAxes,
               ha="center", va="top", fontsize=8)
    for a in ax:
        a.grid(axis="y", alpha=.22)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIG / "experiment_summary.png"
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return path


def save_scores_chart():
    names = ["Conditional IQP", "Data-informed ansatz", "GENIE noise", "QRC layout"]
    paper = [4.0, 3.5, 3.0, 6.5]
    novelty = [6.5, 6.0, 5.5, 5.5]
    impact = [7.0, 6.0, 6.0, 7.0]
    y = np.arange(len(names)); h = .22
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    ax.barh(y+h, paper, h, label="Paper potential after MVP", color="#315B7D")
    ax.barh(y, novelty, h, label="Novelty", color="#E29D45")
    ax.barh(y-h, impact, h, label="Likely research reach", color="#6C8F73")
    ax.set_yticks(y, names); ax.set_xlim(0, 10); ax.set_xlabel("Score (0-10)")
    ax.grid(axis="x", alpha=.22); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.invert_yaxis(); fig.tight_layout()
    path = FIG / "idea_scores.png"
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return path


exp_chart, score_chart = save_experiment_chart(), save_scores_chart()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=23, leading=27, textColor=colors.HexColor("#17324D"),
                          alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"], fontSize=11, leading=15,
                          textColor=colors.HexColor("#506171"), alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
                          fontSize=15, leading=18, textColor=colors.HexColor("#17324D"),
                          spaceBefore=8, spaceAfter=7))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11.5, leading=14, textColor=colors.HexColor("#315B7D"),
                          spaceBefore=6, spaceAfter=4))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.3, leading=13,
                          spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.8, leading=10.5,
                          textColor=colors.HexColor("#465461")))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#71808C"))
    canvas.drawString(18*mm, 10*mm, "Quantum ML Radar - MP-QGNN MVP note")
    canvas.drawRightString(192*mm, 10*mm, f"6 Nov 2026  |  {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
                        topMargin=16*mm, bottomMargin=16*mm)
story = [P("Quantum ML Radar", "Title2"),
         P("High-signal papers from 28 October-6 November 2026, plus three real MP-QGNN feasibility tests", "Deck"),
         P("Bottom line", "H1x"),
         P("No new paper strengthened the Weisfeiler-Leman expressivity story this week. The strongest ideas were about conditional train-classical/deploy-quantum models, choosing circuit depth before training, controlled residual noise, and hardware-aware layout selection. We tested the first three in the real <b>SnehalRaj/mp-qgnns</b> TSP path. All three missed their pre-set go metric after a harder rerun."),
         P("Tour gap is easy to read", "H2x"),
         P("Tour gap = 100 x (decoded tour length / exact optimum - 1). A 0% gap is optimal. A 5% gap means the route is 5% longer than the best possible route. Lower is better."),
         Image(str(exp_chart), width=176*mm, height=52*mm),
         Spacer(1, 3*mm),
         P("The initial beam-search pilot was too easy: most gaps were below 1%. We kept those raw files, then repeated every project with greedy decoding and larger graphs. This avoided presenting saturated numbers as progress.", "Smallx"),
         PageBreak(),
         P("Ranked research shortlist", "H1x")]

rows = [[P("Rank", "Smallx"), P("Paper and evidence", "Smallx"), P("Connection and judgment", "Smallx")],
 [P("1", "Bodyx"), P("<b>Classical training of conditional IQP Born machines</b> - 3 Nov 2026<br/>One shared IQP unitary, condition in the input state; 16-qubit held-out-temperature tests, classical training to 2,048 qubits, and 16/30-qubit IQM deployment. First-version preprint.", "Smallx"), P("Direct match to shared-parameter, cross-size deployment. <b>READ</b>. Our explicit size token failed on TSP-8, but ordinary joint size training helped.", "Smallx")],
 [P("2", "Bodyx"), P("<b>Data informed Ansatz Selection for VQCs</b> - 3 Nov 2026<br/>Fourier rank selects entanglement; QFIM rank growth selects depth; full Z-product readout resolves more state directions. Madelon only; no public code found.", "Smallx"), P("Could replace depth sweeps. <b>READ</b>, but our empirical QFIM did not predict the task-optimal MP-QGNN depth.", "Smallx")],
 [P("3", "Bodyx"), P("<b>Noise-regularized QCBM training under a finite shot budget</b> - 4 Nov 2026<br/>GENIE selects residual noise with capped mitigation and a held-out set; 144 simulated settings, no hardware evidence or public code found.", "Smallx"), P("Useful fixed-budget template. <b>READ</b> for theory; our RBS over-rotation implementation worsened TSP-6.", "Smallx")],
 [P("4", "Bodyx"), P("<b>Hardware-aware QRC layout selection</b> - 2 Nov 2026<br/>Six-qubit reservoir on IBM Kingston; 63 evaluations in 21 paired rounds. Hardware-aware placement beat Qiskit placement in every round; public code/data/records.", "Smallx"), P("Best hardware-realism paper this week. <b>READ</b>. Do not port until a real MP-QGNN QPU allocation is available.", "Smallx")]]
t = Table(rows, colWidths=[13*mm, 84*mm, 79*mm], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCE7EF")),
                       ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#A9B7C2")),
                       ("VALIGN", (0,0), (-1,-1), "TOP"),
                       ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
                       ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
story += [t, Spacer(1, 4*mm), P("Quality signal", "H2x"),
          P("The week has no LANL, Google, IBM Research, or Quantinuum-led architecture result. The strongest community signals are the direct train-classical/deploy-quantum lineage and IQM execution in the conditional-IQP paper, plus fully logged IBM Kingston runs in the QRC paper. These are positive evidence signals, not substitutes for task-matched baselines or proofs."),
          Image(str(score_chart), width=168*mm, height=65*mm),
          P("Scores are judgment calls after the MVPs: 0 means no credible path; 10 means unusually strong. 'Research reach' estimates how broadly the idea could change follow-on work, not a citation forecast.", "Smallx"),
          PageBreak(), P("What we actually ran", "H1x")]

details = [
 ("1. Residual-noise selection - STOP",
  "Mechanism: Gaussian over-rotation on every graph-conditioned SingleExcitation angle during training; validation selects 0, 0.08, 0.16, or 0.24 rad; evaluation uses 512 Bernoulli shots and 0.25 rad noise. Revised benchmark: exact TSP-6, 50 train / 40 validation / 50 test, three seeds, 100 epochs.",
  "Result: clean-trained median noisy gap 3.94%; selected-noise median 4.62%. Every seed worsened. Clean performance did not explain the failure (median clean-gap change -0.20 points)."),
 ("2. Data-informed depth - STOP",
  "Mechanism: compute a pure-state QFIM from normalized node states with respect to compound-layer parameters before training; compare depths 1, 2, and 3. Revised benchmark: exact TSP-6, 60 train / 50 test, three seeds, 100 epochs.",
  "Result: raw ranks were 15, 30, 45, so the first rule simply tracked parameter count. Entropy-effective ranks were 9.72, 13.18, 15.65 and still chose depth 3. Depth 2 had the best median gap: 2.87% versus 3.66% for depth 3, with 7,145 versus 8,733 parameters."),
 ("3. Conditional cross-size training - STOP / narrow PIVOT",
  "Mechanism: share all learned modules across TSP-5 and TSP-6; compare no size token, raw n, and bounded 2/n; deploy unchanged parameters on unseen TSP-8. Revised benchmark: 60 training graphs per seen size, 25 exact TSP-8 tests, three seeds, 100 epochs.",
  "Result: median unseen gap was 10.86% for TSP-5-only transfer, 7.08% for unconditioned joint training, 8.97% for raw n, and 8.10% for 2/n. The bounded token beat the stronger baseline in only one seed. Keep joint multi-size training; drop the token."),
]
for title, mech, result in details:
    story += [P(title, "H2x"), P(mech), P(result)]

story += [Spacer(1, 2*mm), P("Reproducibility", "H2x"),
          P("Full tests passed before and after: 43 passed. Raw JSON records include seeds, exact sample counts, loss traces, QFIM spectra, parameter counts, shot budget, selected settings, caveats, commands, and git revision. Pilot JSON files are preserved."),
          P("Commands", "H2x"),
          P("python -m pytest -q<br/>python experiments/radar_2026_11_06.py noise --out results/radar_2026_11_06/noise.json<br/>python experiments/radar_2026_11_06.py depth --out results/radar_2026_11_06/depth.json<br/>python experiments/radar_2026_11_06.py conditional --out results/radar_2026_11_06/conditional.json", "Smallx"),
          PageBreak(), P("Decision and next move", "H1x"),
          P("What not to do", "H2x"),
          P("Do not write a paper around residual noise, empirical QFIM depth selection, or an explicit graph-size token from these results. Each failed its own threshold, and none has hardware evidence."),
          P("Most actionable direction", "H2x"),
          P("Use <b>unconditioned joint training across graph sizes</b> as the new baseline, then ask a sharper theorem question: under what normalization and parameter-sharing conditions does the same equivariant RBS message map remain stable as n changes? The next experiment should compare joint-size training with a size-uniform bound or kernel-stability diagnostic, not add another free size feature."),
          P("Collaborators", "H2x"),
          P("No outreach yet. The current results are clean negative screens, not a result worth asking an external group to validate. If joint-size stability later clears a go metric, the conditional-IQP authors are the most relevant group to ask about conditioning versus extrapolation; if a QPU layout probe clears, the QRC authors have the most concrete paired-layout protocol."),
          P("Primary references", "H2x"),
          P("[1] arXiv:2611.05483, 3 Nov 2026. [2] arXiv:2611.05079, 3 Nov 2026. [3] arXiv:2611.06630, 4 Nov 2026. [4] arXiv:2611.03891, 2 Nov 2026. QRC code: github.com/rusnakrenata/qrc_insurance; records: doi.org/10.5281/zenodo.23862967.", "Smallx")]

doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(OUT)
