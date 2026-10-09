"""Build a plain-English PDF report for the three real MP-QGNN MVPs."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))


NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#4C78A8")
GREEN = colors.HexColor("#3E8E41")
ORANGE = colors.HexColor("#F58518")
RED = colors.HexColor("#C94C4C")
LIGHT = colors.HexColor("#F3F6F8")
MID = colors.HexColor("#D7E0E7")
TEXT = colors.HexColor("#24313A")


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="TitleCustom", parent=base["Title"], fontName="DejaVuSans-Bold",
        fontSize=25, leading=29, textColor=NAVY, alignment=TA_CENTER, spaceAfter=9,
    ))
    base.add(ParagraphStyle(
        name="Subtitle", parent=base["Normal"], fontName="DejaVuSans", fontSize=12, leading=16,
        textColor=colors.HexColor("#55636E"), alignment=TA_CENTER, spaceAfter=18,
    ))
    base.add(ParagraphStyle(
        name="H1Custom", parent=base["Heading1"], fontName="DejaVuSans-Bold",
        fontSize=17, leading=21, textColor=NAVY, spaceBefore=7, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="H2Custom", parent=base["Heading2"], fontName="DejaVuSans-Bold",
        fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=7, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="BodyCustom", parent=base["BodyText"], fontName="DejaVuSans",
        fontSize=9.4, leading=13.3, textColor=TEXT, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="Small", parent=base["BodyText"], fontName="DejaVuSans",
        fontSize=7.7, leading=10.5, textColor=colors.HexColor("#53616B"), spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="Callout", parent=base["BodyText"], fontName="DejaVuSans-Bold",
        fontSize=10.2, leading=14.2, textColor=NAVY, backColor=LIGHT,
        borderColor=MID, borderWidth=0.7, borderPadding=8, spaceBefore=5, spaceAfter=9,
    ))
    base.add(ParagraphStyle(
        name="VerdictGood", parent=base["BodyText"], fontName="DejaVuSans-Bold",
        fontSize=10, leading=13, textColor=GREEN,
    ))
    base.add(ParagraphStyle(
        name="VerdictStop", parent=base["BodyText"], fontName="DejaVuSans-Bold",
        fontSize=10, leading=13, textColor=RED,
    ))
    return base


def para(text, style):
    return Paragraph(text, style)


def bullet(text, st):
    return Paragraph(f"- {text}", st)


def table(data, widths, header=True, font_size=8.2):
    header_style = ParagraphStyle(
        "TableHeader", fontName="DejaVuSans-Bold", fontSize=font_size,
        leading=font_size + 1.8, textColor=colors.white,
    )
    cell_style = ParagraphStyle(
        "TableCell", fontName="DejaVuSans", fontSize=font_size,
        leading=font_size + 2.0, textColor=NAVY,
    )
    wrapped = []
    for row_index, row in enumerate(data):
        style = header_style if header and row_index == 0 else cell_style
        wrapped.append([
            cell if isinstance(cell, Paragraph) else Paragraph(html.escape(str(cell)), style)
            for cell in row
        ])
    t = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.4),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
        ]
    t.setStyle(TableStyle(commands))
    return t


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(MID)
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("DejaVuSans", 7.5)
    canvas.setFillColor(colors.HexColor("#6B7780"))
    canvas.drawString(18 * mm, 9 * mm, "MP-QGNN research radar | 9 October 2026")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build(results_path: Path, figures: Path, output: Path) -> None:
    data = json.loads(results_path.read_text())
    fourier = data["fourier"]
    shots = data["shots"]
    lie = data["lie"]
    st = styles()
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm,
        title="MP-QGNN Radar: Three Real MVP Experiments",
        author="Snehal Raj research workflow",
    )
    story = []

    story += [Spacer(1, 16 * mm), para("MP-QGNN Radar", st["TitleCustom"]),
              para("Three real-code MVP experiments and a decision about what can become a paper",
                   st["Subtitle"])]
    score_rows = [
        ["Rank", "Idea", "Paper", "Novelty", "Impact", "Decision"],
        ["1", "Finite-shot WL expressivity", "8/10", "8/10", "8/10", "Main project"],
        ["2", "Two-body phase extension", "7/10", "6/10", "7/10", "Parallel track"],
        ["3", "Fourier edge-angle map", "3/10", "5/10", "4/10", "Stop as main story"],
    ]
    story += [table(score_rows, [11*mm, 51*mm, 18*mm, 20*mm, 18*mm, 42*mm], font_size=8.0),
              Spacer(1, 7 * mm),
              para("Bottom line", st["H1Custom"]),
              para("The strongest paper is not the Fourier extension. The strongest result is that the current noiseless CFI separation is carried by a bounded signal near one part in a billion. A second strong direction is to add one shared two-body phase, which expands the actual D=6, k=3 compound layer from a 15-dimensional hopping algebra to the full 400-dimensional fixed-sector algebra.", st["Callout"]),
              para("This report uses the real public <b>mp-qgnns</b> implementation. It does not use the earlier synthetic trigonometric oracle.", st["BodyCustom"]),
              para("How to read the scores", st["H2Custom"]),
              bullet("Paper score: probability that focused work can become a defensible standalone paper.", st["BodyCustom"]),
              bullet("Novelty score: distance from known Fourier, finite-shot, and fermionic-control results.", st["BodyCustom"]),
              bullet("Impact score: breadth of nearby work that would need to consider the result. It is not a citation forecast.", st["BodyCustom"]),
              PageBreak()]

    story += [para("1. Neural Fourier edge-angle map", st["H1Custom"]),
              para("Idea", st["H2Custom"]),
              para("Replace the small classical edge-angle MLP with a learned Fourier map. Keep the graph ordering, RBS gates, compound layer, and quantum deployment path unchanged. The Fourier map has 80 parameters per layer. The original MLP has 81.", st["BodyCustom"]),
              para("Success test", st["H2Custom"]),
              para("Train on five cities. Do not lose five-city accuracy. Improve both six-city and seven-city transfer by at least one percentage point in tour ratio across five seeds.", st["BodyCustom"]),
              Image(str(figures / "fourier_transfer.png"), width=170*mm, height=102*mm),
              para("What the plot means", st["H2Custom"]),
              para("Zero on the vertical axis is an optimal tour. At seven cities, the original model is 2.20% above optimum and the Fourier model is 2.64% above optimum. The Fourier model gives a tiny gain at six cities and loses it at seven cities. The error bars also overlap.", st["BodyCustom"]),
              para("Verdict: threshold failed", st["VerdictStop"]),
              para("This can remain a dequantization control, but it is not yet a performance paper. The source direction is credible because all Neural Fourier Surrogates authors are at IonQ and the work is framed around the advantage boundary. Our graph extension still needs a frozen-circuit spectral-surrogate test before it supports a dequantization claim.", st["BodyCustom"]),
              PageBreak()]

    req = shots["median_j3_shots_for_5sigma"]
    story += [para("2. Operational WL expressivity under finite shots", st["H1Custom"]),
              para("Idea", st["H2Custom"]),
              para("The model separates CFI(K3) at the correct WL level with exact float64 embeddings. A device sees measurement samples, not exact embeddings. We therefore ask whether the separating direction is large enough to measure.", st["BodyCustom"]),
              para("Success test", st["H2Custom"]),
              para("Call the gap material if an optimistic model still needs more than 10^12 shots per bounded feature for a five-sigma distinction. Later, a repaired architecture should reduce the requirement by at least 10,000 times.", st["BodyCustom"]),
              Image(str(figures / "cfi_shot_curve.png"), width=170*mm, height=104*mm),
              para("What the plot means", st["H2Custom"]),
              para(f"Fifty percent is random guessing. Even at 10^16 shots per feature, the optimistic mean accuracy is only about 53%. Across five model seeds, the median five-sigma estimate is {req:.2e} shots per bounded feature.", st["BodyCustom"]),
              para("Why this is only a diagnostic", st["H2Custom"]),
              para("The calculation assumes the best separating direction is already known, independent Pauli-mean noise, and no device noise. A compiled observable set, grouping, shadows, or a trained readout may change the scale. Because the assumptions are optimistic, the result still exposes a real weakness in the current frozen-readout experiment.", st["BodyCustom"]),
              para("Verdict: strongest paper direction", st["VerdictGood"]),
              para("The paper question is simple: WL expressivity tells us that two graphs can be separated, but does it tell us that they can be separated with a realistic number of measurements? This connects the MP-QGNN result to current work on finite-shot learning, shadows, permutation-equivariant simulation, and dequantization.", st["BodyCustom"]),
              PageBreak()]

    actual = next(r for r in lie["rows"] if r["D"] == 6 and r["k"] == 3)
    story += [para("3. One two-body phase beyond the compound layer", st["H1Custom"]),
              para("Idea", st["H2Custom"]),
              para("Insert one shared occupation phase exp(i phi n0 n1) between compound layers. It is non-Gaussian, preserves particle number, and is shared across graph messages, so graph-node permutation equivariance is unchanged.", st["BodyCustom"]),
              para("Success test", st["H2Custom"]),
              para("Extra hopping must not enlarge the algebra. A one-body phase must remain restricted. One two-body phase must reach the full unitary algebra in the fixed-particle sector.", st["BodyCustom"]),
              Image(str(figures / "lie_algebra_fraction.png"), width=170*mm, height=106*mm),
              para("What the plot means", st["H2Custom"]),
              para(f"For the actual D=6, k=3 layer, the state dimension is 20. The full unitary algebra therefore has 400 real directions. Existing hopping gives {actual['path_lie_dimension']}. Long-range hopping still gives {actual['path_plus_long_range_dimension']}. A one-body phase gives {actual['path_plus_one_body_phase_dimension']}. One two-body phase gives all {actual['path_plus_two_body_phase_dimension']}.", st["BodyCustom"]),
              para("Verdict: algebra passed, learning claim not yet tested", st["VerdictGood"]),
              para("The algebraic change is clean, but related universality of fermionic linear optics plus non-Gaussian resources is known. A paper needs the graph-learning consequence: larger tangent rank, a larger measurable CFI margin, or better QM9/TSP performance at matched depth and parameter count.", st["BodyCustom"]),
              PageBreak()]

    story += [para("Community and quality signals", st["H1Custom"]),
              para("Affiliation is a useful prior, not evidence by itself", st["H2Custom"]),
              para("The Fourier source is entirely from IonQ. That is a positive signal for hardware relevance and the advantage question. The finite-shot paper comes from Palermo, Milan, and Queen's University Belfast and is stronger because it gives a detailed bias-variance theory, not because of a company name. The closest permutation-equivariant simulation paper is from Los Alamos, which is a strong signal for the importance of the dequantization boundary.", st["BodyCustom"]),
              para("Robert Huang connection", st["H2Custom"]),
              para("I interpret 'Robert' as Hsin-Yuan (Robert) Huang. The Los Alamos equivariant-simulation paper cites Huang's work on classical observable estimation and classical shadows. The finite-shot paper cites Huang, Kueng, and Preskill on shadows. The IonQ Fourier paper does not appear to cite Huang by name in its arXiv HTML, although it is explicitly motivated by the quantum-advantage boundary.", st["BodyCustom"]),
              para("Best collaborator targets", st["H2Custom"]),
              table([
                  ["Project", "People", "Specific reason to contact"],
                  ["Finite-shot WL", "Su Yeon Chang, Martin Larocca, M. Cerezo", "Equivariant simulation and classical observable baselines"],
                  ["Finite-shot WL", "Gabriele Lo Monaco, Salvatore Lorenzo, Luca Innocenti", "Shot-noise geometry and regularized readouts"],
                  ["Two-body phase", "Jakob Kottmann, Adelina Barligea", "Bounded-Hamming-weight Lie simulation"],
                  ["Two-body phase", "Sahinur Reja", "Minimal connectivity and accessibility"],
                  ["Fourier control", "Oliver Knitter, Martin Roetteler", "Support-matched surrogate and hardware comparison"],
              ], [33*mm, 56*mm, 76*mm], font_size=7.7),
              Spacer(1, 5*mm),
              para("Do not email yet", st["VerdictStop"]),
              para("First produce either an exact measurement model for the CFI readout or one learning result from the two-body phase layer. Then send a two-page note with one result and one specific question.", st["BodyCustom"]),
              PageBreak()]

    story += [para("Next two-week plan", st["H1Custom"]),
              table([
                  ["Priority", "Experiment", "Go metric", "Stop metric"],
                  ["1", "Compile the exact CFI observables and compare direct groups, shadows, and an oracle lower bound.", "A defensible shot curve with explicit circuit calls.", "Readout is not physically specified enough to compile."],
                  ["1", "Train encoders and mixing angles with normalized-margin and shot-aware losses.", ">=10^4 reduction in inferred shots while retaining WL threshold.", "Less than 100x after matched tuning."],
                  ["2", "Add one complex two-body phase to the D=6, k=3 compound layer.", "CFI margin or task error improves at matched depth and parameters.", "Only algebra grows; gradients or tasks degrade."],
                  ["3", "Frozen-circuit Fourier surrogate as a dequantization control.", ">=95% predictive fidelity and materially cheaper inference.", "No cost gain or poor spectral fidelity."],
              ], [15*mm, 66*mm, 47*mm, 38*mm], font_size=7.4),
              Spacer(1, 6*mm),
              para("Recommended paper story", st["H2Custom"]),
              para("Lead with <b>operational expressivity</b>: exact WL separation is not enough when the class margin is smaller than measurement noise. Use the two-body phase as the proposed repair. This creates one coherent paper instead of three unrelated notes: diagnose the finite-shot gap, prove how a minimal generator changes the model class, and test whether the change produces a measurable graph signal.", st["Callout"]),
              para("Reproducibility", st["H2Custom"]),
              para("Baseline repository: 43 tests passed before changes. Final branch: 48 tests passed. Main experiment: five seeds, real TSP and CFI code paths, deterministic dataset seeds, and a strict JSON results file. The finite-shot calculation remains an optimistic measurement proxy and the Lie result remains numerical evidence, not a theorem.", st["BodyCustom"]),
              para("Primary sources", st["H2Custom"]),
              para("Neural Fourier Surrogates: https://arxiv.org/abs/2610.00841<br/>Who can sample forever?: https://arxiv.org/abs/2610.12015<br/>Connectivity Controls Variational Accessibility: https://arxiv.org/abs/2610.07173<br/>Practical permutation-equivariant simulation: https://arxiv.org/abs/2603.13072<br/>Lie-algebraic simulation beyond free fermions: https://arxiv.org/abs/2604.16701<br/>Trainability and dequantization: https://arxiv.org/abs/2406.07072", st["Small"])]

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("figures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.results, args.figures, args.output)


if __name__ == "__main__":
    main()
