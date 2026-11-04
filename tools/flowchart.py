"""Study flowchart as an editable SVG (5 stages), exported to PNG and PDF.

Canvas units: 10 per mm, 180 mm wide, so 8 pt is 28 units. Plotly colours,
Arial, black text. Run: python3 tools/flowchart.py
Writes out/figures/flowchart.svg, .png, .pdf.
"""
import subprocess

W, H = 1800, 2470
INK, GREY, LIGHT = "#111111", "#9A9A9A", "#F4F4F4"
RED, GREEN, BLUE, PURPLE, ORANGE = "#EF553B", "#00CC96", "#636EFA", "#AB63FA", "#E3732B"
FONT = "Arial, Helvetica, sans-serif"
S = []
add = S.append
ARR = []


def text(x, y, s, size=28, weight="normal", anchor="start", fill=INK, italic=False):
    st = " font-style='italic'" if italic else ""
    add(f"<text x='{x}' y='{y}' font-family='{FONT}' font-size='{size}' font-weight='{weight}' text-anchor='{anchor}' fill='{fill}'{st}>{s}</text>")


def lines(x, y, rows, size=28, gap=36, weight="normal"):
    for i, r in enumerate(rows):
        text(x, y + i * gap, r, size, weight)
    return y + len(rows) * gap


def box(x, y, w, h, col, fill="white", rx=14, sw=3):
    add(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{col}' stroke-width='{sw}'/>")


def stage(y, h, num, title):
    add(f"<rect x='20' y='{y}' width='{W - 40}' height='{h}' rx='18' fill='{LIGHT}' stroke='none'/>")
    add(f"<circle cx='62' cy='{y + 44}' r='24' fill='{INK}'/>")
    text(62, y + 54, str(num), 30, "bold", "middle", "white")
    text(100, y + 55, title, 34, "bold")


def arrow_down(x, y1, y2, col=INK):
    ARR.append(f"<line x1='{x}' y1='{y1}' x2='{x}' y2='{y2 - 16}' stroke='{col}' stroke-width='5'/>")
    ARR.append(f"<path d='M{x - 14},{y2 - 18} L{x + 14},{y2 - 18} L{x},{y2} Z' fill='{col}'/>")


def arrow_right(x1, x2, y, col=INK):
    add(f"<line x1='{x1}' y1='{y}' x2='{x2 - 16}' y2='{y}' stroke='{col}' stroke-width='4'/>")
    add(f"<path d='M{x2 - 18},{y - 12} L{x2 - 18},{y + 12} L{x2},{y} Z' fill='{col}'/>")


def cells(x, y, cols, n=7, r=13, dx=30):
    pos = [(0, 0), (1, -0.5), (2, 0), (3, -0.5), (0.5, 1), (1.5, 0.6), (2.5, 1)]
    for i, (a, b) in enumerate(pos[:n]):
        c = cols[i % len(cols)]
        add(f"<circle cx='{x + a * dx}' cy='{y + b * dx}' r='{r}' fill='{c}' fill-opacity='0.85'/>")


def curve(x, y, w, h, kind, col):
    add(f"<line x1='{x}' y1='{y + h}' x2='{x + w}' y2='{y + h}' stroke='{INK}' stroke-width='2'/>")
    add(f"<line x1='{x}' y1='{y}' x2='{x}' y2='{y + h}' stroke='{INK}' stroke-width='2'/>")
    if kind == "convex":
        d = f"M{x},{y + h} C{x + w * 0.6},{y + h} {x + w * 0.85},{y + h * 0.7} {x + w},{y}"
    elif kind == "concave":
        d = f"M{x},{y + h} C{x + w * 0.15},{y + h * 0.3} {x + w * 0.4},{y} {x + w},{y}"
    else:
        d = f"M{x},{y + h} L{x + w},{y}"
    add(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='4'/>")


# ---------------------------------------------------------------- stage 1
y = 20
stage(y, 420, 1, "Data and control cells")
cells(70, y + 130, [BLUE, RED, GREEN, PURPLE, ORANGE, BLUE, GREEN])
lines(210, y + 125, ["CRISPRi Perturb-seq screens",
                     "K562 (essential and genome-wide), RPE1,",
                     "Jurkat, HepG2 (essential)"], 28, 36)
cells(70, y + 290, [GREY, BLUE, BLUE, GREY, BLUE, BLUE, BLUE])
lines(210, y + 285, ["Attenuated guide series",
                     "Jost et al. 2020, 25 genes in K562,",
                     "5 or 6 guides of graded activity per gene"], 28, 36)
add(f"<line x1='900' y1='{y + 95}' x2='900' y2='{y + 400}' stroke='{GREY}' stroke-width='2'/>")
text(940, y + 125, "Non-targeting control cells, split in two", 28, "bold")
cells(950, y + 190, [GREY])
box(1110, y + 160, 640, 100, INK, "white", 10, 2)
lines(1130, y + 200, ["Reference set: gene means and scales,",
                      "target expression model"], 28, 36)
box(1110, y + 285, 640, 100, INK, "white", 10, 2)
lines(1130, y + 325, ["Score set: null of cell scores,",
                      "control groups and simulations"], 28, 36)
arrow_right(1050, 1108, y + 210)
arrow_right(1050, 1108, y + 335)
add(f"<line x1='1050' y1='{y + 210}' x2='1050' y2='{y + 335}' stroke='{INK}' stroke-width='4'/>")
arrow_down(W / 2, y + 420, y + 470)

# ---------------------------------------------------------------- stage 2
y = 470
stage(y, 720, 2, "Two-step scoring and the cross-fitted joint model")
box(50, y + 90, 760, 600, RED, "white")
text(80, y + 140, "Two-step methods (Song score, Mixscale)", 30, "bold")
cells(85, y + 200, [RED], 4, 12, 26)
lines(220, y + 200, ["Signature estimated and applied",
                     "on the same cells"], 28, 36)
add(f"<g>")
for i, hgt in enumerate([40, 22, 52, 14, 34]):
    add(f"<rect x='{80 + i * 22}' y='{y + 340 - hgt}' width='16' height='{hgt}' fill='{RED}'/>")
add("</g>")
lines(220, y + 300, ["Dose read from binned",
                     "target UMI per cell"], 28, 36)
add(f"<line x1='80' y1='{y + 400}' x2='780' y2='{y + 400}' stroke='{GREY}' stroke-width='2'/>")
text(80, y + 450, "Control groups treated as perturbations", 28, "bold")
lines(80, y + 495, ["Effect reported in 100% of groups",
                    "Buffered or sensitive label in 1.5% to 3.2%",
                    "Median 266 to 928 DE genes (Song score)"], 28, 38)

box(850, y + 90, 900, 600, GREEN, "white")
text(880, y + 140, "Cross-fitted joint model (dosecurve)", 30, "bold")
# A
text(880, y + 195, "A", 30, "bold", fill=GREEN)
cells(925, y + 185, [GREEN], 2, 11, 24)
add(f"<line x1='985' y1='{y + 168}' x2='985' y2='{y + 215}' stroke='{INK}' stroke-width='2' stroke-dasharray='4 4'/>")
cells(1000, y + 185, [PURPLE], 2, 11, 24)
lines(1070, y + 195, ["Folds A and B: weights from one fold",
                      "score the other, effect test (q &lt; 0.05)"], 28, 36)
# B
text(880, y + 305, "B", 30, "bold", fill=GREEN)
lines(925, y + 305, ["Joint likelihood of target UMI (negative binomial)",
                     "and cell score (normal, response variation τ²)",
                     "linked by response = knockdown<tspan dy='-12' font-size='22'>γ</tspan><tspan dy='12'>, fitted by EM</tspan>"], 28, 36)
# C
text(880, y + 450, "C", 30, "bold", fill=GREEN)
lines(925, y + 450, ["Bootstrap under γ = 1 (20 replicates)"], 28, 36)
curve(930, y + 490, 90, 70, "convex", PURPLE)
text(1035, y + 535, "buffered, γ &gt; 1", 28)
curve(1270, y + 490, 90, 70, "concave", ORANGE)
text(1375, y + 535, "sensitive, γ &lt; 1", 28)
add(f"<line x1='880' y1='{y + 590}' x2='1720' y2='{y + 590}' stroke='{GREY}' stroke-width='2'/>")
text(880, y + 635, "Control groups: effect in 2.3% to 4.3%, no labels", 28, "bold")
arrow_down(W / 2, y + 720, y + 770)

# ---------------------------------------------------------------- stage 3
y = 1240
stage(y, 430, 3, "Simulations and the identifiable regime")
# mini heatmap
vals = [[0.05, 0.1, 0.2, 0.3], [0.1, 0.25, 0.55, 0.6], [0.3, 0.7, 0.9, 0.95]]
for r, row in enumerate(vals):
    for c, v in enumerate(row):
        op = 0.15 + 0.85 * v
        add(f"<rect x='{70 + c * 34}' y='{y + 110 + r * 34}' width='30' height='30' fill='{BLUE}' fill-opacity='{op:.2f}'/>")
lines(230, y + 125, ["Simulated genes from control cells",
                     "with known dose response, varying",
                     "knockdown distribution, response",
                     "variation and effect strength"], 28, 36)
lines(70, y + 300, ["False labels rise when most cells are",
                    "fully knocked down and responses vary",
                    "(up to 94%)"], 28, 36)
arrow_right(690, 760, y + 230)
box(770, y + 95, 470, 300, BLUE, "white")
text(795, y + 140, "Classifier on 7 features", 28, "bold")
lines(795, y + 176, [                     "dose spread, τ², mean knockdown,",
                     "target expression, cells,",
                     "effect size, DE genes",
                     "", "AUC 0.92 to 0.95"], 28, 36)
arrow_right(1250, 1310, y + 230)
box(1320, y + 95, 430, 300, GREEN, "white")
text(1345, y + 140, "Identifiable regime", 28, "bold")
lines(1345, y + 140 + 36, ["47% to 70% of perturbations",
                      "", "False labels",
                      "joint model 7% to 10%",
                      "two-step 23% to 35%"], 28, 36)
arrow_down(W / 2, y + 430, y + 480)

# ---------------------------------------------------------------- stage 4
y = 1720
stage(y, 380, 4, "Labels in four cell lines, identifiable regime")
for i, (b, sn, col) in enumerate([(0.29, 0.004, GREEN), (0.26, 0.15, RED)]):
    x0 = 70
    yy = y + 115 + i * 60
    add(f"<rect x='{x0}' y='{yy}' width='240' height='40' fill='#D3D3D3'/>")
    add(f"<rect x='{x0}' y='{yy}' width='{240 * b:.0f}' height='40' fill='{PURPLE}'/>")
    add(f"<rect x='{x0 + 240 * b:.0f}' y='{yy}' width='{max(240 * sn, 2):.0f}' height='40' fill='{ORANGE}'/>")
    text(x0 + 252, yy + 30, "joint" if i == 0 else "two-step", 26)
add(f"<rect x='460' y='{y + 120}' width='26' height='26' fill='{PURPLE}'/>")
text(496, y + 142, "buffered", 26)
add(f"<rect x='460' y='{y + 160}' width='26' height='26' fill='{ORANGE}'/>")
text(496, y + 182, "sensitive", 26)
lines(70, y + 270, ["Buffered 12% to 29%, sensitive ≤ 1%",
                    "Two-step: sensitive 6% to 15%"], 28, 36)
add(f"<line x1='620' y1='{y + 95}' x2='620' y2='{y + 350}' stroke='{GREY}' stroke-width='2'/>")
text(650, y + 125, "Target UMI test, K562", 28, "bold")
lines(650, y + 125 + 36, ["graded knockdown in 68%",
                     "of buffered genes"], 28, 36)
add(f"<line x1='1040' y1='{y + 95}' x2='1040' y2='{y + 350}' stroke='{GREY}' stroke-width='2'/>")
text(1070, y + 125, "Buffering relative to line baseline", 28, "bold")
lines(1070, y + 165, ["cytosolic ribosome 2.4 to 3.7×",
                      "RNA polymerase II 1.8 to 4.4×",
                      "proteasome 1.2 to 6.3×",
                      "mitochondrial ribosome 0.2 to 0.9×",
                      "same in TP53 mutant and wild-type lines"], 28, 36)
arrow_down(W / 2, y + 380, y + 430)

# ---------------------------------------------------------------- stage 5
y = 2150
stage(y, 300, 5, "Independent validation")
curve(70, y + 120, 110, 90, "convex", PURPLE)
for gx, gy in [(95, 205), (125, 196), (150, 175), (168, 140)]:
    add(f"<path d='M{gx},{y + gy - 9} L{gx + 9},{y + gy} L{gx},{y + gy + 9} L{gx - 9},{y + gy} Z' fill='{INK}'/>")
text(220, y + 125, "Attenuated guides (Jost, 25 genes)", 28, "bold")
lines(220, y + 125 + 36, ["buffered shape across guides: 15 genes",
                     "joint model: 15 of 15 buffered",
                     "Song score: 9 of 15 sensitive"], 28, 36)
add(f"<line x1='900' y1='{y + 95}' x2='900' y2='{y + 270}' stroke='{GREY}' stroke-width='2'/>")
cells(940, y + 130, [BLUE], 3, 11, 24)
cells(940, y + 205, [BLUE], 3, 11, 24)
text(1070, y + 125, "K562 essential vs genome-wide (279 genes)", 28, "bold")
lines(1070, y + 125 + 36, ["buffered labels reproduced 62% to 64%",
                      "sensitive labels reproduced",
                      "17% (Song score), 3% (Mixscale)"], 28, 36)

svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'><rect width='{W}' height='{H}' fill='white'/>" + "".join(S) + "".join(ARR) + "</svg>"
open("out/figures/flowchart.svg", "w").write(svg)
subprocess.run(["rsvg-convert", "-z", "2", "-o", "out/figures/flowchart.png", "out/figures/flowchart.svg"], check=True)
subprocess.run(["rsvg-convert", "-f", "pdf", "-o", "out/figures/flowchart.pdf", "out/figures/flowchart.svg"], check=True)
print("out/figures/flowchart.svg/.png/.pdf")
