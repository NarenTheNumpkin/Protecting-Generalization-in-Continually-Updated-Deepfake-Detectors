"""Figures for J3: per-seed unseen scores (block S) and stage-wise unseen score over the nine-update stream (block L).
Reads radial/runs/j3_summary.json and radial/analysis_j3/stagewise_long.json."""

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _f in ["regular", "bold", "italic", "bolditalic"]:
    font_manager.fontManager.addfont(f"{_HERE}/fonts/texgyretermes-{_f}.ttf")
RAD = _os.path.join(_HERE, "..", "experiments")
OUT = _os.path.join(_HERE, "figures")
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc7"
COL = {"realkd_lwf": BLUE, "seq": ORANGE, "lwf": AQUA, "featkd_lwf": YELLOW, "realkd": MAGENTA, "gpl": VIOLET, "featkd": "#b8b6b0", "none": NEUTRAL}
NAME = {"none": "No update", "seq": "Seq. FT", "lwf": "LwF", "featkd": "Sym. KD", "realkd": "GA", "realkd_lwf": "GA+LwF",
        "featkd_lwf": "Sym. KD+LwF", "gpl": "GPL (re-impl.)"}
plt.rcParams.update({"font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 7.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "legend.fontsize": 6.5})


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


S = json.load(open(f"{RAD}/runs/j3_summary.json"))
U0 = 87.97

# ---- per-seed unseen score
order = ["seq", "gpl", "featkd", "lwf", "realkd", "featkd_lwf", "realkd_lwf"]
fig, ax = plt.subplots(figsize=(3.5, 2.0))
for i, m in enumerate(order):
    us = [v["U"] for v in S["S"]["per_seed"][m].values()]
    jit = np.linspace(-0.12, 0.12, len(us))
    hi = m == "realkd_lwf"
    ax.scatter(np.array(us), i + jit, s=14 if hi else 11, color=COL[m], edgecolor="white", linewidth=0.4, zorder=3)
    mu = np.mean(us)
    ax.plot([mu, mu], [i - 0.28, i + 0.28], color=INK, linewidth=1.0, zorder=4)
    ax.annotate(f"{mu:.2f}$\\pm${np.std(us, ddof=1):.2f}", (max(us), i), textcoords="offset points", xytext=(5, -2.5), fontsize=5.8, color=INK2)
ax.axvline(U0, color=NEUTRAL, linewidth=0.8, linestyle="--")
ax.text(U0 + 0.05, len(order) - 0.45, "no update", fontsize=5.8, color=INK2)
ax.set_yticks(range(len(order)), [NAME[m] for m in order])
ax.set_xlim(83.6, 89.4)
ax.set_xlabel("Unseen-panel score $U$ (%), five seeds; bar: mean")
style(ax, "x")
save(fig, "fig_seeds")

# ---- stage-wise U along the nine-update stream
p = f"{RAD}/analysis_j3/stagewise_long.json"
if os.path.exists(p):
    sw = json.load(open(p))
    toks = sw["tokens"]
    labs = ["Base"] + [t.replace("DFDCP#", "P").replace("DFD#", "D").replace("CDF", "C") for t in toks[1:]]
    u0 = sw["U"]["none"]["FF"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for m in ["seq", "gpl", "lwf", "featkd_lwf", "realkd", "realkd_lwf"]:
        ys = [u0] + [sw["U"][m][t] for t in toks[1:]]
        hi = m == "realkd_lwf"
        ax.plot(range(len(ys)), ys, color=COL[m], linewidth=1.5 if hi else 1.0, marker="o", markersize=2.8 if hi else 2.2,
                label=NAME[m], zorder=4 if hi else 3)
        ax.annotate(f"{ys[-1]:.2f}", (len(ys) - 1, ys[-1]), textcoords="offset points", xytext=(4, -2), fontsize=5.6, color=INK2)
    ax.axhline(u0, color=NEUTRAL, linewidth=0.7, linestyle="--")
    for x in (4.5, 8.5):
        ax.axvline(x, color=GRID, linewidth=0.6)
    ax.set_xticks(range(len(labs)), labs)
    ax.set_xlim(-0.3, len(labs) - 0.2)
    ax.set_xlabel("After update (P: DFDC-P session, D: DFD session, C: Celeb-DF)")
    ax.set_ylabel("$U$ (%)")
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=6)
    style(ax)
    save(fig, "fig_long")
