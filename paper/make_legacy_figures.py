"""Development curves and unseen-panel breakdown for the journal version (copied from paper_continual/make_paper_figures.py; labels renamed to GA / GA+LwF). The trade-off figure it also writes is unused."""
"""Vector figures for the IEEE paper, sized for one column (3.5 in). All numbers are read from saved results.
Colours follow method identity (validated categorical slots; ours = slot 1) and match the project report."""

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = _os.path.join(_HERE, "..", "experiments", "runs")
OUT = _os.path.join(_HERE, "figures")
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
COL = {"realkd_lwf": "#2a78d6", "seq": "#eb6834", "lwf": "#1baf7a", "featkd": "#eda100", "realkd": "#e87ba4", "none": NEUTRAL}
NAME = {"realkd_lwf": "GA+LwF", "seq": "Sequential FT", "lwf": "LwF", "featkd": "Symmetric KD",
        "realkd": "GA", "none": "No update"}
from matplotlib import font_manager  # noqa: E402

for _f in ["regular", "bold", "italic", "bolditalic"]:  # TeX Gyre Termes: Times-compatible, matches IEEEtran body text
    font_manager.fontManager.addfont(f"{_HERE}/fonts/texgyretermes-{_f}.ttf")
plt.rcParams.update({"font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "legend.fontsize": 6.5})


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)


# ---- Fig: development probe curves (3 seeds)
dev = {"seq": "gate1_v1.1", "lwf": "gate2", "featkd": "gate2", "realkd": "e1", "realkd_lwf": "e2"}
fig, ax = plt.subplots(figsize=(3.5, 2.15))
ends = {}
for m in ["realkd_lwf", "realkd", "lwf", "featkd", "seq"]:
    c = np.array([[json.load(open(f"{R}/{dev[m]}/{m}_seed{s}/results.json"))["stages"][i]["dev"]["FSh_probe"]["auroc"] * 100
                   for i in range(4)] for s in range(3)])
    ax.plot(range(4), c.mean(0), color=COL[m], linewidth=1.3, marker="o", markersize=3, label=NAME[m], zorder=3)
    ax.fill_between(range(4), c.min(0), c.max(0), color=COL[m], alpha=0.10, linewidth=0)
    ends[m] = c.mean(0)[-1]
placed = []
for m, v in sorted(ends.items(), key=lambda kv: kv[1]):
    yl = v if not placed or v - placed[-1] >= 0.55 else placed[-1] + 0.55
    placed.append(yl)
    ax.annotate(f"{v:.1f}", (3, v), xytext=(3.08, yl), va="center", fontsize=6.5, color=INK2)
ax.set_xticks(range(4), ["Base (FF++)", "+DFDC-P", "+DFD", "+CDF"])
ax.set_xlim(-0.15, 3.4)
ax.set_ylabel("FaceShifter probe AUROC (%)")
ax.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.6)
style(ax)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig_dev_curves.pdf")
plt.close(fig)

# ---- Fig: C1 unseen panel per source
c1 = {m: json.load(open(f"{R}/c1/{m}_seed0/results.json")) for m in ["none", "realkd_lwf", "lwf", "seq", "realkd", "featkd"]}
sets = [("WDF_aligned", "WDF"), ("DFF_inpainting", "DFF-Inpaint"), ("DFF_insight", "DFF-Insight"), ("DFF_text2img", "DFF-T2I")]
ms = ["none", "seq", "lwf", "realkd_lwf"]
fig, ax = plt.subplots(figsize=(3.5, 2.05))
w, x = 0.2, np.arange(len(sets))
for i, m in enumerate(ms):
    v = [c1[m]["panel"][k]["auroc"] * 100 for k, _ in sets]
    ax.bar(x + (i - 1.5) * w, v, width=w - 0.02, color=COL[m], edgecolor="white", linewidth=0.6, label=NAME[m])
    for xx, vv in zip(x + (i - 1.5) * w, v):
        ax.text(xx, vv + 0.4, f"{vv:.1f}", ha="center", fontsize=4.6, color=INK2, rotation=90)
ax.set_xticks(x, [n for _, n in sets])
ax.set_ylim(60, 98)
ax.set_ylabel("AUROC (%)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.0, columnspacing=0.8)
style(ax)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig_unseen_breakdown.pdf")
plt.close(fig)

# ---- Fig: trade-off scatter
T = ["FF", "DFDCP", "DFD", "CDF"]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
offs = {"none": (5, -3), "realkd_lwf": (6, -2), "realkd": (-30, 5), "featkd": (5, 1), "lwf": (-14, -9), "seq": (5, -2)}
for m in ["none", "seq", "lwf", "featkd", "realkd", "realkd_lwf"]:
    a = np.mean([c1[m]["stages"][-1]["dev"][t]["auroc"] * 100 for t in T])
    u = c1[m]["panel"]["U"]["auroc"] * 100
    hi = m in ("realkd_lwf", "realkd")
    ax.scatter(a, u, s=26 if hi else 18, color=COL[m], edgecolor="white", linewidth=0.6, zorder=3)
    ax.annotate(NAME[m], (a, u), textcoords="offset points", xytext=offs[m], fontsize=6.5, color=INK if hi else INK2)
ax.set_xlabel("Mean test AUROC on trained datasets (%)")
ax.set_ylabel("Unseen-panel score $U$ (%)")
ax.set_xlim(93.0, 95.6)
style(ax, "both")
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig_tradeoff.pdf")
plt.close(fig)
print("ok")
