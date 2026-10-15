"""All data figures of the paper, drawn in the figures4papers house style (https://github.com/ChenLiu-1996/figures4papers,
scientific-figure-making conventions): semantic blue/green/red/neutral palette, Helvetica-like sans font, no top/right
spines, frameless legends, thick lines, black-edged bars, no grid, explicit y-limits, 300-dpi export.

Figures are drawn at twice their printed size with the same aspect ratio as before (so page layouts do not change) and
scaled to the column width in LaTeX. Every number is read from the saved results.
Usage: python make_figures.py [<experiments_dir> <out_dir>]; fig_examples also needs the face crops of the unseen panel."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
RAD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "..", "experiments")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "figures")
os.makedirs(OUT, exist_ok=True)
SCALE = 2.0  # drawn at 2x, printed at 1x

# ------------------------------------------------------------------ figures4papers palette and semantic roles
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_1": "#DDF3DE", "green_2": "#AADCA9",
           "green_3": "#8BCF8B", "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342", "neutral": "#CFCECE",
           "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E", "grey_mid": "#767676", "grey_dark": "#4D4D4D",
           "ink": "#272727"}
COL = {"realkd_lwf": PALETTE["blue_main"], "realkd": PALETTE["blue_secondary"],          # studied method (blues)
       "featkd_lwf": "#5BAF5B", "featkd": PALETTE["green_3"],                              # distillation variants (greens)
       "seq": PALETTE["red_strong"], "lwf": "#D88F8A",                                     # baselines (reds)
       "gpl": PALETTE["violet"], "none": PALETTE["grey_mid"], "other": PALETTE["neutral"]}
MARK = {"realkd_lwf": "o", "realkd": "s", "featkd_lwf": "D", "featkd": "P", "seq": "v", "lwf": "^", "gpl": "X", "none": "o"}
NAME = {"none": "No update", "seq": "Seq. FT", "lwf": "LwF", "featkd": "Sym. KD", "realkd": "GA", "realkd_lwf": "GA+LwF",
        "featkd_lwf": "Sym. KD+LwF", "ewc": "EWC", "lnmerge": "LN merging", "head": "Head-only FT", "replay": "Replay",
        "joint": "Joint", "fakekd": "Fake-only KD", "randkd": "Random-subset KD", "gpl": "GPL (re-impl.)",
        "base": "Before updates"}
SEQ_BLUE = LinearSegmentedColormap.from_list("f4p_blue", ["#FFFFFF", "#BFD3EA", PALETTE["blue_secondary"], PALETTE["blue_main"]])


def apply_publication_style(font_size=16, axes_linewidth=2.0):
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": font_size,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": axes_linewidth,
        "axes.edgecolor": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"], "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"], "xtick.major.width": 1.5, "ytick.major.width": 1.5, "xtick.major.size": 6,
        "ytick.major.size": 6, "legend.frameon": False, "legend.fontsize": font_size - 3, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 3, "ytick.labelsize": font_size - 3, "svg.fonttype": "none", "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "axes.grid": False, "lines.linewidth": 3, "lines.markersize": 8})


def fig_ax(w, h, nrows=1, ncols=1, **kw):
    return plt.subplots(nrows, ncols, figsize=(w * SCALE, h * SCALE), **kw)


def finalize_figure(fig, name, pad=1.0, tight=True, bbox=False):
    if tight:
        fig.tight_layout(pad=pad)
    kw = {"bbox_inches": "tight", "pad_inches": 0.08} if bbox else {}
    fig.savefig(f"{OUT}/{name}.pdf", dpi=300, **kw)
    fig.savefig(f"{OUT}/{name}.png", dpi=150, **kw)
    plt.close(fig)
    print("wrote", name)


def line(ax, x, y, m, label=None, hi=None, **kw):
    hi = (m in ("realkd_lwf",)) if hi is None else hi
    ax.plot(x, y, color=COL.get(m, COL["other"]), marker=MARK.get(m, "o"), linewidth=4 if hi else 2.6,
            markersize=10 if hi else 8, markeredgecolor="white", markeredgewidth=1.2, label=label or NAME.get(m, m),
            zorder=5 if hi else 3, linestyle="--" if m == "none" else "-", **kw)


def jl(p):
    return json.load(open(p))


apply_publication_style()
R = f"{RAD}/runs"
S1 = jl(f"{R}/j1_summary.json")
U0 = S1["S1"]["none"]["U"]

# ------------------------------------------------------------------ 1. development probe curves (3 seeds, min-max band)
dev = {"seq": "gate1_v1.1", "lwf": "gate2", "featkd": "gate2", "realkd": "e1", "realkd_lwf": "e2"}
fig, ax = fig_ax(3.5, 2.15)
ends = {}
for m in ["seq", "featkd", "lwf", "realkd", "realkd_lwf"]:
    c = np.array([[jl(f"{R}/{dev[m]}/{m}_seed{s}/results.json")["stages"][i]["dev"]["FSh_probe"]["auroc"] * 100
                   for i in range(4)] for s in range(3)])
    line(ax, range(4), c.mean(0), m)
    ax.fill_between(range(4), c.min(0), c.max(0), color=COL[m], alpha=0.15, linewidth=0)
    ends[m] = c.mean(0)[-1]
placed = []
for m, v in sorted(ends.items(), key=lambda kv: kv[1]):
    yl = v if not placed or v - placed[-1] >= 0.6 else placed[-1] + 0.6
    placed.append(yl)
    ax.annotate(f"{v:.1f}", (3, v), xytext=(3.1, yl), va="center", fontsize=12, color=PALETTE["ink"])
ax.set_xticks(range(4), ["Base (FF++)", "+DFDC-P", "+DFD", "+CDF"])
ax.set_xlim(-0.15, 3.45)
ax.set_ylabel("FaceShifter probe AUROC (%)")
ax.legend(loc="lower left", ncol=2, handlelength=1.8, fontsize=12)
finalize_figure(fig, "fig_dev_curves")

# ------------------------------------------------------------------ 2. unseen panel by source (grouped bars, annotated)
c1 = {m: jl(f"{R}/c1/{m}_seed0/results.json") for m in ["none", "realkd_lwf", "lwf", "seq", "realkd", "featkd"]}
sets = [("WDF_aligned", "WDF"), ("DFF_inpainting", "DFF-Inpaint"), ("DFF_insight", "DFF-Insight"), ("DFF_text2img", "DFF-T2I")]
ms = ["none", "seq", "lwf", "realkd_lwf"]
fig, ax = fig_ax(3.5, 2.05)
w, x = 0.2, np.arange(len(sets))
for i, m in enumerate(ms):
    v = [c1[m]["panel"][k]["auroc"] * 100 for k, _ in sets]
    ax.bar(x + (i - 1.5) * w, v, width=w * 0.92, color=COL[m], edgecolor="black", linewidth=1.5, label=NAME[m],
           hatch="//" if m == "none" else None, zorder=3)
    for xx, vv in zip(x + (i - 1.5) * w, v):
        ax.text(xx, vv + 0.5, f"{vv:.1f}", ha="center", va="bottom", fontsize=9, rotation=90, color=PALETTE["ink"])
ax.set_xticks(x, [n for _, n in sets])
ax.set_ylim(60, 101)
ax.set_ylabel("AUROC (%)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), handlelength=1.2, columnspacing=1.0, fontsize=12)
finalize_figure(fig, "fig_unseen_breakdown")

# ------------------------------------------------------------------ 3. learning vs generalization, all S1 methods
fig, ax = fig_ax(3.5, 2.6)
LEADER = {"lwf": (95.0, 86.72), "featkd": (95.0, 87.08), "randkd": (94.18, 87.08), "replay": (95.0, 86.47),
          "ewc": (94.2, 86.3), "seq": (95.0, 86.2)}
offs = {"none": (8, -14), "head": (8, 6), "realkd_lwf": (-72, 10), "realkd": (8, 6), "lnmerge": (-68, -4), "joint": (-28, -18),
        "featkd_lwf": (8, 2), "fakekd": (-76, -4)}
for m, v in S1["S1"].items():
    hi = m in ("realkd_lwf", "realkd")
    c = COL.get(m, COL["other"])
    ax.scatter(v["AA"], v["U"], s=170 if hi else 110, color=c, marker=MARK.get(m, "o"), edgecolor="black",
               linewidth=1.0, zorder=4 if hi else 3)
    if m in LEADER:
        ax.annotate(NAME[m], (v["AA"], v["U"]), xytext=LEADER[m], textcoords="data", fontsize=11, color=PALETTE["grey_dark"],
                    arrowprops=dict(arrowstyle="-", color=PALETTE["grey_mid"], linewidth=1, shrinkA=0, shrinkB=4))
    else:
        ax.annotate(NAME[m], (v["AA"], v["U"]), textcoords="offset points", xytext=offs.get(m, (8, 4)), fontsize=11,
                    color=PALETTE["ink"] if hi else PALETTE["grey_dark"], fontweight="bold" if hi else "normal")
ax.set_xlabel("Mean test AUROC on trained datasets, AA (%)")
ax.set_ylabel("Unseen-panel score $U$ (%)")
finalize_figure(fig, "fig_tradeoff_all")

# ------------------------------------------------------------------ 4. weight sensitivity frontiers
fig, ax = fig_ax(3.5, 2.4)
for fam in ["featkd", "realkd"]:
    pts = sorted(((float(k), v) for k, v in S1["lambda"][fam].items()), key=lambda t: t[0])
    line(ax, [v["AA"] for _, v in pts], [v["U"] for _, v in pts], fam, label=("GA" if fam == "realkd" else "Symmetric KD"), hi=False)
    for lam, v in pts:
        off = ((6, -20) if lam == 10 else (8, -6)) if fam == "featkd" else ((-22, -22) if lam == 10 else (6, 6))
        ax.annotate(f"$\\lambda$={lam:g}", (v["AA"], v["U"]), textcoords="offset points", xytext=off,
                    fontsize=10, color=PALETTE["grey_dark"])
for k, v in S1["lambda"]["realkd_lwf"].items():
    ax.scatter(v["AA"], v["U"], s=110, marker="o", color=COL["realkd_lwf"], edgecolor="white", linewidth=1.2, zorder=5)
    g, l_ = k.split("_")
    ax.annotate(f"({g[1:]},{l_[1:]})", (v["AA"], v["U"]), textcoords="offset points", xytext=(6, 4), fontsize=9.5,
                color=COL["realkd_lwf"])
ax.scatter([], [], s=110, marker="o", color=COL["realkd_lwf"], label="GA+LwF $(\\lambda_g,\\lambda_l)$")
ax.axhline(U0, color=PALETTE["grey_mid"], linewidth=2, linestyle="--", alpha=0.7)
ax.text(94.36, U0 - 0.04, "no update", fontsize=10, color=PALETTE["grey_dark"], va="top")
ax.set_xlabel("AA (%)")
ax.set_ylabel("$U$ (%)")
ax.legend(loc="lower left", fontsize=11)
finalize_figure(fig, "fig_lambda")

# ------------------------------------------------------------------ 5. replication small multiples
methods = ["none", "seq", "lwf", "featkd", "featkd_lwf", "realkd", "realkd_lwf"]
settings = [("S1", {m: S1["S1"][m] for m in methods}), ("S2", S1["replication"]["S2"]["metrics"]),
            ("S3", S1["replication"]["S3"]["metrics"]), ("PE", S1["replication"]["PE"]["metrics"])]
fig, axes = fig_ax(7.16, 1.9, 1, 4, sharey=True)
for ax, (nm, met) in zip(axes, settings):
    ref = met.get("none", S1["S1"]["none"])["U"]
    for i, m in enumerate(methods):
        if m not in met:
            continue
        hi = m in ("realkd", "realkd_lwf")
        ax.hlines(i, ref, met[m]["U"], color=COL[m], linewidth=3 if hi else 2, alpha=0.6, zorder=2)
        ax.scatter(met[m]["U"], i, s=130 if hi else 90, color=COL[m], marker=MARK[m], edgecolor="black", linewidth=0.8, zorder=3)
        left = met[m]["U"] < ref
        ax.annotate(f"{met[m]['U']:.2f}", (met[m]["U"], i), textcoords="offset points", xytext=(-10 if left else 10, 0),
                    ha="right" if left else "left", va="center", fontsize=10, color=PALETTE["grey_dark"])
    ax.axvline(ref, color=PALETTE["grey_mid"], linewidth=1.8, linestyle="--")
    ax.set_title({"S1": "Order S1 (CLIP)", "S2": "Order S2 (CLIP)", "S3": "Order S3 (CLIP)", "PE": "Order S1 (PE backbone)"}[nm],
                 fontsize=14)
    lo, hi_ = min(v["U"] for v in met.values()), max(v["U"] for v in met.values())
    ax.set_xlim(lo - 0.75, hi_ + 0.75)
axes[0].set_yticks(range(len(methods)), [NAME[m] for m in methods])
fig.supxlabel("Unseen-panel score $U$ (%); dashed line: no update", fontsize=14)
finalize_figure(fig, "fig_replication")

# ------------------------------------------------------------------ 6. stage-wise unseen score along S1
sw = jl(f"{RAD}/analysis_f/stagewise_U.json")
fig, ax = fig_ax(3.5, 2.1)
for m in ["seq", "featkd", "lwf", "realkd", "realkd_lwf"]:
    ys = [U0, sw[m]["DFDCP"]["U"], sw[m]["DFD"]["U"], S1["S1"][m]["U"]]
    line(ax, range(4), ys, m)
    ax.annotate(f"{ys[-1]:.2f}", (3, ys[-1]), xytext=(3.08, ys[-1] + {"featkd": 0.07, "lwf": -0.07}.get(m, 0)), va="center",
                fontsize=11, color=PALETTE["grey_dark"])
ax.set_xticks(range(4), ["Base (FF++)", "+DFDC-P", "+DFD", "+CDF"])
ax.set_xlim(-0.15, 3.5)
ax.set_ylabel("Unseen-panel score $U$ (%)")
ax.legend(loc="lower left", ncol=2, fontsize=11)
finalize_figure(fig, "fig_stagewise")

# ------------------------------------------------------------------ 7. data regime
fig, axes = fig_ax(3.5, 1.9, 1, 2)
for m in ["seq", "lwf", "realkd_lwf"]:
    ks = [5, 10, 25]
    line(axes[0], ks, [S1["shots"][f"{m}_{k}"]["U"] for k in ks], m)
    line(axes[1], ks, [S1["shots"][f"{m}_{k}"]["AA"] for k in ks], m)
axes[0].axhline(U0, color=PALETTE["grey_mid"], linewidth=1.8, linestyle="--")
axes[1].axhline(S1["S1"]["none"]["AA"], color=PALETTE["grey_mid"], linewidth=1.8, linestyle="--")
for ax, yl in zip(axes, ["$U$ (%)", "AA (%)"]):
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25], ["5", "10", "25"])
    ax.minorticks_off()
    ax.set_xlabel("Videos per class per task")
    ax.set_ylabel(yl)
h, l_ = axes[0].get_legend_handles_labels()
fig.legend(h, l_, loc="upper center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.88))
finalize_figure(fig, "fig_shots", tight=False)

# ------------------------------------------------------------------ 8. genuine-feature drift vs change in U
d = jl(f"{RAD}/analysis_f/drift.json")
fig, ax = fig_ax(3.5, 2.2)


def fam(n):
    if n.startswith("realkd_lwf"):
        return "realkd_lwf"
    if n.startswith("realkd"):
        return "realkd"
    if n.startswith("featkd") and "lwf" not in n:
        return "featkd"
    return "other"


xs, ys = [], []
for n, xv, yv in d["points"]:
    f = fam(n)
    ax.scatter(xv, yv, s=110, color=COL[f], marker=MARK.get(f, "o"), edgecolor="black", linewidth=0.8, zorder=3)
    xs.append(xv); ys.append(yv)
b = np.polyfit(xs, ys, 1)
xx = np.linspace(0, max(xs), 50)
ax.plot(xx, np.polyval(b, xx), color=PALETTE["grey_dark"], linewidth=2, linestyle=":")
handles = [Line2D([0], [0], marker=MARK.get(f, "o"), color="w", markerfacecolor=COL[f], markeredgecolor="black", markersize=11,
                  label=lab) for f, lab in [("realkd", "GA ($\\lambda$ sweep)"), ("realkd_lwf", "GA+LwF"),
                                            ("featkd", "Sym. KD ($\\lambda$ sweep)"), ("other", "other methods")]]
ax.legend(handles=handles, loc="lower left", fontsize=11)
ax.set_xlabel("Genuine-feature drift $1-\\cos(z, z_{\\mathrm{base}})$")
ax.set_ylabel("$U - U_{\\mathrm{no\\ update}}$ (points)")
ax.text(0.98, 0.95, f"Spearman $\\rho$ = {d['corr_genuine_drift_vs_dU']['spearman']:.2f}, n = {d['corr_genuine_drift_vs_dU']['n']}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11, color=PALETTE["grey_dark"])
finalize_figure(fig, "fig_drift")

# ------------------------------------------------------------------ 9. t-SNE (wide and compact) and patch drift maps
J2 = f"{RAD}/analysis_j2"
q = np.load(f"{J2}/qual.npz")
gid = q["tsne_gid"].astype(str)
grp = np.where(np.char.startswith(gid, "genuine"), 0, np.where(np.isin(gid, ["fake_WDF", "fake_DFF"]), 2, 1))
# 10-NN separability of genuine faces and unseen manipulations in the full 1024-d feature space (5-fold CV)
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

pf = np.load(f"{RAD}/analysis_f/probe_features.npz")
knn = {k: float(cross_val_score(KNeighborsClassifier(10, metric="cosine"), pf[k][grp != 1].astype(np.float32),
                                grp[grp != 1] == 2, cv=5).mean() * 100) for k in ["base", "seq", "lwf", "realkd_lwf"]}
print("kNN", knn)
TS = [(0, PALETTE["grey_mid"], "genuine"), (1, PALETTE["red_strong"], "stream manipulations"), (2, PALETTE["blue_main"], "unseen manipulations")]


def tsne_panel(ax, k, title):
    e = q[f"tsne_{k}"]
    for g, c, lab in TS:
        ax.scatter(e[grp == g, 0], e[grp == g, 1], s=6, color=c, alpha=0.55, linewidths=0, label=lab, rasterized=True)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_visible(False)


fig, axes = fig_ax(7.16, 1.95, 1, 4)
for ax, k in zip(axes, ["base", "seq", "lwf", "realkd_lwf"]):
    tsne_panel(ax, k, f"{NAME[k]}\n10-NN genuine vs unseen: {knn[k]:.1f}%")
h, l_ = axes[0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=3, markerscale=4, fontsize=13, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=1.0, rect=(0, 0.08, 1, 1))
finalize_figure(fig, "fig_tsne", tight=False)
fig, axes = fig_ax(3.5, 3.05, 2, 2)
for ax, k in zip(axes.flat, ["base", "seq", "lwf", "realkd_lwf"]):
    tsne_panel(ax, k, f"{NAME[k]} (10-NN {knn[k]:.1f}%)")
h, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=3, markerscale=4, fontsize=11, bbox_to_anchor=(0.5, 0.0), handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(pad=0.6, h_pad=1.0, w_pad=0.6, rect=(0, 0.06, 1, 1))
finalize_figure(fig, "fig_tsne_col", tight=False)

fig, axes = fig_ax(3.5, 2.55, 2, 3)
dm = ["seq", "lwf", "realkd_lwf"]
vmax = max(q[f"driftmap_{m}_{g}"].max() for m in dm for g in ["genuine", "fake"])
for i, g in enumerate(["genuine", "fake"]):
    for j, m in enumerate(dm):
        ax = axes[i, j]
        im = ax.imshow(q[f"driftmap_{m}_{g}"], cmap=SEQ_BLUE, vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_visible(False)
        ax.text(0.5, -0.06, f"mean {q[f'driftmap_{m}_{g}'].mean():.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=10.5)
        if i == 0:
            ax.set_title(NAME[m], fontsize=13)
        if j == 0:
            ax.set_ylabel("genuine faces" if g == "genuine" else "manipulated faces", fontsize=12)
cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=10, length=4)
cb.outline.set_visible(False)
cb.set_label("$1-\\cos$ to the model before updates", fontsize=11)
finalize_figure(fig, "fig_driftmaps", tight=False, bbox=True)

# ------------------------------------------------------------------ 10. example faces (images; style-consistent titles)
from PIL import Image  # noqa: E402

ex = jl(f"{J2}/qual_examples.json")
sel = [i for i, e in enumerate(ex) if e["rule"] in ("largest_seq_drop", "failure_gadlwf")]
fig, axes = fig_ax(7.16, 1.45, 1, len(sel))
for j, i in enumerate(sel):
    e = ex[i]
    axes[j].imshow(Image.open(e["image"]).convert("RGB").resize((224, 224)))
    axes[j].set_title(("lost by Seq. FT" if e["rule"] == "largest_seq_drop" else "missed by GA+LwF") + "\n"
                      + e["set"].replace("DFF_", "").replace("text2img", "T2I").replace("inpainting", "inpaint"), fontsize=11)
    axes[j].text(0.5, -0.05, f"{e['p_none']:.2f} / {e['p_seq']:.2f} / {e['p_realkd_lwf']:.2f}", transform=axes[j].transAxes,
                 ha="center", va="top", fontsize=11)
    axes[j].set_xticks([]); axes[j].set_yticks([])
    for sp in ["left", "bottom"]:
        axes[j].spines[sp].set_visible(False)
finalize_figure(fig, "fig_examples", pad=0.4)

# ------------------------------------------------------------------ 11. misranked pairs vs base margin
mg = jl(f"{J2}/margin.json")
labs = ["0–0.5", "0.5–1", "1–2", "2–3", "3–4", "4–6", "6–8", ">8"]
fig, ax = fig_ax(3.5, 1.95)
for m in ["seq", "lwf", "realkd", "featkd_lwf", "realkd_lwf"]:
    fr = 100 * np.array(mg["models"][m]["flip_rate"])
    line(ax, range(len(fr)), fr, m, label=f"{NAME[m]} ({100 * mg['models'][m]['overall']:.1f}%)")
ax.set_xticks(range(len(labs)), labs)
ax.set_xlabel("Logit margin $s(u)-s(r)$ before the updates")
ax.set_ylabel("Pairs misranked (%)")
ax.set_ylim(-1, 45)
ax.legend(fontsize=11, loc="upper right")
finalize_figure(fig, "fig_margin")

# ------------------------------------------------------------------ 12. robustness to degradations
tb = jl(f"{J2}/robust_summary.json")["table"]
fams = [("JPEG quality", ["clean", "jpeg75", "jpeg50", "jpeg30"], ["none", "75", "50", "30"]),
        ("Gaussian blur $\\sigma$ (px)", ["clean", "blur1", "blur2", "blur3"], ["0", "1", "2", "3"]),
        ("Gaussian noise $\\sigma$ (/255)", ["clean", "noise5", "noise10", "noise20"], ["0", "5", "10", "20"]),
        ("Down/up-scaling factor", ["clean", "resize0.5", "resize0.35", "resize0.25"], ["1", "0.5", "0.35", "0.25"])]
fig, axes = fig_ax(7.16, 1.85, 1, 4, sharey=True)
for ax, (title, cs, lb) in zip(axes, fams):
    for m in ["none", "seq", "lwf", "featkd_lwf", "realkd_lwf"]:
        line(ax, range(4), [tb[m][c] for c in cs], m)
    ax.set_xticks(range(4), lb)
    ax.set_xlabel(title)
axes[0].set_ylabel("Unseen-panel score $U$ (%)")
h, l_ = axes[0].get_legend_handles_labels()
fig.legend(h, l_, loc="upper center", ncol=5, fontsize=13, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.9))
finalize_figure(fig, "fig_robust", tight=False)

# ------------------------------------------------------------------ 13. five seeds (strip plot with means)
S3 = jl(f"{R}/j3_summary.json")
order = ["seq", "gpl", "featkd", "lwf", "realkd", "featkd_lwf", "realkd_lwf"]
fig, ax = fig_ax(3.5, 2.0)
for i, m in enumerate(order):
    us = [v["U"] for v in S3["S"]["per_seed"][m].values()]
    jit = np.linspace(-0.14, 0.14, len(us))
    hi = m == "realkd_lwf"
    ax.scatter(np.array(us), i + jit, s=110 if hi else 85, color=COL[m], marker=MARK[m], edgecolor="black", linewidth=0.8, zorder=3)
    mu = np.mean(us)
    ax.plot([mu, mu], [i - 0.32, i + 0.32], color=PALETTE["ink"], linewidth=3, zorder=4)
    ax.annotate(f"{mu:.2f}$\\pm${np.std(us, ddof=1):.2f}", (max(us), i), textcoords="offset points", xytext=(9, -5),
                fontsize=10.5, color=PALETTE["grey_dark"])
ax.axvline(87.97, color=PALETTE["grey_mid"], linewidth=1.8, linestyle="--")
ax.text(88.02, len(order) - 0.45, "no update", fontsize=10.5, color=PALETTE["grey_dark"])
ax.set_yticks(range(len(order)), [NAME[m] for m in order])
ax.set_xlim(83.6, 89.6)
ax.set_xlabel("Unseen-panel score $U$ (%), five seeds; bar: mean")
finalize_figure(fig, "fig_seeds")

# ------------------------------------------------------------------ 14. nine-update stream
swl = jl(f"{RAD}/analysis_j3/stagewise_long.json")
toks = swl["tokens"]
labs = ["Base"] + [t.replace("DFDCP#", "P").replace("DFD#", "D").replace("CDF", "C") for t in toks[1:]]
u0 = swl["U"]["none"]["FF"]
fig, ax = fig_ax(3.5, 2.2)
for m in ["seq", "gpl", "lwf", "featkd_lwf", "realkd", "realkd_lwf"]:
    ys = [u0] + [swl["U"][m][t] for t in toks[1:]]
    line(ax, range(len(ys)), ys, m)
    ax.annotate(f"{ys[-1]:.2f}", (len(ys) - 1, ys[-1]), textcoords="offset points",
                xytext=(8, {"featkd_lwf": 1, "realkd": -10}.get(m, -4)), fontsize=10.5,
                color=PALETTE["grey_dark"])
ax.axhline(u0, color=PALETTE["grey_mid"], linewidth=1.8, linestyle="--")
for xv in (4.5, 8.5):
    ax.axvline(xv, color=PALETTE["neutral"], linewidth=1.5)
ax.set_xticks(range(len(labs)), labs)
ax.set_xlim(-0.3, len(labs) - 0.1)
ax.set_xlabel("After update (P: DFDC-P session, D: DFD session, C: Celeb-DF)")
ax.set_ylabel("$U$ (%)")
h, l_ = ax.get_legend_handles_labels()
fig.legend(h, l_, loc="upper center", ncol=3, fontsize=11.5, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.8)
fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.87))
finalize_figure(fig, "fig_long", tight=False)
print("ALL FIGURES DONE")
