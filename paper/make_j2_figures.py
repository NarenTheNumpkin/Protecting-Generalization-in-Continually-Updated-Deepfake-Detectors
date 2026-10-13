"""Figures for the J2 (Tier 1) additions. Reads radial/analysis_j2/* and radial/analysis_f/probe_features.npz.
Sections whose inputs do not exist yet are skipped."""

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

for _f in ["regular", "bold", "italic", "bolditalic"]:
    font_manager.fontManager.addfont(f"{_HERE}/fonts/texgyretermes-{_f}.ttf")
RAD = _os.path.join(_HERE, "..", "experiments")
J2 = f"{RAD}/analysis_j2"
OUT = _os.path.join(_HERE, "figures")
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COL = {"realkd_lwf": BLUE, "seq": ORANGE, "lwf": AQUA, "featkd_lwf": YELLOW, "realkd": MAGENTA, "none": NEUTRAL, "featkd": "#c9c8c3"}
NAME = {"none": "No update", "seq": "Seq. FT", "lwf": "LwF", "featkd": "Sym. KD", "realkd": "GA", "realkd_lwf": "GA+LwF",
        "featkd_lwf": "Sym. KD+LwF", "base": "Before updates", "ewc": "EWC", "lnmerge": "LN merging"}
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#a9c8f0", "#2a78d6", "#123f7a"])
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


stats_out = {}

# ---- t-SNE of probe features, with kNN separability of genuine vs unseen fakes in the 1024-d space
if os.path.exists(f"{J2}/qual.npz"):
    q = np.load(f"{J2}/qual.npz")
    pf = np.load(f"{RAD}/analysis_f/probe_features.npz")
    gid = q["tsne_gid"].astype(str)
    grp = np.where(np.char.startswith(gid, "genuine"), 0, np.where(np.isin(gid, ["fake_WDF", "fake_DFF"]), 2, 1))
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95))
    knn = {}
    for ax, k in zip(axes, ["base", "seq", "lwf", "realkd_lwf"]):
        e = q[f"tsne_{k}"]
        for g, c, lab, s in [(0, NEUTRAL, "genuine", 1.6), (1, ORANGE, "stream manipulations", 1.6), (2, BLUE, "unseen manipulations", 1.6)]:
            ax.scatter(e[grp == g, 0], e[grp == g, 1], s=s, color=c, alpha=0.55, linewidths=0, label=lab, rasterized=True)
        sel = grp != 1
        X = pf[k][sel].astype(np.float32)
        acc = cross_val_score(KNeighborsClassifier(10, metric="cosine"), X, grp[sel] == 2, cv=5).mean() * 100
        knn[k] = float(acc)
        ax.set_title(f"{NAME[k]}\n10-NN genuine vs unseen: {acc:.1f}%", fontsize=7, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, markerscale=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1))
    save(fig, "fig_tsne")
    stats_out["knn_genuine_vs_unseen"] = knn
    print("kNN", knn)

    # ---- mean per-patch drift maps
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.55))
    ms = ["seq", "lwf", "realkd_lwf"]
    vmax = max(q[f"driftmap_{m}_{g}"].max() for m in ms for g in ["genuine", "fake"])
    for i, g in enumerate(["genuine", "fake"]):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            im = ax.imshow(q[f"driftmap_{m}_{g}"], cmap=SEQ_CMAP, vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.5, -0.06, f"mean {q[f'driftmap_{m}_{g}'].mean():.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=INK2)
            if i == 0:
                ax.set_title(NAME[m], fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel("genuine faces" if g == "genuine" else "manipulated faces", fontsize=6.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_visible(False)
    cb.set_label("$1-\\cos$ to the model before updates", fontsize=6)
    save(fig, "fig_driftmaps")

    # ---- examples: unseen fakes lost by Seq. FT, and failure cases of GA+LwF (gradient saliency was uninformative)
    ex = json.load(open(f"{J2}/qual_examples.json"))
    sel = [i for i, e in enumerate(ex) if e["rule"] in ("largest_seq_drop", "failure_gadlwf")]
    fig, axes = plt.subplots(1, len(sel), figsize=(7.16, 1.45))
    for j, i in enumerate(sel):
        e = ex[i]
        axes[j].imshow(Image.open(e["image"]).convert("RGB").resize((224, 224)))
        axes[j].set_title(("lost by Seq. FT" if e["rule"] == "largest_seq_drop" else "missed by GA+LwF") + "\n"
                          + e["set"].replace("DFF_", "").replace("text2img", "T2I").replace("inpainting", "inpaint"), fontsize=5.8, color=INK)
        axes[j].text(0.5, -0.05, f"{e['p_none']:.2f} / {e['p_seq']:.2f} / {e['p_realkd_lwf']:.2f}", transform=axes[j].transAxes,
                     ha="center", va="top", fontsize=6, color=INK2)
        axes[j].set_xticks([]); axes[j].set_yticks([])
        for sp in axes[j].spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.3)
    save(fig, "fig_examples")

# ---- pair flip rate versus base margin (post hoc check of Proposition 2)
if os.path.exists(f"{J2}/margin.json"):
    mg = json.load(open(f"{J2}/margin.json"))
    labs = ["0\u20130.5", "0.5\u20131", "1\u20132", "2\u20133", "3\u20134", "4\u20136", "6\u20138", ">8"]
    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    for m in ["seq", "lwf", "realkd", "featkd_lwf", "realkd_lwf"]:
        fr = 100 * np.array(mg["models"][m]["flip_rate"])
        hi = m == "realkd_lwf"
        ax.plot(range(len(fr)), fr, color=COL[m], marker="o", markersize=3 if hi else 2.4, linewidth=1.5 if hi else 1.0,
                label=f"{NAME[m]} ({100 * mg['models'][m]['overall']:.1f}%)", zorder=4 if hi else 3)
    ax.set_xticks(range(len(labs)), labs)
    ax.set_xlabel("Logit margin $s(u)-s(r)$ before the updates")
    ax.set_ylabel("Pairs misranked after updates (%)")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    style(ax)
    save(fig, "fig_margin")

# ---- robustness to degradations
if os.path.exists(f"{J2}/robust_summary.json"):
    rb = json.load(open(f"{J2}/robust_summary.json"))
    tb = rb["table"]
    fams = [("JPEG quality", ["clean", "jpeg75", "jpeg50", "jpeg30"], ["none", "75", "50", "30"]),
            ("Gaussian blur $\\sigma$ (px)", ["clean", "blur1", "blur2", "blur3"], ["0", "1", "2", "3"]),
            ("Gaussian noise $\\sigma$ (/255)", ["clean", "noise5", "noise10", "noise20"], ["0", "5", "10", "20"]),
            ("Down/up-scaling factor", ["clean", "resize0.5", "resize0.35", "resize0.25"], ["1", "0.5", "0.35", "0.25"])]
    ms = ["none", "seq", "lwf", "featkd_lwf", "realkd_lwf"]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85), sharey=True)
    for ax, (title, cs, labs) in zip(axes, fams):
        for m in ms:
            ys = [tb[m][c] for c in cs]
            hi = m == "realkd_lwf"
            ax.plot(range(4), ys, color=COL[m], linewidth=1.5 if hi else 1.0, marker="o", markersize=3 if hi else 2.4,
                    linestyle="--" if m == "none" else "-", label=NAME[m], zorder=4 if hi else 3)
        ax.set_xticks(range(4), labs)
        ax.set_xlabel(title)
        style(ax)
    axes[0].set_ylabel("Unseen-panel score $U$ (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    save(fig, "fig_robust")

json.dump(stats_out, open(f"{J2}/figure_stats.json", "w"), indent=1)
