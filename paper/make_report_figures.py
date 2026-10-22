"""Portrait-page variants of the wide paper figures for the project report (same data, same style as make_figures.py):
replication and robustness as 2x2 grids, example faces as 2x4. Usage: python make_report_figures.py <experiments_dir> <out_dir>"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "make_figures.py")).read()
header = src[: src.index("apply_publication_style()\nR = ")]
exec(header)  # palette, names, helpers; RAD and OUT come from sys.argv
apply_publication_style()
R = f"{RAD}/runs"
S1 = jl(f"{R}/j1_summary.json")

# replication, 2 x 2
methods = ["none", "seq", "lwf", "featkd", "featkd_lwf", "realkd", "realkd_lwf"]
settings = [("S1", {m: S1["S1"][m] for m in methods}), ("S2", S1["replication"]["S2"]["metrics"]),
            ("S3", S1["replication"]["S3"]["metrics"]), ("PE", S1["replication"]["PE"]["metrics"])]
TITLE = {"S1": "Order S1 (CLIP)", "S2": "Order S2 (CLIP)", "S3": "Order S3 (CLIP)", "PE": "Order S1 (PE backbone)"}
fig, axes = fig_ax(3.5, 3.4, 2, 2)
for ax, (nm, met) in zip(axes.flat, settings):
    ref = met.get("none", S1["S1"]["none"])["U"]
    for i, m in enumerate(methods):
        if m not in met:
            continue
        hi = m in ("realkd", "realkd_lwf")
        ax.hlines(i, ref, met[m]["U"], color=COL[m], linewidth=3 if hi else 2, alpha=0.6, zorder=2)
        ax.scatter(met[m]["U"], i, s=120 if hi else 80, color=COL[m], marker=MARK[m], edgecolor="black", linewidth=0.8, zorder=3)
        left = met[m]["U"] < ref
        ax.annotate(f"{met[m]['U']:.2f}", (met[m]["U"], i), textcoords="offset points", xytext=(-9 if left else 9, 0),
                    ha="right" if left else "left", va="center", fontsize=10, color=PALETTE["grey_dark"])
    ax.axvline(ref, color=PALETTE["grey_mid"], linewidth=1.8, linestyle="--")
    ax.set_title(TITLE[nm], fontsize=13)
    lo, hi_ = min(v["U"] for v in met.values()), max(v["U"] for v in met.values())
    ax.set_xlim(lo - 0.8, hi_ + 0.8)
    ax.set_yticks(range(len(methods)), [NAME[m] for m in methods] if ax in axes[:, 0] else [""] * len(methods))
fig.supxlabel("Unseen-panel score $U$ (%); dashed line: no update", fontsize=13)
finalize_figure(fig, "fig_replication_2x2")

# robustness, 2 x 2
tb = jl(f"{RAD}/analysis_j2/robust_summary.json")["table"]
fams = [("JPEG quality", ["clean", "jpeg75", "jpeg50", "jpeg30"], ["none", "75", "50", "30"]),
        ("Gaussian blur $\\sigma$ (px)", ["clean", "blur1", "blur2", "blur3"], ["0", "1", "2", "3"]),
        ("Gaussian noise $\\sigma$ (/255)", ["clean", "noise5", "noise10", "noise20"], ["0", "5", "10", "20"]),
        ("Down/up-scaling factor", ["clean", "resize0.5", "resize0.35", "resize0.25"], ["1", "0.5", "0.35", "0.25"])]
fig, axes = fig_ax(3.5, 3.3, 2, 2)
for ax, (title, cs, lb) in zip(axes.flat, fams):
    for m in ["none", "seq", "lwf", "featkd_lwf", "realkd_lwf"]:
        line(ax, range(4), [tb[m][c] for c in cs], m)
    ax.set_xticks(range(4), lb)
    ax.set_xlabel(title, fontsize=13)
for ax in axes[:, 0]:
    ax.set_ylabel("$U$ (%)")
h, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc="upper center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.88))
finalize_figure(fig, "fig_robust_2x2", tight=False)

# example faces, 2 x 4
from PIL import Image  # noqa: E402

ex = jl(f"{RAD}/analysis_j2/qual_examples.json")
sel = [i for i, e in enumerate(ex) if e["rule"] in ("largest_seq_drop", "failure_gadlwf")]
fig, axes = fig_ax(3.5, 2.25, 2, 4)
for j, i in enumerate(sel):
    e, ax = ex[i], axes.flat[j]
    ax.imshow(Image.open(e["image"]).convert("RGB").resize((224, 224)))
    ax.set_title(("lost by Seq. FT" if e["rule"] == "largest_seq_drop" else "missed by GA+LwF") + "\n"
                 + e["set"].replace("DFF_", "").replace("text2img", "T2I").replace("inpainting", "inpaint"), fontsize=10.5)
    ax.text(0.5, -0.05, f"{e['p_none']:.2f} / {e['p_seq']:.2f} / {e['p_realkd_lwf']:.2f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_visible(False)
finalize_figure(fig, "fig_examples_2x4", pad=0.5)
