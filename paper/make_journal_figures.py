"""Vector figures for the journal version. Every number is read from saved results:
radial/runs/j1_summary.json, radial/analysis_f/*.json and the C1 run directories."""

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json

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
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COL = {"realkd_lwf": BLUE, "seq": ORANGE, "lwf": AQUA, "featkd": YELLOW, "realkd": MAGENTA, "none": NEUTRAL}
NAME = {"none": "No update", "seq": "Seq. FT", "lwf": "LwF", "featkd": "Sym. KD", "realkd": "GA", "realkd_lwf": "GA+LwF",
        "featkd_lwf": "Sym. KD+LwF", "ewc": "EWC", "lnmerge": "LN merging", "head": "Head-only FT", "replay": "Replay",
        "joint": "Joint", "fakekd": "Fake-only KD", "randkd": "Random-subset KD"}
plt.rcParams.update({"font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 7.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "legend.fontsize": 6.5})
S = json.load(open(f"{RAD}/runs/j1_summary.json"))
U0 = S["S1"]["none"]["U"]


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)


# ---- trade-off scatter, all S1 methods
fig, ax = plt.subplots(figsize=(3.5, 2.6))
offs = {"none": (4, -7), "head": (4, 3), "realkd_lwf": (-36, 5), "realkd": (4, 3), "lnmerge": (-34, -2), "joint": (-14, -9),
        "featkd_lwf": (4, 1), "featkd": (4, -6), "randkd": (4, 2), "lwf": (-16, 4), "fakekd": (-38, -2), "replay": (4, 1),
        "ewc": (-4, -9), "seq": (4, -4)}
LEADER = {"lwf": (95.0, 86.72), "featkd": (95.0, 87.08), "randkd": (94.18, 87.08), "replay": (95.0, 86.47),
          "ewc": (94.2, 86.3), "seq": (95.0, 86.2)}
for m, x in S["S1"].items():
    hi = m in ("realkd_lwf", "realkd")
    c = COL.get(m, NEUTRAL) if m in COL else "#c9c8c3"
    ax.scatter(x["AA"], x["U"], s=24 if hi else 15, color=c, edgecolor="white", linewidth=0.5, zorder=3)
    if m in LEADER:  # crowded cluster: label with a leader line to a fixed text position
        ax.annotate(NAME[m], (x["AA"], x["U"]), xytext=LEADER[m], textcoords="data", fontsize=6, color=INK2,
                    arrowprops=dict(arrowstyle="-", color="#b5b4af", linewidth=0.5, shrinkA=0, shrinkB=2))
    else:
        ax.annotate(NAME[m], (x["AA"], x["U"]), textcoords="offset points", xytext=offs.get(m, (4, 2)), fontsize=6,
                    color=INK if hi else INK2)
ax.set_xlabel("Mean test AUROC on trained datasets, AA (%)")
ax.set_ylabel("Unseen-panel score $U$ (%)")
style(ax, "both")
save(fig, "fig_tradeoff_all")

# ---- sensitivity frontiers
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for fam, col, lab in [("realkd", MAGENTA, "GA"), ("featkd", YELLOW, "Symmetric KD")]:
    pts = sorted(((float(k), v) for k, v in S["lambda"][fam].items()), key=lambda t: t[0])
    xs, ys = [v["AA"] for _, v in pts], [v["U"] for _, v in pts]
    ax.plot(xs, ys, color=col, linewidth=1.2, marker="o", markersize=3.5, label=lab, zorder=3)
    for (l, v) in pts:
        ax.annotate(f"$\\lambda$={l:g}", (v["AA"], v["U"]), textcoords="offset points", xytext=(3, -8 if fam == "featkd" else 3),
                    fontsize=5.5, color=INK2)
gl = S["lambda"]["realkd_lwf"]
for k, v in gl.items():
    ax.scatter(v["AA"], v["U"], s=16, marker="s", color=BLUE, edgecolor="white", linewidth=0.4, zorder=4)
    g, l = k.split("_")
    ax.annotate(f"({g[1:]},{l[1:]})", (v["AA"], v["U"]), textcoords="offset points", xytext=(3, 2), fontsize=5.2, color=BLUE)
ax.scatter([], [], s=16, marker="s", color=BLUE, label="GA+LwF $(\\lambda_g,\\lambda_l)$")
ax.axhline(U0, color=NEUTRAL, linewidth=0.8, linestyle="--")
ax.text(94.05, U0 + 0.05, "no update", fontsize=5.5, color=INK2)
ax.set_xlabel("AA (%)")
ax.set_ylabel("$U$ (%)")
ax.legend(frameon=False, loc="lower left")
style(ax, "both")
save(fig, "fig_lambda")

# ---- replication small multiples (U per method, per setting)
methods = ["none", "seq", "lwf", "featkd", "featkd_lwf", "realkd", "realkd_lwf"]
settings = [("S1", {m: S["S1"][m] for m in methods}), ("S2", S["replication"]["S2"]["metrics"]),
            ("S3", S["replication"]["S3"]["metrics"]), ("PE", S["replication"]["PE"]["metrics"])]
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.9), sharey=True)
for ax, (name, met) in zip(axes, settings):
    ref = met.get("none", S["S1"]["none"])["U"]
    for i, m in enumerate(methods):
        if m not in met:
            continue
        hi = m in ("realkd", "realkd_lwf")
        ax.scatter(met[m]["U"], i, s=20 if hi else 14, color=COL.get(m, "#c9c8c3") if m != "featkd_lwf" else "#c9c8c3",
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(f"{met[m]['U']:.2f}", (met[m]["U"], i), textcoords="offset points", xytext=(4, -2.5), fontsize=5.5, color=INK2)
    ax.axvline(ref, color=NEUTRAL, linewidth=0.7, linestyle="--")
    ttl = {"S1": "Order S1 (CLIP)", "S2": "Order S2 (CLIP)", "S3": "Order S3 (CLIP)", "PE": "Order S1 (PE backbone)"}[name]
    ax.set_title(ttl, fontsize=7.5, color=INK)
    lo = min(v["U"] for v in met.values())
    hi_ = max(v["U"] for v in met.values())
    ax.set_xlim(lo - 0.4, hi_ + 0.9)
    style(ax, "x")
axes[0].set_yticks(range(len(methods)), [NAME[m] for m in methods])
fig.supxlabel("Unseen-panel score $U$ (%); dashed line: no update (S2 and S3 share the S1 initial model)", fontsize=7, color=INK2)
save(fig, "fig_replication")

# ---- stage-wise unseen score
sw = json.load(open(f"{RAD}/analysis_f/stagewise_U.json"))
fig, ax = plt.subplots(figsize=(3.5, 2.1))
for m in ["realkd_lwf", "realkd", "lwf", "featkd", "seq"]:
    ys = [U0, sw[m]["DFDCP"]["U"], sw[m]["DFD"]["U"], S["S1"][m]["U"]]
    ax.plot(range(4), ys, color=COL[m], linewidth=1.3, marker="o", markersize=3, label=NAME[m], zorder=3)
    ax.annotate(f"{ys[-1]:.2f}", (3, ys[-1]), xytext=(3.07, ys[-1] + {"featkd": 0.06, "lwf": -0.06}.get(m, 0)), va="center", fontsize=5.8, color=INK2)
ax.set_xticks(range(4), ["Base (FF++)", "+DFDC-P", "+DFD", "+CDF"])
ax.set_xlim(-0.15, 3.45)
ax.set_ylabel("Unseen-panel score $U$ (%)")
ax.legend(frameon=False, loc="lower left", ncol=2)
style(ax)
save(fig, "fig_stagewise")

# ---- data regime
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
for m in ["realkd_lwf", "lwf", "seq"]:
    ks = [5, 10, 25]
    axes[0].plot(ks, [S["shots"][f"{m}_{k}"]["U"] for k in ks], color=COL[m], marker="o", markersize=3, linewidth=1.2, label=NAME[m])
    axes[1].plot(ks, [S["shots"][f"{m}_{k}"]["AA"] for k in ks], color=COL[m], marker="o", markersize=3, linewidth=1.2)
axes[0].axhline(U0, color=NEUTRAL, linewidth=0.7, linestyle="--")
axes[1].axhline(S["S1"]["none"]["AA"], color=NEUTRAL, linewidth=0.7, linestyle="--")
for ax, yl in zip(axes, ["$U$ (%)", "AA (%)"]):
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25], ["5", "10", "25"])
    ax.minorticks_off()
    ax.set_xlabel("Videos per class per task")
    ax.set_ylabel(yl)
    style(ax)
axes[0].legend(frameon=False, fontsize=5.8, loc="lower left")
save(fig, "fig_shots")

# ---- drift vs change in U
d = json.load(open(f"{RAD}/analysis_f/drift.json"))
fig, ax = plt.subplots(figsize=(3.5, 2.2))
fam_col = lambda n: MAGENTA if n.startswith("realkd") and "lwf" not in n else BLUE if n.startswith("realkd_lwf") else YELLOW if n.startswith("featkd") and "lwf" not in n else NEUTRAL  # noqa: E731
xs, ys = [], []
for n, x, y in d["points"]:
    ax.scatter(x, y, s=13, color=fam_col(n), edgecolor="white", linewidth=0.4, zorder=3)
    xs.append(x); ys.append(y)
b = np.polyfit(xs, ys, 1)
xx = np.linspace(0, max(xs), 50)
ax.plot(xx, np.polyval(b, xx), color=INK2, linewidth=0.7, linestyle=":")
for lab, c in [("GA ($\\lambda$ sweep)", MAGENTA), ("GA+LwF", BLUE), ("Sym. KD ($\\lambda$ sweep)", YELLOW), ("other methods", NEUTRAL)]:
    ax.scatter([], [], s=13, color=c, label=lab)
ax.set_xlabel("Mean genuine-feature drift $1-\\cos(z, z_{\\mathrm{base}})$")
ax.set_ylabel("$U - U_{\\mathrm{no\\,update}}$ (points)")
ax.legend(frameon=False, loc="lower left", fontsize=5.8)
ax.text(0.98, 0.95, f"Spearman $\\rho$ = {d['corr_genuine_drift_vs_dU']['spearman']:.2f}, n = {d['corr_genuine_drift_vs_dU']['n']}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6, color=INK2)
style(ax, "both")
save(fig, "fig_drift")
print("ok")
