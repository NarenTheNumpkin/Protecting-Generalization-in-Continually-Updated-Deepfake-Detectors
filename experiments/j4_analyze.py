"""J4 analysis (registration/journal_J4.md): blocks A (matched plasticity), B (low-FPR operating points), C (label-independent
face selection), D (retention by family, non-inferiority), E (identity-aware AA intervals). Writes analysis_j4/j4_summary.json."""

import json
import os
import re
from multiprocessing import Pool

import numpy as np
from scipy.stats import kendalltau, ttest_rel
from sklearn.metrics import roc_auc_score

from analyze_c1 import boot_indices
from analyze_j1 import load, metrics
from j3_analyze import holm

R = "runs"
OUT = "analysis_j4"
T = ["FF", "DFDCP", "DFD", "CDF"]
GENS = ["inpainting", "insight", "text2img"]
SETS = ["WDF_aligned"] + [f"DFF_{g}" for g in GENS]
SEED0 = {m: f"{R}/c1/{m}_seed0" for m in ["seq", "lwf", "featkd", "realkd", "realkd_lwf"]}
SEED0 |= {"featkd_lwf": f"{R}/j1_s1/featkd_lwf_seed0", "gpl": f"{R}/j3_s1/gpl_seed0", "none": f"{R}/c1/none_seed0",
          "head": f"{R}/j1_s1/head_seed0", "ewc": f"{R}/j1_s1/ewc_seed0", "replay": f"{R}/j1_s1/replay_seed0"}
M11 = ["none", "head", "seq", "ewc", "replay", "lwf", "featkd", "featkd_lwf", "realkd", "realkd_lwf", "gpl"]


def seed_path(m, s):
    return SEED0[m] if s == 0 else f"{R}/j3_seeds/{m}_seed{s}"


def jl(p):
    return json.load(open(p))


# ------------------------------------------------------------------ A: matched plasticity
def block_a():
    ch = jl("registration/J4_choice.json")
    out = {"choice": ch, "runs": {}, "tests": {}}

    def paths(fam):
        c = ch["chosen"][fam]
        if c["reuse_existing"]:
            return [seed_path(fam, s) for s in range(5)]
        name = c["setting"].replace("j3:", "")
        return [f"{R}/j4_matched/{name}_seed{s}" for s in range(5)]
    ref = [load(seed_path("realkd_lwf", s)) for s in range(5)]
    mref = [metrics(r) for r in ref]
    out["runs"]["realkd_lwf"] = {k: [x[k] for x in mref] for k in ["U", "AA", "gain", "FF_acc", "DFF_text2img"]}
    pv = {}
    for name, fam in [("MA1", "lwf"), ("MA2", "seq"), ("GA", "realkd")]:
        runs = [load(p) for p in paths(fam)]
        if not all(runs):
            print("  A pending", fam)
            continue
        mm = [metrics(r) for r in runs]
        out["runs"][fam] = {k: [x[k] for x in mm] for k in ["U", "AA", "gain", "FF_acc", "DFF_text2img"]}
        d = [a["U"] - b["U"] for a, b in zip(mref, mm)]
        alt = "two-sided" if name == "GA" else "greater"
        p = float(ttest_rel([a["U"] for a in mref], [b["U"] for b in mm], alternative=alt).pvalue)
        out["tests"][name] = {"family": fam, "setting": ch["chosen"][fam]["setting"], "diffs": d, "mean": float(np.mean(d)),
                              "sd": float(np.std(d, ddof=1)), "p": p, "n_pos": int(sum(x > 0 for x in d)),
                              "gain_ref": float(np.mean(out["runs"]["realkd_lwf"]["gain"])), "gain_fam": float(np.mean(out["runs"][fam]["gain"])),
                              "AA_ref": float(np.mean(out["runs"]["realkd_lwf"]["AA"])), "AA_fam": float(np.mean(out["runs"][fam]["AA"]))}
        if name != "GA":
            pv[name] = p
    for k, h in holm(pv).items():
        t = out["tests"][k]
        t |= {"p_holm": h["p_holm"], "confirmed": bool(t["mean"] > 0 and h["reject"])}
    for k, t in out["tests"].items():
        print(f"  {k} GA+LwF vs {t['family']} [{t['setting']}]: ΔU {t['mean']:+.2f}±{t['sd']:.2f}, positive {t['n_pos']}/5, p={t['p']:.4f}"
              + (f", p_holm={t['p_holm']:.4f} -> {'CONFIRMED' if t['confirmed'] else 'not confirmed'}" if "p_holm" in t else " (descriptive)")
              + f" | gain {t['gain_ref']:.2f} vs {t['gain_fam']:.2f} | AA {t['AA_ref']:.2f} vs {t['AA_fam']:.2f}")
    return out


# ------------------------------------------------------------------ B: low-FPR operating points
def block_b():
    out = {}
    for m in M11:
        dev = [jl(f"{OUT}/devpreds32_{m}_{t}.json") for t in T]
        g = np.concatenate([np.array(d["p_fake"])[np.array(d["label"]) == 0] for d in dev])
        res = {"n_dev_genuine": int(len(g))}
        for fpr in (0.05, 0.01):
            thr = float(np.quantile(g, 1 - fpr, method="higher"))
            r = {"threshold": thr, "test": {}, "panel": {}}
            for t in T:
                tp = f"{OUT}/preds32_gpl_{t}.json" if m == "gpl" else f"analysis_f/preds32_{m}_{t}.json"
                d = jl(tp)
                y, p = np.array(d["label"]), np.array(d["p_fake"])
                r["test"][t] = {"tpr": float((p[y == 1] > thr).mean() * 100), "fpr": float((p[y == 0] > thr).mean() * 100)}
            pp = jl(f"{SEED0[m]}/panel_preds.json")
            for s in SETS:
                y, p = np.array(pp[s]["label"]), np.array(pp[s]["p_fake"])
                r["panel"][s] = {"tpr": float((p[y == 1] > thr).mean() * 100), "fpr": float((p[y == 0] > thr).mean() * 100)}
            r["test_mean"] = {k: float(np.mean([r["test"][t][k] for t in T])) for k in ("tpr", "fpr")}
            r["panel_tpr_mean"] = float(np.mean([r["panel"][s]["tpr"] for s in SETS]))
            r["panel_fpr_wdf"], r["panel_fpr_dff"] = r["panel"]["WDF_aligned"]["fpr"], r["panel"]["DFF_inpainting"]["fpr"]
            res[f"fpr{int(fpr * 100)}"] = r
        out[m] = res
        r5 = res["fpr5"]
        print(f"  {m:11s} @5%dev: test TPR {r5['test_mean']['tpr']:.1f} FPR {r5['test_mean']['fpr']:.1f} | panel TPR "
              + " ".join(f"{s.replace('DFF_', '')[:6]} {r5['panel'][s]['tpr']:.1f}" for s in SETS)
              + f" | panel FPR wdf {r5['panel_fpr_wdf']:.1f} dff {r5['panel_fpr_dff']:.1f}")
    return out


# ------------------------------------------------------------------ C: label-independent face selection
def block_c():
    out = {"auroc": {}, "AA": {}}
    for m in M11:
        a = {}
        for t in T:
            d = jl(f"{OUT}/test8_mask_{m}_{t}.json")
            a[f"mask:{t}"] = 100 * roc_auc_score(d["label"], d["p_fake"])
            if t in ("FF", "DFD"):
                d2 = jl(f"{OUT}/test8_largest_{m}_{t}.json")
                a[f"largest:{t}"] = 100 * roc_auc_score(d2["label"], d2["p_fake"])
        out["auroc"][m] = a
        out["AA"][m] = {"mask": float(np.mean([a[f"mask:{t}"] for t in T])),
                        "largest": float(np.mean([a[f"largest:{t}"] if t in ("FF", "DFD") else a[f"mask:{t}"] for t in T]))}
        print(f"  {m:11s} FF {a['mask:FF']:.2f}->{a['largest:FF']:.2f}  DFD {a['mask:DFD']:.2f}->{a['largest:DFD']:.2f}  "
              f"AA {out['AA'][m]['mask']:.2f}->{out['AA'][m]['largest']:.2f}")
    upd = [m for m in M11 if m != "none"]
    for key in ("AA",):
        x = [out[key][m]["mask"] for m in upd]
        y = [out[key][m]["largest"] for m in upd]
        out["kendall_AA"] = float(kendalltau(x, y).statistic)
    out["kendall_DFD"] = float(kendalltau([out["auroc"][m]["mask:DFD"] for m in upd], [out["auroc"][m]["largest:DFD"] for m in upd]).statistic)
    out["kendall_FF"] = float(kendalltau([out["auroc"][m]["mask:FF"] for m in upd], [out["auroc"][m]["largest:FF"] for m in upd]).statistic)
    print(f"  Kendall tau of method ranking, mask vs largest: AA {out['kendall_AA']:.2f}, DFD {out['kendall_DFD']:.2f}, FF {out['kendall_FF']:.2f}")
    return out


# ------------------------------------------------------------------ D: retention by family, non-inferiority
def fam_scores(preds, ix=None):
    r = {}
    for s in SETS:
        y, p = np.array(preds[s]["label"]), np.array(preds[s]["p_fake"])
        if ix is not None:
            y, p = y[ix[s]], p[ix[s]]
        r[s] = 100 * roc_auc_score(y, p)
    r["U"] = (r["WDF_aligned"] + np.mean([r[f"DFF_{g}"] for g in GENS])) / 2
    return r


def _retention(args):
    m, B = args
    none = jl(f"{SEED0['none']}/panel_preds.json")
    runs = [jl(f"{seed_path(m, s)}/panel_preds.json") for s in range(5)]
    keys = SETS + ["U"]
    base0 = fam_scores(none)
    point_s0 = {k: fam_scores(runs[0])[k] - base0[k] for k in keys}
    point_5 = {k: float(np.mean([fam_scores(r)[k] for r in runs]) - base0[k]) for k in keys}
    rng = np.random.default_rng(20261011)
    d0, d5 = {k: [] for k in keys}, {k: [] for k in keys}
    for _ in range(B):
        ix = boot_indices(none, rng)
        bn = fam_scores(none, ix)
        sc = [fam_scores(r, ix) for r in runs]
        ss = rng.integers(0, 5, 5)
        for k in keys:
            d0[k].append(sc[0][k] - bn[k])
            d5[k].append(np.mean([sc[i][k] for i in ss]) - bn[k])
    return m, {"seed0": {k: {"diff": point_s0[k], "ci": np.percentile(d0[k], [2.5, 97.5]).tolist()} for k in keys},
               "five": {k: {"diff": point_5[k], "ci": np.percentile(d5[k], [2.5, 97.5]).tolist()} for k in keys}}


def block_d(B=10000):
    with Pool(4) as pool:
        res = dict(pool.map(_retention, [(m, B) for m in ["realkd_lwf", "lwf", "seq", "gpl"]]))
    out = {"per_method": res}
    for m, r in res.items():
        f5 = r["five"]
        worst = min(SETS, key=lambda s: f5[s]["diff"])
        print(f"  {m:11s} 5 seeds: ΔU {f5['U']['diff']:+.2f} [{f5['U']['ci'][0]:+.2f},{f5['U']['ci'][1]:+.2f}] | "
              + " ".join(f"{s.replace('DFF_', '')[:7]} {f5[s]['diff']:+.2f}" for s in SETS) + f" | worst {worst}")
    ga = res["realkd_lwf"]
    out["noninferiority"] = {k: {"lower": ga[k]["U"]["ci"][0], "margin": -1.0, "noninferior": bool(ga[k]["U"]["ci"][0] > -1.0)}
                             for k in ("seed0", "five")}
    print("  non-inferiority GA+LwF vs none on U (margin 1.0):", out["noninferiority"])
    return out


# ------------------------------------------------------------------ E: identity-aware AA intervals
def ids(task, path):
    v = path.rstrip("/").split("/")[-1]
    if task == "FF":
        return v.split("_")
    if task == "DFD":
        return v.split("__")[0].split("_")
    if task == "CDF":
        return [x for x in v.split("_") if x.startswith("id")] or [v]
    if task == "DFDCP":
        out = []
        for x in v.split("_"):
            if not x.isdigit() or len(x) < 5:
                break
            out.append(x)
        return out or [v]
    raise KeyError(task)


def clusters(task, vids):
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    toks = [ids(task, v) for v in vids]
    for tk in toks:
        for b in tk[1:]:
            parent[find(tk[0])] = find(b)
    comp = [find(tk[0]) for tk in toks]
    sizes = {c: comp.count(c) for c in set(comp)}
    mode = "components"
    if max(sizes.values()) > len(vids) / 2:
        comp, mode = [tk[0] for tk in toks], "first identifier"
    return np.array(comp), mode, len(set(comp))


def block_e(reps=2000):
    data = {}
    for m in M11:
        data[m] = {}
        for t in T:
            d = jl(f"{OUT}/preds32_gpl_{t}.json" if m == "gpl" else f"analysis_f/preds32_{m}_{t}.json")
            data[m][t] = (d["video"], np.array(d["label"]), np.array(d["p_fake"]))
    info, groups = {}, {}
    for t in T:
        v, y, _ = data["realkd_lwf"][t]
        for m in M11:
            assert data[m][t][0] == v, (m, t)
        comp, mode, n = clusters(t, v)
        info[t] = {"mode": mode, "n_clusters": n}
        groups[t] = {c: [np.flatnonzero((comp == k) & (y == c)) for k in np.unique(comp[y == c])] for c in (0, 1)}
    print("  identity clusters:", info)
    rng = np.random.default_rng(20261011)
    idx = []
    for _ in range(reps):
        it = {}
        for t in T:
            parts = []
            for c in (0, 1):
                g = groups[t][c]
                parts += [g[i] for i in rng.integers(0, len(g), len(g))]
            it[t] = np.concatenate(parts)
        idx.append(it)

    def aa(m, ix=None):
        return 100 * np.mean([roc_auc_score(data[m][t][1][ix[t]] if ix else data[m][t][1], data[m][t][2][ix[t]] if ix else data[m][t][2]) for t in T])
    ref = np.array([aa("realkd_lwf", ix) for ix in idx])
    out = {"clusters": info, "diffs": {}}
    for m in M11:
        if m == "realkd_lwf":
            continue
        d = ref - np.array([aa(m, ix) for ix in idx])
        out["diffs"][m] = {"diff": float(aa("realkd_lwf") - aa(m)), "ci": np.percentile(d, [2.5, 97.5]).tolist()}
        print(f"  AA GA+LwF - {m}: {out['diffs'][m]['diff']:+.2f} [{out['diffs'][m]['ci'][0]:+.2f}, {out['diffs'][m]['ci'][1]:+.2f}]")
    return out


if __name__ == "__main__":
    S = {}
    for name, fn in [("E", block_e), ("D", block_d), ("B", block_b), ("C", block_c), ("A", block_a)]:
        print(f"== block {name}", flush=True)
        try:
            S[name] = fn()
        except FileNotFoundError as e:
            print("  pending:", e)
    prev = json.load(open(f"{OUT}/j4_summary.json")) if os.path.exists(f"{OUT}/j4_summary.json") else {}
    json.dump(prev | S, open(f"{OUT}/j4_summary.json", "w"), indent=1, default=float)
