"""J2 blocks P and S (registration/journal_J2.md): per-manipulation breakdown, Holm correction of the 18
registered tests, test-split intervals for AA. CPU only; reads saved predictions. Writes analysis_j2/stats.json."""

import json
import os
import zlib
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_c1 import U_from, boot_indices
from analyze_j1 import load

R = "runs"
F = "analysis_f"
OUT = "analysis_j2"
T = ["FF", "DFDCP", "DFD", "CDF"]
M32 = ["none", "head", "seq", "ewc", "replay", "lwf", "featkd", "featkd_lwf", "realkd", "realkd_lwf"]
S1PATH = {m: f"{R}/c1/{m}_seed0" for m in ["none", "seq", "lwf", "featkd", "realkd", "realkd_lwf"]}
S1PATH |= {m: f"{R}/j1_s1/{m}_seed0" for m in ["ewc", "lnmerge", "head", "featkd_lwf", "replay", "fakekd", "randkd"]}
TESTS = [("C-H1", "S1", "realkd_lwf", "lwf"), ("C-H2", "S1", "realkd_lwf", "seq"), ("C-H3", "S1", "realkd", "featkd"),
         ("M1", "S1", "realkd", "randkd"), ("M2", "S1", "realkd", "fakekd"), ("A", "S1", "realkd_lwf", "featkd_lwf")]
TESTS += [(f"{r}-{s}", s, a, b) for s in ["S2", "S3", "PE"] for r, a, b in
          [("R1", "realkd_lwf", "lwf"), ("R2", "realkd_lwf", "seq"), ("R3", "realkd", "featkd"), ("R4", "realkd_lwf", "featkd_lwf")]]
TAG = {"S2": "j1_S2", "S3": "j1_S3", "PE": "j1_pe"}


def path(setting, m):
    return S1PATH[m] if setting == "S1" else f"{R}/{TAG[setting]}/{m}_seed0"


def preds32(m, t):
    d = json.load(open(f"{F}/preds32_{m}_{t}.json"))
    return d["video"], np.array(d["label"]), np.array(d["p_fake"])


# ---------------------------------------------------------------- P: per-manipulation breakdown
def per_manipulation():
    out = {}
    for m in M32:
        out[m] = {}
        for t, groups in [("FF", ["DF", "F2F", "FS", "NT"]), ("DFDCP", ["method_A", "method_B"])]:
            v, y, p = preds32(m, t)
            src = np.array([x.rstrip("/").split("/")[-2] for x in v])
            for g in groups:
                sel = (y == 0) | (src == g)
                out[m][f"{t}:{g}"] = 100 * roc_auc_score(y[sel], p[sel])
    summ = json.load(open(f"{R}/j1_summary.json"))
    for m in summ["S1"]:
        out.setdefault(m, {})["FSh_probe"] = summ["S1"][m]["FSh"]
    return out


# ---------------------------------------------------------------- S1: Holm over the 18 registered tests
def one_test(args):
    name, setting, a, b, B = args
    PA, PB = load(path(setting, a))["preds"], load(path(setting, b))["preds"]
    for k in PA:
        assert PA[k]["video"] == PB[k]["video"], k
    rng = np.random.default_rng([20261006, zlib.crc32(name.encode())])
    d = np.array([U_from(PA, ix) - U_from(PB, ix) for ix in (boot_indices(PA, rng) for _ in range(B))])
    return name, {"setting": setting, "a": a, "b": b, "diff": float(U_from(PA) - U_from(PB)),
                  "p": float((1 + (d <= 0).sum()) / (B + 1)), "ci95": np.percentile(d, [2.5, 97.5]).tolist(),
                  "min_boot": float(d.min())}


def holm(res, alpha=0.05):
    names = sorted(res, key=lambda n: res[n]["p"])
    k = len(names)
    stop = False
    for i, n in enumerate(names):
        thr = alpha / (k - i)
        res[n]["holm_threshold"] = thr
        res[n]["p_holm"] = min(1.0, max(res[m]["p"] * (k - j) for j, m in enumerate(names[:i + 1])))
        if not stop and res[n]["p"] <= thr:
            res[n]["holm_reject"] = True
        else:
            stop = True
            res[n]["holm_reject"] = False
    return res


# ---------------------------------------------------------------- S2: test-split AA intervals
def aa_intervals(reps=2000):
    data = {m: {t: preds32(m, t) for t in T} for m in M32}
    for m in M32:
        for t in T:
            assert data[m][t][0] == data["realkd_lwf"][t][0]
    rng = np.random.default_rng(20261006)
    idx = []
    for _ in range(reps):
        it = {}
        for t in T:
            y = data["realkd_lwf"][t][1]
            it[t] = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in (0, 1)])
        idx.append(it)

    def aa(m, ix=None):
        return 100 * np.mean([roc_auc_score(data[m][t][1][ix[t] if ix else slice(None)], data[m][t][2][ix[t] if ix else slice(None)]) for t in T])
    base = [aa("realkd_lwf", ix) for ix in idx]
    out = {}
    for m in M32:
        if m == "realkd_lwf":
            continue
        d = np.array(base) - np.array([aa(m, ix) for ix in idx])
        out[m] = {"AA": aa(m), "diff": aa("realkd_lwf") - aa(m), "ci95": np.percentile(d, [2.5, 97.5]).tolist()}
    out["realkd_lwf"] = {"AA": aa("realkd_lwf")}
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = {"per_manipulation": per_manipulation()}
    print("per-manipulation done", flush=True)
    with Pool(8) as pool:
        tests = dict(pool.map(one_test, [(*t, 10000) for t in TESTS]))
    res["holm"] = holm(tests)
    for n, x in res["holm"].items():
        print(f"{n:8s} {x['a']}>{x['b']} ({x['setting']}): {x['diff']:+.2f} p={x['p']:.5f} p_holm={x['p_holm']:.4f} reject={x['holm_reject']}")
    res["aa_intervals"] = aa_intervals()
    for m, x in res["aa_intervals"].items():
        print(m, {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 2)) for k, v in x.items()})
    json.dump(res, open(f"{OUT}/stats.json", "w"), indent=1)
    print("STATS DONE")
