"""J2 block RB analysis: U per condition and model, U_deg = mean over the 13 degraded conditions, hypotheses RB1-RB3
with paired cluster bootstrap (2000 resamples; indices shared across conditions). Writes analysis_j2/robust_summary.json."""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from j2_robust import CONDS, OUT

GENS = ["inpainting", "insight", "text2img"]
DEG = [c for c in CONDS if c != "clean"]
TESTS = [("RB1", "realkd_lwf", "lwf"), ("RB2", "realkd_lwf", "seq"), ("RB3", "realkd", "featkd")]


def boot_indices(preds, rng):
    """C1 cluster scheme restricted to the sets scored here (WDF by source archive within class; DFF reals shared)."""
    idx = {}
    v = preds["WDF_aligned"]["video"]; y = np.array(preds["WDF_aligned"]["label"])
    arc = np.array([os.path.basename(x).split("_")[0] + f"/{lab}" for x, lab in zip(v, y)])
    take = []
    for lab in [0, 1]:
        cl = np.unique(arc[y == lab])
        for c in rng.choice(cl, len(cl), replace=True):
            take.append(np.flatnonzero(arc == c))
    idx["WDF_aligned"] = np.concatenate(take)
    real_names = [os.path.basename(x) for x, l in zip(preds["DFF_inpainting"]["video"], preds["DFF_inpainting"]["label"]) if l == 0]
    real_pick = rng.choice(real_names, len(real_names), replace=True)
    for g in GENS:
        vv = [os.path.basename(x) for x in preds[f"DFF_{g}"]["video"]]; yy = np.array(preds[f"DFF_{g}"]["label"])
        pos = {n: i for i, (n, l) in enumerate(zip(vv, yy)) if l == 0}
        fakes = np.flatnonzero(yy == 1)
        idx[f"DFF_{g}"] = np.concatenate([[pos[n] for n in real_pick], rng.choice(fakes, len(fakes), replace=True)])
    return idx


def U(preds, ix=None):
    def auc(n):
        y, p = np.array(preds[n]["label"]), np.array(preds[n]["p_fake"])
        return roc_auc_score(y[ix[n]], p[ix[n]]) if ix is not None else roc_auc_score(y, p)
    return 100 * (auc("WDF_aligned") + np.mean([auc(f"DFF_{g}") for g in GENS])) / 2


if __name__ == "__main__":
    R = {c: json.load(open(f"{OUT}/robust_{c}.json")) for c in CONDS}
    models = list(R["clean"])
    table = {m: {c: U(R[c][m]) for c in CONDS} for m in models}
    for m in models:
        table[m]["U_deg"] = float(np.mean([table[m][c] for c in DEG]))
        table[m]["parts"] = {c: {"WDF": 100 * roc_auc_score(R[c][m]["WDF_aligned"]["label"], R[c][m]["WDF_aligned"]["p_fake"]),
                                 **{g: 100 * roc_auc_score(R[c][m][f"DFF_{g}"]["label"], R[c][m][f"DFF_{g}"]["p_fake"]) for g in GENS}}
                             for c in CONDS}
    print(f"{'model':11s} " + " ".join(f"{c[:9]:>9s}" for c in CONDS) + "     U_deg")
    for m in models:
        print(f"{m:11s} " + " ".join(f"{table[m][c]:9.2f}" for c in CONDS) + f" {table[m]['U_deg']:9.2f}")
    ref = R["clean"]["none"]
    for c in CONDS:
        for m in models:
            for k in ref:
                assert R[c][m][k]["video"] == ref[k]["video"]
    rng = np.random.default_rng(20261006)
    tests = {}
    diffs = {t: [] for t, _, _ in TESTS}
    for _ in range(2000):
        ix = boot_indices(ref, rng)
        u = {m: np.mean([U(R[c][m], ix) for c in DEG]) for m in {x for _, a, b in TESTS for x in (a, b)}}
        for t, a, b in TESTS:
            diffs[t].append(u[a] - u[b])
    for t, a, b in TESTS:
        d = table[a]["U_deg"] - table[b]["U_deg"]
        ci = np.percentile(diffs[t], [2.5, 97.5]).tolist()
        tests[t] = {"a": a, "b": b, "diff": d, "ci": ci, "confirmed": bool(d > 0 and ci[0] > 0)}
        print(f"{t} U_deg({a}) - U_deg({b}) = {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] {'CONFIRMED' if tests[t]['confirmed'] else 'NOT CONFIRMED'}")
    json.dump({"table": table, "tests": tests, "conds": CONDS}, open(f"{OUT}/robust_summary.json", "w"), indent=1)
