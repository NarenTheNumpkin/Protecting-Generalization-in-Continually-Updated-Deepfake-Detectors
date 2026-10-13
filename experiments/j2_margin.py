"""J2 post-hoc descriptive check of Proposition 2 (not registered): for genuine/manipulated pairs of the unseen panel that
the model before updates ranks correctly, the fraction misranked after the updates, as a function of the base logit
margin s(u) - s(r). Uses saved panel predictions only. Writes analysis_j2/margin.json."""
import json

import numpy as np

from j2_stats import S1PATH

SETS = ["WDF_aligned", "DFF_inpainting", "DFF_insight", "DFF_text2img"]
EDGES = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, np.inf])


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


base = json.load(open(f"{S1PATH['none']}/panel_preds.json"))
out = {"edges": EDGES[:-1].tolist(), "models": {}}
for m in ["seq", "lwf", "featkd", "realkd", "featkd_lwf", "realkd_lwf", "ewc", "lnmerge", "head"]:
    pp = json.load(open(f"{S1PATH[m]}/panel_preds.json"))
    flips, tot = np.zeros(len(EDGES) - 1), np.zeros(len(EDGES) - 1)
    for s in SETS:
        y = np.array(base[s]["label"]); s0 = logit(base[s]["p_fake"]); s1 = logit(pp[s]["p_fake"])
        assert pp[s]["video"] == base[s]["video"]
        d0 = s0[y == 1][:, None] - s0[y == 0][None, :]
        d1 = s1[y == 1][:, None] - s1[y == 0][None, :]
        ok = d0 > 0
        b = np.digitize(d0[ok], EDGES) - 1
        f = (d1[ok] <= 0)
        flips += np.bincount(b, weights=f, minlength=len(EDGES) - 1)[: len(EDGES) - 1]
        tot += np.bincount(b, minlength=len(EDGES) - 1)[: len(EDGES) - 1]
    out["models"][m] = {"flip_rate": (flips / tot).tolist(), "n": tot.tolist(), "overall": float(flips.sum() / tot.sum())}
    print(m, np.round(100 * flips / tot, 2), "overall", round(100 * flips.sum() / tot.sum(), 2))
json.dump(out, open("analysis_j2/margin.json", "w"), indent=1)
