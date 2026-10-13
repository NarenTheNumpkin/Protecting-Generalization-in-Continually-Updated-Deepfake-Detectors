"""J2 block R (registration/journal_J2.md): post-update recalibration with current-task data only.

For each final S1 model: score 25 real + 25 fake Celeb-DF dev videos (32 frames), fit a bias-only and a Platt map
on the video-level fake probability by NLL, and evaluate ECE / ACC@0.5 / genuine FPR@0.5 on the saved 32-frame
test predictions (J1 block F) and on the unseen-panel predictions. Writes analysis_j2/recal.json."""

import paths
import json
import random

import numpy as np
import torch
from scipy.optimize import minimize

from analysis_f import ece
from common import load_model, load_registration, score, videos
from j2_stats import M32, S1PATH, T

INIT = paths.INIT_CKPT
PANEL = ["WDF_aligned", "DFF_inpainting", "DFF_insight", "DFF_text2img"]
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit(p, y, slope):
    z, y = logit(p), np.asarray(y)

    def nll(w):
        a, b = (w[0], w[1]) if slope else (1.0, w[0])
        s = a * z + b
        return np.mean(np.logaddexp(0, s) - y * s)
    w = minimize(nll, [1.0, 0.0] if slope else [0.0], method="Nelder-Mead").x
    return (float(w[0]), float(w[1])) if slope else (1.0, float(w[0]))


def apply(p, ab):
    return 1 / (1 + np.exp(-(ab[0] * logit(p) + ab[1])))


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"ece": ece(y, p), "acc": float(((p > 0.5) == y).mean() * 100), "fpr": float((p[y == 0] > 0.5).mean() * 100),
            "tpr": float((p[y == 1] > 0.5).mean() * 100)}


if __name__ == "__main__":
    reg = load_registration()
    rng = random.Random(20261006)
    real, fake = sorted(videos(reg, "CDF", "dev", 0)), sorted(videos(reg, "CDF", "dev", 1))
    fit_set = set(v for v, _ in videos(reg, "CDF", "fit")) | set(v for v, _ in videos(reg, "CDF", "test"))
    cal = rng.sample(real, 25) + rng.sample(fake, 25)
    assert not (set(v for v, _ in cal) & fit_set)
    model = load_model(INIT)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    out = {"calibration_videos": [v for v, _ in cal]}
    for m in M32:
        model.load_state_dict(base_state, strict=True)
        if m != "none":
            model.load_state_dict(torch.load(f"{S1PATH[m]}/after_CDF.pt", map_location="cuda"), strict=False)
        _, pr = score(model, cal, per_video=32, return_preds=True)
        maps = {"raw": (1.0, 0.0), "bias": fit(pr["p_fake"], pr["label"], False), "platt": fit(pr["p_fake"], pr["label"], True)}
        panel = json.load(open(f"{S1PATH[m]}/panel_preds.json"))
        res = {"maps": maps}
        for k, ab in maps.items():
            r = {}
            for t in T:
                d = json.load(open(f"analysis_f/preds32_{m}_{t}.json"))
                r[t] = metrics(d["label"], apply(d["p_fake"], ab))
            for s in PANEL:
                r[s] = metrics(panel[s]["label"], apply(panel[s]["p_fake"], ab))
            r["test_mean"] = {q: float(np.mean([r[t][q] for t in T])) for q in ["ece", "acc", "fpr", "tpr"]}
            r["panel_mean"] = {q: float(np.mean([r[s][q] for s in PANEL])) for q in ["ece", "acc", "fpr", "tpr"]}
            res[k] = r
        out[m] = res
        print(m, {k: (np.round(maps[k], 3).tolist(), round(res[k]["test_mean"]["ece"], 3), round(res[k]["test_mean"]["acc"], 1),
                      round(res[k]["panel_mean"]["ece"], 3), round(res[k]["panel_mean"]["acc"], 1)) for k in maps}, flush=True)
    json.dump(out, open("analysis_j2/recal.json", "w"), indent=1)
    print("RECAL DONE")
