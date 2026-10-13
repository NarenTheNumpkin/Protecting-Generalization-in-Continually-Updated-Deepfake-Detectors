"""J1 block F: inference-only analyses on saved checkpoints (no training, no selection).

1. Feature drift: for a fixed probe set (genuine faces, seen fakes, stream fakes, unseen fakes), mean angular drift
   1 - cos(z_model, z_base) of every S1 run's final model; correlated with the change in U.
2. Stage-wise unseen score: U after every update for the main methods (intermediate checkpoints after_*.pt).
3. 32-frame rescoring of final models on the four test splits, per-video predictions saved (bootstrap CIs, ECE).
4. Calibration: expected calibration error (15 bins) on test splits and panel; genuine FPR@0.5.
5. Features of base / seq / GAD+LwF / symmetric+LwF for a t-SNE figure.
Writes analysis_f/*.json|npz.
"""

import paths
import json
import os
import random

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from common import FrameSet, embed, frames, load_model, load_registration, panel_sets, score, videos

R = "runs"
OUT = "analysis_f"
INIT = paths.INIT_CKPT
T = ["FF", "DFDCP", "DFD", "CDF"]
GENS = ["inpainting", "insight", "text2img"]
os.makedirs(OUT, exist_ok=True)

S1_RUNS = {m: f"{R}/c1/{m}_seed0" for m in ["seq", "lwf", "featkd", "realkd", "realkd_lwf"]}
S1_RUNS |= {m: f"{R}/j1_s1/{m}_seed0" for m in ["ewc", "lnmerge", "head", "featkd_lwf", "replay", "fakekd", "randkd"]}
LAM_RUNS = {os.path.basename(p): f"{R}/j1_lam/{p}" for p in (sorted(os.listdir(f"{R}/j1_lam")) if os.path.isdir(f"{R}/j1_lam") else [])}


def final_ckpt(run):
    a = json.load(open(f"{run}/args.json"))
    return f"{run}/after_{a['stream'].split(',')[-1]}.pt"


def model_from(state_path, base):
    sd = torch.load(state_path, map_location="cuda")
    base.load_state_dict(base_state, strict=True)  # reset to init, then apply the run's trainable params
    base.load_state_dict(sd, strict=False)
    return base.eval()


@torch.no_grad()
def feats(model, items, batch=128):
    dl = DataLoader(FrameSet(items, model.feature_extractor.preprocess, False), batch_size=batch, num_workers=8)
    Z, P = [], []
    for x, _, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, z = embed(model, x.cuda(non_blocking=True))
        Z.append(z.float().cpu())
        P.append(lg.float().softmax(-1)[:, 1].cpu())
    return torch.cat(Z), torch.cat(P)


def ece(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    e = 0.0
    edges = np.linspace(0.5, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def probe_items(reg):
    """Fixed probe set: 4 evenly spaced frames per video/sequence; seeded subsamples."""
    rng = random.Random(20261005)
    ps = panel_sets()

    def pick(vl, n):
        vl = sorted(vl)
        return rng.sample(vl, min(n, len(vl)))
    groups = {
        "genuine_FF": pick(videos(reg, "FF", "test", 0), 140),
        "genuine_WDF": pick([v for v in ps["WDF_aligned"] if v[1] == 0], 120),
        "genuine_DFF": pick([v for v in ps["DFF_inpainting"] if v[1] == 0], 300),
        "fake_FF": pick(videos(reg, "FF", "test", 1), 140),
        "fake_stream": pick(videos(reg, "DFDCP", "test", 1), 60) + pick(videos(reg, "DFD", "test", 1), 60) + pick(videos(reg, "CDF", "test", 1), 60),
        "fake_WDF": pick([v for v in ps["WDF_aligned"] if v[1] == 1], 120),
        "fake_DFF": sum((pick([v for v in ps[f"DFF_{g}"] if v[1] == 1], 100) for g in GENS), []),
    }
    items, gid = [], []
    for g, vl in groups.items():
        it = frames(vl, 4)
        items += it
        gid += [g] * len(it)
    return items, np.array(gid)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.9)
    reg = load_registration()
    base = load_model(INIT)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    summ = json.load(open(f"{R}/j1_summary.json"))

    # ---------------- 1. feature drift on the probe set
    items, gid = probe_items(reg)
    z0, p0 = feats(base, items)
    drift, keep = {}, {"base": z0.numpy().astype(np.float16)}
    for name, run in {**S1_RUNS, **LAM_RUNS}.items():
        if not os.path.exists(final_ckpt(run)):
            continue
        z, p = feats(model_from(final_ckpt(run), base), items)
        d = 1 - (z * z0).sum(1).numpy()
        drift[name] = {g: float(d[gid == g].mean()) for g in np.unique(gid)}
        drift[name]["genuine_all"] = float(d[np.char.startswith(gid.astype(str), "genuine")].mean())
        drift[name]["fake_unseen"] = float(d[np.isin(gid, ["fake_WDF", "fake_DFF"])].mean())
        if name in ("seq", "realkd_lwf", "featkd_lwf", "lwf"):
            keep[name] = z.numpy().astype(np.float16)
    np.savez_compressed(f"{OUT}/probe_features.npz", gid=gid, **keep)
    # correlation with change in U (S1 sequential runs incl. sweeps)
    U0 = summ["S1"]["none"]["U"]
    xs, ys, labs = [], [], []
    for name in drift:
        if name in summ["S1"]:
            u = summ["S1"][name]["U"]
        else:  # lambda sweep run names, e.g. realkd_lam3 -> summary["lambda"]["realkd"]["3"]
            fam, lab = (name.split("_lam")[0], name.split("_lam")[1]) if "_lam" in name and not name.startswith("realkd_lwf") else ("realkd_lwf", name.replace("realkd_lwf_", ""))
            u = summ["lambda"].get(fam, {}).get(lab, {}).get("U")
        if u is not None:
            xs.append(drift[name]["genuine_all"]); ys.append(u - U0); labs.append(name)
    corr = {"n": len(xs)}
    if len(xs) >= 4:
        corr |= {"spearman": spearmanr(xs, ys)[0], "pearson": pearsonr(xs, ys)[0]}
    json.dump({"drift": drift, "corr_genuine_drift_vs_dU": corr, "points": list(zip(labs, xs, ys))}, open(f"{OUT}/drift.json", "w"), indent=1)
    print("drift done", corr, flush=True)

    # ---------------- 2. stage-wise unseen score for main methods
    ps = panel_sets()
    stagewise = {}
    for m in ["seq", "lwf", "featkd", "realkd", "featkd_lwf", "realkd_lwf"]:
        run = S1_RUNS[m]
        stagewise[m] = {}
        for task in ["DFDCP", "DFD"]:
            mdl = model_from(f"{run}/after_{task}.pt", base)
            w = score(mdl, ps["WDF_aligned"], per_video=None)["auroc"]
            dff = np.mean([score(mdl, ps[f"DFF_{g}"], per_video=None)["auroc"] for g in GENS])
            stagewise[m][task] = {"U": 100 * (w + dff) / 2, "WDF": 100 * w, "DFF": 100 * dff}
        print("stagewise", m, stagewise[m], flush=True)
    json.dump(stagewise, open(f"{OUT}/stagewise_U.json", "w"), indent=1)

    # ---------------- 3/4. 32-frame rescoring of final models + calibration
    rescore = {}
    targets = {"none": None, **{m: S1_RUNS[m] for m in ["seq", "lwf", "featkd", "realkd", "featkd_lwf", "realkd_lwf", "ewc", "head", "replay"]}}
    for m, run in targets.items():
        mdl = base.eval() if run is None else model_from(final_ckpt(run), base)
        if run is None:
            mdl.load_state_dict(base_state, strict=True)
        rescore[m] = {}
        for t in T:
            met, pr = score(mdl, videos(reg, t, "test"), per_video=None, return_preds=True)
            met["ece"] = ece(pr["label"], pr["p_fake"])
            rescore[m][t] = met
            json.dump(pr, open(f"{OUT}/preds32_{m}_{t}.json", "w"))
        rescore[m]["AA"] = float(np.mean([rescore[m][t]["auroc"] for t in T]) * 100)
        print("rescore32", m, {t: round(rescore[m][t]["auroc"] * 100, 2) for t in T}, "AA", round(rescore[m]["AA"], 2), flush=True)
    # panel ECE from saved panel predictions (no inference needed)
    for m, run in {**targets, "none": f"{R}/c1/none_seed0"}.items():
        pp = json.load(open(f"{run}/panel_preds.json"))
        rescore.setdefault(m, {})["panel_ece"] = {k: ece(v["label"], v["p_fake"]) for k, v in pp.items()}
    json.dump(rescore, open(f"{OUT}/rescore32_calibration.json", "w"), indent=1)
    print("ALL ANALYSES DONE", flush=True)
