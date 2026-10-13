"""J4 blocks B and C (registration/journal_J4.md), inference only, S1 seed-0 final models.

B: 32-frame predictions on the development splits of the four trained datasets (thresholds for low-FPR operating
   points) and 32-frame test predictions for GPL (the other models have them from J1 block F).
C: 8-frame test predictions under the mask-based protocol (as used so far) and under label-independent largest-face
   crops for FF++ and DFD fakes; DFDC-P and Celeb-DF are scored once (no masks are involved there).
Each batch is decoded once and scored by every plain model (trainable parameters swapped in place); GPL, which wraps the
encoder input, is scored in a separate pass. Writes analysis_j4/*.json."""

import paths
import json
import os
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import FrameSet, embed, frames, load_model, load_registration, videos
from stream import setup_gpl

INIT = paths.INIT_CKPT
W = paths.DATASETS
OUT = "analysis_j4"
T = ["FF", "DFDCP", "DFD", "CDF"]
PLAIN = {"seq": "runs/c1/seq_seed0", "lwf": "runs/c1/lwf_seed0", "featkd": "runs/c1/featkd_seed0", "realkd": "runs/c1/realkd_seed0",
         "realkd_lwf": "runs/c1/realkd_lwf_seed0", "head": "runs/j1_s1/head_seed0", "ewc": "runs/j1_s1/ewc_seed0",
         "replay": "runs/j1_s1/replay_seed0", "featkd_lwf": "runs/j1_s1/featkd_lwf_seed0"}
GPL = "runs/j3_s1/gpl_seed0"


def score_many(model, states, vl, per_video):
    """{model_name: per-video mean fake probability} for video list vl."""
    items = frames(vl, per_video)
    vid = np.array([v for _, _, v in items])
    params = {n: p for n, p in model.named_parameters() if n in next(iter(states.values()))}
    dl = DataLoader(FrameSet(items, model.feature_extractor.preprocess, False), batch_size=128, num_workers=16)
    P = {k: np.zeros(len(items), dtype=np.float32) for k in states}
    k0 = 0
    with torch.no_grad():
        for x, _, _ in dl:
            x = x.cuda(non_blocking=True)
            for k, sd in states.items():
                for n, p in params.items():
                    p.copy_(sd[n])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = embed(model, x)
                P[k][k0:k0 + len(x)] = lg.float().softmax(-1)[:, 1].cpu().numpy()
            k0 += len(x)
    cnt = np.bincount(vid, minlength=len(vl))
    return {k: (np.bincount(vid, weights=P[k], minlength=len(vl)) / cnt).tolist() for k in states}


def dump(name, vl, preds):
    json.dump({"video": [d for d, _ in vl], "label": [l for _, l in vl], "p_fake": preds}, open(f"{OUT}/{name}.json", "w"))


def largest_face(vl, task):
    out = []
    for d, l in vl:
        if l == 1 and task == "FF":
            m, v = d.rstrip("/").split("/")[-2:]
            d = f"{W}/FF_largestface/{m}/{v}"
        elif task == "DFD":
            d = d.replace("DFD_masked", "DFD")
        assert os.path.isdir(d) and os.listdir(d), d
        out.append((d, l))
    return out


def run(model, states, tag):
    reg = load_registration()
    t0 = time.time()
    # ---- B: development splits, 32 frames
    for t in T:
        vl = videos(reg, t, "dev")
        for k, p in score_many(model, states, vl, 32).items():
            dump(f"devpreds32_{k}_{t}", vl, p)
        print(tag, "dev", t, round(time.time() - t0), "s", flush=True)
    # ---- C: test splits, 8 frames, mask-based and largest-face
    for t in T:
        vl = videos(reg, t, "test")
        for k, p in score_many(model, states, vl, 8).items():
            dump(f"test8_mask_{k}_{t}", vl, p)
        if t in ("FF", "DFD"):
            while t == "FF" and "PREPROCESS_DONE" not in open(os.path.join(paths.GEND_ROOT, "work", "preprocess_ff_largestface.log")).read():
                time.sleep(60)
            vl2 = largest_face(vl, t)
            for k, p in score_many(model, states, vl2, 8).items():
                dump(f"test8_largest_{k}_{t}", vl2, p)
        print(tag, "test8", t, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.42)
    reg = load_registration()
    plain = load_model(INIT).eval()
    states = {"none": {n: p.detach().clone() for n, p in plain.named_parameters() if p.requires_grad}}
    for k, r in PLAIN.items():
        sd = torch.load(f"{r}/after_CDF.pt", map_location="cuda")
        states[k] = {n: sd[n] if n in sd else states["none"][n] for n in states["none"]}  # head-only runs store head params only
    run(plain, states, "plain")
    del plain
    torch.cuda.empty_cache()
    gm = setup_gpl(load_model(INIT), types.SimpleNamespace(gpl_lam_w=1.0)).eval()
    gstates = {"gpl": torch.load(f"{GPL}/after_CDF.pt", map_location="cuda")}
    run(gm, gstates, "gpl")
    for t in T:  # B: GPL 32-frame test predictions (the other models have them from J1 block F)
        vl = videos(reg, t, "test")
        dump(f"preds32_gpl_{t}", vl, score_many(gm, gstates, vl, 32)["gpl"])
    print("J4 INFER DONE", flush=True)
