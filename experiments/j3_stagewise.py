"""J3 block L, descriptive: unseen-panel score U after every update of the nine-update stream (post hoc, from the saved
checkpoints; WildDeepfake at 8 frames per sequence and all DeepFakeFace images, as in J2). Each batch is loaded once
and scored by every checkpoint. Writes analysis_j3/stagewise_long.json."""

import paths
import json
import os
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import FrameSet, embed, frames, load_model, panel_sets
from j2_analyze_robust import U
from stream import setup_gpl

R = "runs/j3_long"
INIT = paths.INIT_CKPT
SETS = ["WDF_aligned", "DFF_inpainting", "DFF_insight", "DFF_text2img"]
TOKENS = json.load(open(f"{R}/realkd_lwf_seed0/args.json"))["stream"].split(",")[1:]


def score_states(model, states, items, owner, ps):
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
    out = {}
    o_set = np.array([s for s, _ in owner]); o_vid = np.array([v for _, v in owner])
    for k in states:
        preds = {}
        for s in SETS:
            sel = o_set == s
            vp = np.bincount(o_vid[sel], weights=P[k][sel], minlength=len(ps[s])) / np.bincount(o_vid[sel], minlength=len(ps[s]))
            preds[s] = {"video": [d for d, _ in ps[s]], "label": [l for _, l in ps[s]], "p_fake": vp.tolist()}
        out[k] = U(preds)
    return out


if __name__ == "__main__":
    os.makedirs("analysis_j3", exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.42)
    ps = panel_sets()
    items, owner = [], []
    for s in SETS:
        it = frames(ps[s], 8 if s.startswith("WDF") else None)
        items += it
        owner += [(s, vi) for _, _, vi in it]
    res = {}
    plain = load_model(INIT).eval()
    states = {("none", "FF"): {n: p.detach().clone() for n, p in plain.named_parameters() if p.requires_grad}}
    for m in ["seq", "lwf", "featkd_lwf", "realkd", "realkd_lwf"]:
        for t in TOKENS:
            states[(m, t)] = torch.load(f"{R}/{m}_seed0/after_{t}.pt", map_location="cuda")
    res.update(score_states(plain, states, items, owner, ps))
    del plain
    gm = setup_gpl(load_model(INIT), types.SimpleNamespace(gpl_lam_w=1.0)).eval()
    gstates = {("gpl", t): torch.load(f"{R}/gpl_seed0/after_{t}.pt", map_location="cuda") for t in TOKENS}
    res.update(score_states(gm, gstates, items, owner, ps))
    out = {"tokens": ["FF"] + TOKENS, "U": {}}
    for (m, t), u in res.items():
        out["U"].setdefault(m, {})[t] = u
    json.dump(out, open("analysis_j3/stagewise_long.json", "w"), indent=1)
    for m, d in out["U"].items():
        print(m, {t: round(v, 2) for t, v in d.items()})
