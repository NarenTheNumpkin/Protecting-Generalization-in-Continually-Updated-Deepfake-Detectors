"""J2 block RB (registration/journal_J2.md): unseen-panel scores of the final S1 models under image degradations.

Inference only. Each batch is loaded and degraded once and scored by every model; the models differ only in
their trainable LayerNorm/head parameters, which are swapped in place. Writes analysis_j2/robust_<cond>.json with,
per model and panel set, the per-unit predictions {video, label, p_fake} (the C1 panel_preds format).
"""

import paths
import io
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from common import embed, frames, load_model, panel_sets

R = "runs"
OUT = "analysis_j2"
INIT = paths.INIT_CKPT
RUNS = {m: f"{R}/c1/{m}_seed0" for m in ["seq", "lwf", "featkd", "realkd", "realkd_lwf"]}
RUNS |= {m: f"{R}/j1_s1/{m}_seed0" for m in ["featkd_lwf", "lnmerge", "ewc"]}
SETS = ["WDF_aligned", "DFF_inpainting", "DFF_insight", "DFF_text2img"]
CONDS = ["clean", "jpeg75", "jpeg50", "jpeg30", "blur1", "blur2", "blur3", "noise5", "noise10", "noise20",
         "resize0.5", "resize0.35", "resize0.25", "social"]


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize(img, s):
    w, h = img.size
    small = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def degrade(img, cond, path):
    if cond == "clean":
        return img
    if cond.startswith("jpeg"):
        return jpeg(img, int(cond[4:]))
    if cond.startswith("blur"):
        return img.filter(ImageFilter.GaussianBlur(radius=float(cond[4:])))
    if cond.startswith("noise"):
        rng = np.random.default_rng(zlib.crc32(f"{path}|{cond}".encode()))
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, float(cond[5:]), size=(img.size[1], img.size[0], 3))
        return Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))
    if cond.startswith("resize"):
        return resize(img, float(cond[6:]))
    if cond == "social":
        return jpeg(resize(img, 0.5), 50)
    raise KeyError(cond)


class DegradedSet(Dataset):
    def __init__(self, items, preprocess, cond):
        self.items, self.preprocess, self.cond = items, preprocess, cond

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, _, _ = self.items[i]
        return self.preprocess(degrade(Image.open(path).convert("RGB"), self.cond, path)), i


def main(conds):
    os.makedirs(OUT, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.9)
    model = load_model(INIT).eval()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    states = {"none": {n: p.detach().clone() for n, p in params.items()}}
    for m, run in RUNS.items():
        sd = torch.load(f"{run}/after_CDF.pt", map_location="cuda")
        assert set(sd) == set(params), m
        states[m] = sd
    ps = panel_sets()
    # one item list over all sets: WDF 8 evenly spaced frames per sequence, DFF all images (one per unit)
    items, owner = [], []
    for s in SETS:
        it = frames(ps[s], 8 if s.startswith("WDF") else None)
        items += it
        owner += [(s, vi) for _, _, vi in it]
    owner_set = np.array([s for s, _ in owner])
    owner_vid = np.array([v for _, v in owner])
    print(f"{len(items)} frames, {len(states)} models", flush=True)
    for cond in conds:
        if os.path.exists(f"{OUT}/robust_{cond}.json"):
            print("skip", cond, flush=True)
            continue
        t0 = time.time()
        dl = DataLoader(DegradedSet(items, model.feature_extractor.preprocess, cond), batch_size=128, num_workers=24)
        P = {m: np.zeros(len(items), dtype=np.float32) for m in states}
        with torch.no_grad():
            for x, idx in dl:
                x = x.cuda(non_blocking=True)
                for m, sd in states.items():
                    for n, p in params.items():
                        p.copy_(sd[n])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg, _ = embed(model, x)
                    P[m][idx.numpy()] = lg.float().softmax(-1)[:, 1].cpu().numpy()
        res = {}
        for m in states:
            res[m] = {}
            for s in SETS:
                vl = ps[s]
                sel = owner_set == s
                vid, pf = owner_vid[sel], P[m][sel]
                vp = np.bincount(vid, weights=pf, minlength=len(vl)) / np.bincount(vid, minlength=len(vl))
                res[m][s] = {"video": [d for d, _ in vl], "label": [lab for _, lab in vl], "p_fake": vp.tolist()}
        json.dump(res, open(f"{OUT}/robust_{cond}.json", "w"))
        print(f"{cond} done in {time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or CONDS)
