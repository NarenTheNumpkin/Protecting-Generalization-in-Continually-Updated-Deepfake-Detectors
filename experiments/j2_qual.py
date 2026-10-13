"""J2 block Q (registration/journal_J2.md): qualitative analyses, inference only.

1. t-SNE (per model, seed 0, perplexity 30) of the J1 probe features (analysis_f/probe_features.npz).
2. Per-patch drift maps: 1 - cos between base and updated last-layer patch tokens; mean maps over probe-set groups
   and maps for the registered example images.
3. Fake-logit gradient saliency (|grad x input|, channel mean) for the example images.
Writes analysis_j2/qual.npz and analysis_j2/qual_examples.json.
"""

import paths
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE

from analysis_f import probe_items
from common import load_model, load_registration, panel_sets
from j2_stats import S1PATH

INIT = paths.INIT_CKPT
OUT = "analysis_j2"
MODELS = ["none", "seq", "lwf", "featkd_lwf", "realkd_lwf"]
PANEL = ["WDF_aligned", "DFF_inpainting", "DFF_insight", "DFF_text2img"]


def first_frame(unit):
    return unit if os.path.isfile(unit) else f"{unit}/{sorted(os.listdir(unit))[0]}"


def pick_examples():
    ps = panel_sets()
    pp = {m: json.load(open(f"{S1PATH[m]}/panel_preds.json")) for m in ["none", "seq", "realkd_lwf"]}
    ex = []
    for s in PANEL:
        for d, lab in ps[s][:2] + [u for u in ps[s] if u[1] == 1][:2]:  # first units of each set (2 real, 2 fake)
            ex.append({"set": s, "unit": d, "label": lab, "rule": "first"})
    s = "DFF_text2img"
    y = np.array(pp["none"][s]["label"])
    drop = np.array(pp["none"][s]["p_fake"]) - np.array(pp["seq"][s]["p_fake"])
    drop[y == 0] = -np.inf
    for i in np.argsort(-drop)[:4]:
        ex.append({"set": s, "unit": pp["none"][s]["video"][i], "label": 1, "rule": "largest_seq_drop"})
    cand = []
    for s in PANEL:
        a, b = np.array(pp["none"][s]["p_fake"]), np.array(pp["realkd_lwf"][s]["p_fake"])
        y = np.array(pp["none"][s]["label"])
        for i in np.flatnonzero((y == 1) & (a > 0.5)):
            cand.append((b[i], s, pp["none"][s]["video"][i]))
    for _, s, u in sorted(cand)[:4]:
        ex.append({"set": s, "unit": u, "label": 1, "rule": "failure_gadlwf"})
    for e in ex:
        e["image"] = first_frame(e["unit"])
        for m in pp:
            i = pp[m][e["set"]]["video"].index(e["unit"])
            e[f"p_{m}"] = pp[m][e["set"]]["p_fake"][i]
    return ex


def tokens(model, x):
    h = model.feature_extractor.vision_model(x).last_hidden_state
    return torch.nn.functional.normalize(h[:, 1:].float(), dim=-1)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.55)
    reg = load_registration()
    model = load_model(INIT).eval()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    states = {m: torch.load(f"{S1PATH[m]}/after_CDF.pt", map_location="cuda") for m in MODELS if m != "none"}

    def use(m):
        model.load_state_dict(base_state, strict=True)
        if m != "none":
            model.load_state_dict(states[m], strict=False)

    save = {}
    # ---- 1. t-SNE of saved probe features
    pf = np.load("analysis_f/probe_features.npz")
    save["tsne_gid"] = pf["gid"]
    for k in ["base", "seq", "lwf", "featkd_lwf", "realkd_lwf"]:
        save[f"tsne_{k}"] = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(pf[k].astype(np.float32))
        print("tsne", k, flush=True)
    # ---- 2. mean drift maps over probe groups (one frame per probe item, at most 200 items per group)
    items, gid = probe_items(reg)
    rng = np.random.default_rng(0)
    sel = np.concatenate([rng.choice(np.flatnonzero(gid == g), min(200, (gid == g).sum()), replace=False) for g in np.unique(gid)])
    prep = model.feature_extractor.preprocess
    groups = gid[sel]
    tok = {}
    with torch.no_grad():
        for m in MODELS:
            use(m)
            acc = []
            for i in range(0, len(sel), 64):
                x = torch.stack([prep(Image.open(items[j][0]).convert("RGB")) for j in sel[i:i + 64]]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    acc.append(tokens(model, x).half().cpu())
            tok[m] = torch.cat(acc)
            print("tokens", m, flush=True)
    side = int(round(tok["none"].shape[1] ** 0.5))
    for m in MODELS[1:]:
        d = 1 - (tok[m].float() * tok["none"].float()).sum(-1)
        for g in ["genuine", "fake"]:
            mask = torch.tensor(np.char.startswith(groups.astype(str), g))
            save[f"driftmap_{m}_{g}"] = d[mask].mean(0).reshape(side, side).numpy()
    # ---- 2b/3. example images: drift maps and saliency
    ex = pick_examples()
    x = torch.stack([prep(Image.open(e["image"]).convert("RGB")) for e in ex]).cuda()
    with torch.no_grad():
        use("none")
        t0 = tokens(model, x)
        for m in MODELS[1:]:
            use(m)
            save[f"ex_drift_{m}"] = (1 - (tokens(model, x) * t0).sum(-1)).reshape(len(ex), side, side).cpu().numpy()
    for m in ["none", "seq", "realkd_lwf"]:
        use(m)
        sal = []
        for i in range(0, len(x), 2):  # small chunks: full-precision backward through ViT-L
            xi = x[i:i + 2].clone().requires_grad_(True)
            lg = model.model(model.feature_extractor(xi))
            lg[:, 1].sum().backward()
            sal.append((xi.grad * xi).abs().mean(1).detach().cpu().numpy())
            model.zero_grad(set_to_none=True)
        save[f"ex_sal_{m}"] = np.concatenate(sal)
    np.savez_compressed(f"{OUT}/qual.npz", **save)
    json.dump(ex, open(f"{OUT}/qual_examples.json", "w"), indent=1)
    print("QUAL DONE", flush=True)
