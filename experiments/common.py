"""Shared pieces for the continual GenD experiments: model, task data, evaluation.

Reuses upstream GenD code unchanged (third_party/GenD @ 387a422): the HF model definition, the
uniformity/alignment losses and the training augmentations (applied to the PIL image before the
model's own preprocessing, as in src/dataset/base.py).
"""

import paths
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

GEND = paths.GEND_ROOT
sys.path.insert(0, GEND)
from src.config import Augmentations  # noqa: E402
from src.dataset.augmentations import init_augmentations  # noqa: E402
from src.hf.modeling_gend import GenD  # noqa: E402
from src.losses.unifalign import alignment, uniformity  # noqa: E402

REG = paths.REGISTRATION
W = paths.DATASETS
LN_KEYS = ("pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm")


# ----------------------------------------------------------------------------- data

def crop_dir(task: str, role: str, video: str) -> str:
    """Registration video id -> directory of GenD-preprocessed crops."""
    if task == "FF":
        return f"{W}/FF/{video}"
    if task == "FSh_probe":
        return f"{W}/FF/{video}"
    if task == "DFDCP":
        return f"{W}/DFDCP_{'test' if role == 'test' else 'train'}_aligned/{video}"
    if task == "DFD":
        return f"{W}/DFD_masked/{video}"
    if task == "CDF":
        return f"{W}/CDFv2/{video}" if role == "test" else f"{W}/CDFv2_trainpool/{video}"
    raise KeyError(task)


def load_registration():
    # v1.1 (approved amendment): CDF fit reals restricted to Celeb-real; set RADIAL_SPLITS=splits_v1 to reproduce v1 runs
    reg = json.load(open(f"{REG}/{os.environ.get('RADIAL_SPLITS', 'splits_v1.1')}.json"))
    sub = json.load(open(f"{REG}/eval_subsample_v1.json"))
    reg["tasks"]["CDF"]["dev"] = sub["CDF_dev"]
    # FSh probe reals = FF dev reals
    reg["tasks"]["FSh_probe"]["dev"]["real"] = reg["tasks"]["FF"]["dev"]["real"]
    return reg


def videos(reg, task, role, label=None):
    """[(video_dir, label)] for videos that have at least one crop; label 0 real / 1 fake."""
    out = []
    for lab, name in [(0, "real"), (1, "fake")]:
        if label is not None and lab != label:
            continue
        for v in reg["tasks"][task][role][name]:
            t = "FF" if (task == "FSh_probe" and lab == 0) else task
            d = crop_dir(t, role, v)
            if os.path.isdir(d) and os.listdir(d):
                out.append((d, lab))
    return out


def few_shot(reg, task, seed, n_real=25, n_fake=25):
    rng = random.Random(f"{task}-{seed}")
    real = sorted(videos(reg, task, "fit", 0))
    fake = sorted(videos(reg, task, "fit", 1))
    return rng.sample(real, min(n_real, len(real))) + rng.sample(fake, min(n_fake, len(fake)))


def frames(video_list, per_video=None):
    """Expand videos to (frame_path, label, video_index); per_video=k keeps k evenly spaced frames."""
    out = []
    for vi, (d, lab) in enumerate(video_list):
        fs = sorted(os.listdir(d))
        if per_video and len(fs) > per_video:
            fs = [fs[i] for i in np.linspace(0, len(fs) - 1, per_video).round().astype(int)]
        out += [(f"{d}/{f}", lab, vi) for f in fs]
    return out


class FrameSet(Dataset):
    def __init__(self, items, preprocess, train: bool):
        self.items, self.preprocess = items, preprocess
        self.aug = init_augmentations(Augmentations()) if train else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab, vi = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        return self.preprocess(img), lab, vi


# ----------------------------------------------------------------------------- model

def load_model(src="yermandy/GenD_CLIP_L_14"):
    """HF repo id (released GenD) or a Lightning .ckpt from our GenD retrain (identical parameter names,
    LinearNorm head), loaded strictly into the released model structure."""
    if src.endswith(".ckpt"):
        model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14")
        sd = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=True)
        model = model.cuda()
    else:
        model = GenD.from_pretrained(src).cuda()
    # GenD's own rule (src/model/GenD.py: substring match of unfreeze_layers): CLIP uses LN_KEYS; the PE and DINO
    # configs (src/exp/wacv_rebuttal.py) list ["norm_pre", "norm1", "norm2", "norm"], i.e. every name containing "norm"
    keys = LN_KEYS if "clip" in model.config.backbone.lower() else ("norm_pre", "norm1", "norm2", "norm")
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in keys) or n.startswith("model.")
    return model


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    missing = set(trainable_state(model)) ^ set(state)
    assert not missing, missing
    model.load_state_dict(state, strict=False)


def embed(model, x):
    """Returns (logits, unit-norm features). The LinearNorm head normalises its input itself."""
    f = model.feature_extractor(x)
    return model.model(f), F.normalize(f.float(), dim=1)


def gend_loss(logits, z, y, ce=1.0, unif=0.5, align=0.1):
    return ce * F.cross_entropy(logits.float(), y) + unif * uniformity(z) + align * alignment(z, y)


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def score(model, video_list, per_video=8, batch=128, return_preds=False):
    model.eval()
    items = frames(video_list, per_video)
    dl = DataLoader(FrameSet(items, model.feature_extractor.preprocess, False), batch_size=batch, num_workers=8)
    p = np.zeros(len(items))
    vid = np.array([v for _, _, v in items])
    k = 0
    for x, _, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = embed(model, x.cuda(non_blocking=True))
        p[k:k + len(x)] = logits.float().softmax(-1)[:, 1].cpu().numpy()
        k += len(x)
    vp = np.array([p[vid == i].mean() for i in range(len(video_list))])
    y = np.array([lab for _, lab in video_list])
    if return_preds:
        return _summary(y, vp), {"video": [d for d, _ in video_list], "label": y.tolist(), "p_fake": vp.tolist()}
    return _summary(y, vp)


def _summary(y, vp):
    return {
        "auroc": float(roc_auc_score(y, vp)),
        "acc@0.5": float(((vp > 0.5) == y).mean()),
        "real_fpr@0.5": float((vp[y == 0] > 0.5).mean()),
        "fake_tpr@0.5": float((vp[y == 1] > 0.5).mean()),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
    }


# ----------------------------------------------------------------------------- sealed unseen panel (C1 only)

def panel_sets():
    """registration/unseen_panel_v1.md + amendment P1. Returns {name: [(dir, label)]}."""
    def units(root, real_src, fake_srcs):
        out = [(f"{root}/{real_src}/{u}", 0) for u in sorted(os.listdir(f"{root}/{real_src}"))]
        for fs in fake_srcs:
            out += [(f"{root}/{fs}/{u}", 1) for u in sorted(os.listdir(f"{root}/{fs}"))]
        return out
    sets = {
        "WDF_aligned": units(f"{W}/WildDeepfake_test_aligned", "real", ["fake"]),
        "WDF_provided": units(f"{W}/WildDeepfake_test_provided", "real", ["fake"]),
    }
    for g in ["inpainting", "insight", "text2img"]:
        sets[f"DFF_{g}"] = units(f"{W}/DeepFakeFace_aligned", "real", [g])
    return sets


def score_panel(model):
    res, preds = {}, {}
    for name, vl in panel_sets().items():
        res[name], preds[name] = score(model, vl, per_video=None, return_preds=True)
    dff = float(np.mean([res[f"DFF_{g}"]["auroc"] for g in ["inpainting", "insight", "text2img"]]))
    res["U"] = {"auroc": float(np.mean([res["WDF_aligned"]["auroc"], dff])), "DFF_mean": dff}
    return res, preds
