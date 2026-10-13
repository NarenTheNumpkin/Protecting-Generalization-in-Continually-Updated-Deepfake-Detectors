"""Phase 0: score released GenD checkpoints on locally preprocessed crops and save per-frame /
per-video predictions with identity-clustered bootstrap intervals.

Mirrors the official test path (src/model/GenDHF.py + src/model/base.py::compute_across_videos):
HF checkpoint, the model's own feature_extractor.preprocess, bf16 autocast, softmax per frame,
mean of frame probabilities per video folder.

Usage: python work/phase0_eval.py --model GenD_CLIP_L_14 --lists work/lists/CDFv2/test/*.txt --out work/results/...
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from sklearn import metrics as M
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hf.modeling_gend import GenD  # noqa: E402


class Frames(Dataset):
    def __init__(self, files, preprocess):
        self.files, self.preprocess = files, preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.files[i]).convert("RGB")), i


def label_of(path):
    # GenD convention: <dataset>/<source>/<video>/<frame>; source containing "real" -> 0, else 1
    return 0 if "real" in path.split("/")[-3] else 1


def cluster_of(video):
    # Celeb-DF: Celeb-real/idA_NNNN and Celeb-synthesis/idA_idB_NNNN share the source-video person idA.
    # YouTube-real videos have no identity metadata and form singleton clusters.
    # DFDC-P: real/<T>_<cam>_<n> and method_X/<S>_<T>_<cam>_<n> share the source-video person T.
    dataset, source, name = video.split("/")[-3:]
    if dataset.startswith("DFDCP"):
        parts = name.split("_")
        return f"dfdcp/{parts[0] if source == 'real' else parts[1]}"
    # DFD: real/<A>__<scene> and fake/<A>_<B>__<scene>__<hash> share the source-video actor A.
    if dataset.startswith("DFD"):
        return f"dfd/{name.split('__')[0].split('_')[0]}"
    m = re.match(r"(id\d+)_", name)
    return m.group(1) if m and source.startswith("Celeb") else f"{source}/{name}"


def summarize(labels, scores):
    fpr, tpr, _ = M.roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return {
        "auroc": float(M.roc_auc_score(labels, scores)),
        "ap": float(M.average_precision_score(labels, scores)),
        "eer": float((fpr[i] + fnr[i]) / 2),
        "acc@0.5": float(np.mean((scores > 0.5) == labels)),
        "tpr@fpr5": float(np.interp(0.05, fpr, tpr)),
        "n": int(len(labels)),
        "n_fake": int(labels.sum()),
    }


def cluster_bootstrap_auc(labels, scores, clusters, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = []
    for _ in range(reps):
        take = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(labels[take])) == 2:
            out.append(M.roc_auc_score(labels[take], scores[take]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], len(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    files = sorted(f for lst in args.lists for f in open(lst).read().split())
    model = GenD.from_pretrained(f"yermandy/{args.model}").cuda().eval()
    loader = DataLoader(Frames(files, model.feature_extractor.preprocess), batch_size=args.batch, num_workers=8)

    probs = np.zeros(len(files), dtype=np.float64)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, i in loader:
            probs[i.numpy()] = model(x.cuda(non_blocking=True)).float().softmax(-1)[:, 1].cpu().numpy()

    labels = np.array([label_of(f) for f in files])
    videos = np.array([os.path.dirname(f) for f in files])
    uniq_videos = np.unique(videos)
    v_prob = np.array([probs[videos == v].mean() for v in uniq_videos])
    v_label = np.array([label_of(v + "/x") for v in uniq_videos])
    v_nframes = np.array([(videos == v).sum() for v in uniq_videos])
    v_cluster = np.array([cluster_of(v) for v in uniq_videos])

    ci, n_clusters = cluster_bootstrap_auc(v_label, v_prob, v_cluster)
    result = {
        "model": args.model,
        "lists": args.lists,
        "lists_sha256": {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in args.lists},
        "frame": summarize(labels, probs),
        "video": summarize(v_label, v_prob),
        "video_auroc_cluster_ci95": ci,
        "n_clusters": n_clusters,
        "frames_per_video": {"min": int(v_nframes.min()), "median": float(np.median(v_nframes)), "max": int(v_nframes.max())},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/frames.csv", "w") as f:
        f.write("path,label,p_fake\n")
        f.writelines(f"{p},{y},{s:.8f}\n" for p, y, s in zip(files, labels, probs))
    with open(f"{args.out}/videos.csv", "w") as f:
        f.write("video,label,cluster,n_frames,p_fake\n")
        f.writelines(
            f"{v},{y},{c},{n},{s:.8f}\n" for v, y, c, n, s in zip(uniq_videos, v_label, v_cluster, v_nframes, v_prob)
        )
    json.dump(result, open(f"{args.out}/metrics.json", "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
