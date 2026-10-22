"""Deepfake detection demo for "Protecting Generalization in Continually Updated Deepfake Detectors".

Scores every image in a folder with three versions of the GenD detector:
  base     GenD trained on FaceForensics++ only (no update)
  seq_ft   after the update stream DFDC-P -> DFD -> Celeb-DF with plain fine-tuning
  ga_lwf   after the same stream with genuine-sample alignment + LwF (the studied update)

The sample images in samples/ come from DeepFakeFace, a dataset that none of the models was trained on.
Images in a folder named "real" or "fake" (or with "real"/"fake" in the file name) are checked against that label.

Usage:
  python demo.py                          # score samples/ with all three models
  python demo.py --images my_photos       # score your own images
  python demo.py --detect                 # detect and align the face first (for photos that are not face crops)
  python demo.py --models base,ga_lwf     # choose models
"""

import argparse
import csv
import os
import sys
import time

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import paths  # noqa: E402
GEND_DIR = paths.GEND_ROOT
from common import LN_KEYS, GenD, embed, load_trainable_state, trainable_state  # noqa: E402

def _pick_device():
    try:
        torch.zeros(1).cuda()  # is_available() can be True while CUDA itself fails to start
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = _pick_device()

MODELS = {
    "base": ("No update (GenD)", None),
    "seq_ft": ("Sequential fine-tuning", os.path.join(HERE, "weights", "seq_ft_after_stream.pt")),
    "ga_lwf": ("GA+LwF (ours)", os.path.join(HERE, "weights", "ga_lwf_after_stream.pt")),
}
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def true_label(path):
    parts = [p.lower() for p in path.split(os.sep)]
    name = parts[-1]
    if "fake" in parts[:-1] or name.startswith("fake"):
        return "fake"
    if "real" in parts[:-1] or name.startswith("real"):
        return "real"
    return None


def face_detector():
    """GenD's preprocessing: RetinaFace, alignment from five landmarks with margin 1.3 (as used for all training data)."""
    sys.path.insert(0, GEND_DIR)
    cwd = os.getcwd()
    os.chdir(GEND_DIR)  # the detector weights are found relative to the GenD folder
    try:
        import cv2
        import numpy as np
        from detector import align_face
        from src.retinaface import RetinaFace
        providers = (["CUDAExecutionProvider"] if DEVICE == "cuda" else []) + ["CPUExecutionProvider"]
        det = RetinaFace("weights/models/buffalo_l/det_10g.onnx", providers=providers)
        det.prepare(ctx_id=0, nms_thresh=0.4, input_size=(640, 640), det_thresh=0.5)
    finally:
        os.chdir(cwd)

    def crop(path):
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        pad = int(0.25 * max(img.shape[:2]))  # a border helps the detector on tightly cropped faces
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        boxes, landmarks = det.detect(img)
        if len(boxes) == 0:
            return None
        i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))  # largest face
        face, _ = align_face(img, landmarks[i], target_size=(256, 256), scale=1.3)
        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    return crop


def load_gend(state_file):
    """GenD (CLIP ViT-L/14) with the base model of the experiments: our clean retrain on FaceForensics++. The retrain only
    changed the LayerNorm and head parameters, so they are stored here (0.4 MB) and loaded on top of the released GenD."""
    model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14")
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in LN_KEYS) or n.startswith("model.")
    load_trainable_state(model, torch.load(state_file, map_location="cpu"))
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, images):
    x = torch.stack([model.feature_extractor.preprocess(im) for im in images]).to(DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        logits, _ = embed(model, x)
    return logits.float().softmax(-1)[:, 1].cpu().tolist()


def save_grid(rows, models, threshold, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    cols = min(4, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.4 * cols, 4.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, rows):
        ax.imshow(r["face"])
        title = os.path.basename(r["path"]) + (f"\ntrue: {r['label']}" if r["label"] else "")
        ax.set_title(title, fontsize=10)
        for k, m in enumerate(models):
            p = r[m]
            verdict = "fake" if p >= threshold else "real"
            color = "black" if r["label"] is None else ("#1b7f3b" if verdict == r["label"] else "#b3261e")
            ax.text(0.5, -0.06 - 0.085 * k, f"{m:7s} {p:.2f}  {verdict}", transform=ax.transAxes, ha="center", va="top",
                    family="monospace", fontsize=9.5, color=color)
    fig.suptitle(f"P(manipulated) per model, threshold {threshold} (green: correct, red: wrong)", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 1.0 / (4.6 * nrows), bottom=0.02 + 0.3 / nrows,
                        hspace=0.15 + 0.085 * len(models) * 1.9, wspace=0.08)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(HERE, "samples"), help="folder of images (searched recursively)")
    ap.add_argument("--models", default="base,seq_ft,ga_lwf", help="comma-separated subset of: " + ", ".join(MODELS))
    ap.add_argument("--detect", action="store_true", help="detect and align the face first (use for ordinary photos)")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold on P(manipulated)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"), help="output folder for the table and the image grid")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in MODELS:
            sys.exit(f"unknown model {m}; choose from {', '.join(MODELS)}")
    paths = find_images(args.images)
    if not paths:
        sys.exit(f"no images found in {args.images}")

    crop = face_detector() if args.detect else None
    rows = []
    for p in paths:
        face = crop(p) if crop else Image.open(p).convert("RGB")
        if face is None:
            print(f"no face found in {p}, skipped")
            continue
        rows.append({"path": p, "label": true_label(p), "face": face})

    t0 = time.time()
    print(f"loading GenD on {DEVICE} ...")
    model = load_gend(os.path.join(HERE, "weights", "gend_base_trainable.pt"))
    base_state = trainable_state(model)
    print(f"loaded in {time.time() - t0:.0f} s")

    faces = [r["face"] for r in rows]
    for m in models:
        ckpt = MODELS[m][1]
        load_trainable_state(model, base_state if ckpt is None else torch.load(ckpt, map_location="cpu"))
        probs = []
        for i in range(0, len(faces), 32 if DEVICE == "cuda" else 4):
            probs += predict(model, faces[i:i + (32 if DEVICE == "cuda" else 4)])
        print(f"scored {len(faces)} images with {MODELS[m][0]}")
        for r, pr in zip(rows, probs):
            r[m] = pr

    # table
    print()
    head = f"{'image':28s} {'true':5s} " + " ".join(f"{m:>14s}" for m in models)
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = []
        for m in models:
            verdict = "fake" if r[m] >= args.threshold else "real"
            mark = "" if r["label"] is None else (" ok" if verdict == r["label"] else " XX")
            cells.append(f"{r[m]:.2f} {verdict}{mark}".rjust(14))
        print(f"{os.path.basename(r['path']):28s} {r['label'] or '?':5s} " + " ".join(cells))
    labelled = [r for r in rows if r["label"]]
    if labelled:
        print()
        for m in models:
            correct = sum(("fake" if r[m] >= args.threshold else "real") == r["label"] for r in labelled)
            print(f"{MODELS[m][0]:26s} correct on {correct}/{len(labelled)} labelled images")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label"] + [f"p_fake_{m}" for m in models])
        for r in rows:
            w.writerow([r["path"], r["label"] or ""] + [f"{r[m]:.4f}" for m in models])
    save_grid(rows, models, args.threshold, os.path.join(args.out, "results_grid.png"))
    print(f"\nsaved {os.path.join(args.out, 'results.csv')} and {os.path.join(args.out, 'results_grid.png')}")


if __name__ == "__main__":
    main()
