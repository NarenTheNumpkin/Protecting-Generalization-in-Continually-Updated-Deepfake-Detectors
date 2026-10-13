"""Phase 0: GenD-style preprocessing of the local DFDC-Preview frames.

The local DFDC-P copy (data/external/DFDC-preview-kaggle-frames) contains only
DeepfakeBench-style 256x256 face crops (up to 30 per clip), not raw videos. Two variants:

  aligned (primary): GenD's own RetinaFace (det_thres 0.1) + detector.align_face(scale=1.3,
                     target_size=None) applied to each provided crop; largest face; frames with no
                     detection are skipped, exactly as detector.process_video does.
  provided:          symlinks to the provided crops, unchanged.

Output layout follows GenD's <dataset>/<source>/<video>/<frame> convention, with sources
`real`, `method_A`, `method_B` (label = 0 iff the source name contains "real").
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import align_face  # noqa: E402
from src.retinaface import prepare_model  # noqa: E402

ROOT = os.environ.get("DFDCP_FRAMES", os.path.join(os.environ.get("EXTERNAL_DATA", "data/external"), "DFDC-preview-kaggle-frames"))
SOURCE_NAME = {"original_videos": "real", "method_A": "method_A", "method_B": "method_B"}


def json_key(method, folder):
    parts = folder.split("_")
    if method == "original_videos":
        target, cam, num = parts
        return f"original_videos/{target}/{target}_{cam}_{num}.mp4"
    _, target, cam, _ = parts
    return f"{method}/{target}/{target}_{cam}/{folder}.mp4"


def clips(split):
    meta = json.load(open(f"{ROOT}/dataset.json"))
    out = []
    for method in SOURCE_NAME:
        for folder in sorted(os.listdir(f"{ROOT}/{method}/frames")):
            entry = meta[json_key(method, folder)]  # KeyError = mapping bug, fail loudly
            if entry["set"] == split:
                assert (entry["label"] == "real") == (method == "original_videos")
                out.append((method, folder))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="work/datasets")
    args = ap.parse_args()

    todo = clips(args.split)
    model = prepare_model(0.1)
    stats = {"clips": len(todo), "frames_in": 0, "frames_aligned": 0, "clips_without_aligned_frames": []}

    def run(item):
        method, folder = item
        src_dir = f"{ROOT}/{method}/frames/{folder}"
        rel = f"{SOURCE_NAME[method]}/{folder}"
        aligned_dir = f"{args.out}/DFDCP_{args.split}_aligned/{rel}"
        provided_dir = f"{args.out}/DFDCP_{args.split}_provided/{rel}"
        os.makedirs(aligned_dir, exist_ok=True)
        os.makedirs(provided_dir, exist_ok=True)
        n_in = n_ok = 0
        for frame in sorted(os.listdir(src_dir)):
            n_in += 1
            link = f"{provided_dir}/{frame}"
            if not os.path.lexists(link):
                os.symlink(f"{src_dir}/{frame}", link)
            img = cv2.imread(f"{src_dir}/{frame}")
            xyxy, landmarks = model.detect(img)
            if len(xyxy) == 0:
                continue
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            face, _ = align_face(img, landmarks[areas.argmax()], target_size=None, scale=1.3)
            cv2.imwrite(f"{aligned_dir}/{frame}", face)
            n_ok += 1
        return rel, n_in, n_ok

    with ThreadPoolExecutor(4) as ex:
        for rel, n_in, n_ok in tqdm(ex.map(run, todo), total=len(todo)):
            stats["frames_in"] += n_in
            stats["frames_aligned"] += n_ok
            if n_ok == 0:
                stats["clips_without_aligned_frames"].append(rel)
                os.rmdir(f"{args.out}/DFDCP_{args.split}_aligned/{rel}")

    json.dump(stats, open(f"{args.out}/DFDCP_{args.split}_preprocess_stats.json", "w"), indent=2)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in stats.items()}))


if __name__ == "__main__":
    main()
