"""Preprocess the sealed unseen panel (registration/unseen_panel_v1.md) with GenD's RetinaFace + align_face.

WildDeepfake: test archives (<id>.tar.gz -> ./<id>/{real,fake}/<seq>/<frame>.png); unit = <id>_<seq>;
              up to 32 evenly spaced frames per sequence (numeric frame order).
DeepFakeFace: zip per type; fixed random subsample (seed 20261004) of 1,000 images per type; unit = image.
Output: GenD layout work/datasets/<DATASET>/<source>/<unit>/<frame>.png; stats JSON with drop counts.
Only preprocessing happens here: no model is run on these data.
"""

import paths
import json
import os
import random
import re
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

GEND = paths.GEND_ROOT
sys.path.insert(0, GEND)
from detector import align_face  # noqa: E402
from src.retinaface import prepare_model  # noqa: E402

EXT = paths.EXTERNAL_DATA
OUT = f"{GEND}/work/datasets"
model = prepare_model(0.1)


def align(img, pad=0.0):
    if pad:
        p = int(pad * img.shape[0])
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    xyxy, lm = model.detect(img)
    if len(xyxy) == 0:
        return None
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return align_face(img, lm[areas.argmax()], target_size=None, scale=1.3)[0]


def wildeepfake(stats):
    root = f"{EXT}/WildDeepfake/deepfake_in_the_wild"
    seqs = {}
    for split in ["real_test", "fake_test"]:
        for arc in sorted(os.listdir(f"{root}/{split}")):
            with tarfile.open(f"{root}/{split}/{arc}") as t:
                for m in t.getmembers():
                    if not (m.isfile() and m.name.endswith(".png")):
                        continue
                    _, aid, cls, seq, fname = m.name.split("/")
                    seqs.setdefault((cls, f"{aid}_{seq}"), []).append((int(re.sub(r"\D", "", fname)), arc, split, m.name))
    todo = []
    for (cls, unit), frames in sorted(seqs.items()):
        frames.sort()
        idx = np.linspace(0, len(frames) - 1, min(32, len(frames))).round().astype(int)
        todo.append((cls, unit, [frames[i] for i in sorted(set(idx))]))
    stats["WildDeepfake"] = {"units": {"real": 0, "fake": 0}, "frames_in": 0, "frames_aligned": 0, "units_dropped": {"real": 0, "fake": 0}}

    # one streaming pass per archive; files are named .tar.gz but are plain tar, so mode "r|*" auto-detects
    want = {}
    for cls, unit, frames in todo:
        for num, arc, split, name in frames:
            want.setdefault((split, arc), {})[name] = (cls, unit, num)
    n_ok = {(cls, unit): 0 for cls, unit, _ in todo}
    n_in = {(cls, unit): len(frames) for cls, unit, frames in todo}

    def run(key):
        split, arc = key
        names = want[key]
        done = []
        with tarfile.open(f"{root}/{split}/{arc}", "r|*") as t:
            for m in t:
                if m.name in names:
                    cls, unit, num = names[m.name]
                    img = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_COLOR)
                    prov = f"{OUT}/WildDeepfake_test_provided/{cls}/{unit}"  # amendment P1: sensitivity variant
                    os.makedirs(prov, exist_ok=True)
                    cv2.imwrite(f"{prov}/{num:06d}.png", img)
                    face = align(img, pad=0.25)  # amendment P1: 25% black border before detection
                    if face is not None:
                        out = f"{OUT}/WildDeepfake_test_aligned/{cls}/{unit}"
                        os.makedirs(out, exist_ok=True)
                        cv2.imwrite(f"{out}/{num:06d}.png", face)
                        done.append((cls, unit))
        return done

    with ThreadPoolExecutor(6) as ex:
        for done in ex.map(run, sorted(want)):
            for k in done:
                n_ok[k] += 1
    s = stats["WildDeepfake"]
    for (cls, unit), ok in n_ok.items():
        s["frames_in"] += n_in[(cls, unit)]
        s["frames_aligned"] += ok
        s["units"][cls] += ok > 0
        s["units_dropped"][cls] += ok == 0


def deepfakeface(stats):
    rng = random.Random(20261004)
    stats["DeepFakeFace"] = {}
    for kind in ["wiki", "inpainting", "insight", "text2img"]:
        src = "real" if kind == "wiki" else kind
        with zipfile.ZipFile(f"{EXT}/DeepFakeFace/{kind}.zip") as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))
            pick = rng.sample(names, 1000)
            ok = 0
            for n in pick:
                img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
                face = align(img) if img is not None else None
                if face is None:
                    continue
                unit = re.sub(r"[^0-9A-Za-z_]", "_", os.path.splitext(n)[0])
                os.makedirs(f"{OUT}/DeepFakeFace_aligned/{src}/{unit}", exist_ok=True)
                cv2.imwrite(f"{OUT}/DeepFakeFace_aligned/{src}/{unit}/{unit}.png", face)
                ok += 1
            stats["DeepFakeFace"][src] = {"pool": len(names), "sampled": 1000, "aligned": ok}
            json.dump(pick, open(f"{OUT}/DeepFakeFace_aligned/{src}_sample.json", "w"))


if __name__ == "__main__":
    stats = json.load(open(f"{OUT}/unseen_panel_v1_preprocess_stats.json")) if "--wdf-only" in sys.argv else {}
    if "--wdf-only" not in sys.argv:
        deepfakeface(stats)
    wildeepfake(stats)
    json.dump(stats, open(f"{OUT}/unseen_panel_v1_preprocess_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
