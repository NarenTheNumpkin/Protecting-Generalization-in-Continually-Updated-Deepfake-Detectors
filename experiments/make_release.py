"""Assemble the local reproducibility package (not published anywhere): split manifests, registrations with hashes,
code, per-run results, predictions, trainable-parameter checkpoints and analysis summaries. No video, image or face crop
import paths
from any dataset is included. Usage: python make_release.py <out_dir>"""

import hashlib
import json
import os
import shutil
import sys

R = paths.EXPERIMENTS
PJ = paths.PAPER
G = paths.GEND_ROOT
out = sys.argv[1] if len(sys.argv) > 1 else paths.RELEASE


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


if os.path.exists(out):
    shutil.rmtree(out)
# registrations, choices, split manifests
for f in sorted(os.listdir(f"{R}/registration")):
    sub = "splits" if f.startswith(("splits_", "eval_subsample")) else "registrations"
    cp(f"{R}/registration/{f}", f"{out}/{sub}/{f}")
sys.path.insert(0, R)
from common import panel_sets  # noqa: E402

json.dump({k: [[d.split("/datasets/")[1], l] for d, l in v] for k, v in panel_sets().items()},
          open(f"{out}/splits/unseen_panel_units.json", "w"), indent=0)
# code
for f in sorted(os.listdir(R)):
    if f.endswith(".py"):
        cp(f"{R}/{f}", f"{out}/code/radial/{f}")
for f in sorted(os.listdir(PJ)):
    if f.startswith("make_") and f.endswith(".py"):
        cp(f"{PJ}/{f}", f"{out}/code/figures/{f}")
for f in sorted(os.listdir(f"{G}/work")):
    if f.endswith(".sh"):
        cp(f"{G}/work/{f}", f"{out}/code/preprocessing/{f}")
# per-run results, predictions, trainable checkpoints
n_runs = 0
for tag in sorted(os.listdir(f"{R}/runs")):
    d = f"{R}/runs/{tag}"
    if not os.path.isdir(d):
        cp(d, f"{out}/results/{tag}")
        continue
    for run in sorted(os.listdir(d)):
        rd = f"{d}/{run}"
        if not os.path.isdir(rd):
            continue
        n_runs += 1
        for f in os.listdir(rd):
            if f.endswith((".json", ".pt")):
                cp(f"{rd}/{f}", f"{out}/results/runs/{tag}/{run}/{f}")
for a in ["analysis_f", "analysis_j2", "analysis_j3", "analysis_j4"]:
    if os.path.isdir(f"{R}/{a}"):
        for f in os.listdir(f"{R}/{a}"):
            if f == "qual.npz":  # drop gradient-x-input saliency maps: they embed pixel values of dataset images
                import numpy as np
                q = np.load(f"{R}/{a}/{f}")
                os.makedirs(f"{out}/results/{a}", exist_ok=True)
                np.savez_compressed(f"{out}/results/{a}/{f}", **{k: q[k] for k in q.files if not k.startswith("ex_sal")})
            elif f.endswith((".json", ".npz")):
                cp(f"{R}/{a}/{f}", f"{out}/results/{a}/{f}")
for f in sorted(os.listdir(R)):
    if f.endswith(".md"):
        cp(f"{R}/{f}", f"{out}/results/notes/{f}")
# manifest of SHA256 hashes for every file in the package
man = {}
for root, _, files in os.walk(out):
    for f in files:
        p = os.path.join(root, f)
        man[os.path.relpath(p, out)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(dict(sorted(man.items())), open(f"{out}/MANIFEST_sha256.json", "w"), indent=0)
open(f"{out}/README.md", "w").write(f"""# Reproducibility package: Protecting Generalization in Continually Updated Deepfake Detectors

Contents
- `splits/`: identity-disjoint split manifests (`splits_v1.1.json` is used for all reported results; `splits_v1.json` is the
  superseded version with the Celeb-DF source confound), the evaluation subsample and the unit lists of the sealed unseen panel.
- `registrations/`: every registration (development protocol, exploratory E1/E2, confirmation C1, extensions J1 to J4),
  their SHA256 files, code snapshots and the learning-rate / matched-setting choices written by the registered rules.
  The registrations were hashed and stored locally with file-system dates; they were not deposited with an independent
  timestamping service, so their dates cannot be independently verified.
- `code/radial/`: training (`stream.py`, `common.py`), analysis, bootstrap and inference scripts; `code/figures/`: figure
  scripts; `code/preprocessing/`: face-cropping scripts run with GenD's detector.
- `results/runs/<stage>/<run>/`: `args.json`, `results.json` (all stage-wise metrics), `panel_preds.json` (per-unit unseen-panel
  predictions) and `after_<task>.pt` (the trainable LayerNorm and head parameters, or the GPL watermark generator, after every
  update; load them on top of the GenD checkpoint). {n_runs} runs.
- `results/analysis_*`: per-video test and development predictions, robustness predictions, summaries of every analysis.
- `MANIFEST_sha256.json`: SHA256 of every file.

Not included: videos, images or face crops of FaceForensics++, DFDC-Preview, DeepFakeDetection, Celeb-DF, WildDeepfake or
DeepFakeFace (obtain them from their owners under their licences), and the GenD base checkpoints (GenD, MIT licence,
https://github.com/yermandy/GenD, commit 387a42266dd385fbe8f7626c5d3aa03eea3bfaab; the clean retrain follows its README recipe
with checkpoint selection on FF++ validation data).
""")
print("package", out, "files", len(man), "runs", n_runs)
