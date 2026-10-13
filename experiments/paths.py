"""Filesystem locations used by the experiment code. Every location can be overridden with an environment variable;
the defaults assume the repository layout described in README.md (GenD cloned into third_party/GenD)."""

import os
import sys

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXPERIMENTS)

# GenD code (https://github.com/yermandy/GenD, commit 387a42266dd385fbe8f7626c5d3aa03eea3bfaab)
GEND_ROOT = os.environ.get("GEND_ROOT", os.path.join(ROOT, "third_party", "GenD"))
# face crops produced by GenD's detector (see experiments/scripts/preprocessing)
DATASETS = os.environ.get("GEND_DATASETS", os.path.join(GEND_ROOT, "work", "datasets"))
# GenD retrained on FF++ with checkpoint selection on FF++ validation data only (the clean initialization)
INIT_CKPT = os.environ.get("GEND_INIT_CKPT", os.path.join(
    GEND_ROOT, "work", "runs", "phase0", "phase0-retrain-GenD_CLIP-FFval-seed0", "checkpoints", "best_mAP.ckpt"))
# raw downloads (FaceForensics++, DFDC-P, Celeb-DF) and the official FF++ split lists
RAW_DATA = os.environ.get("RAW_DATA", os.path.join(ROOT, "data", "raw"))
EXTERNAL_DATA = os.environ.get("EXTERNAL_DATA", os.path.join(ROOT, "data", "external"))
FF_SPLITS = os.environ.get("FF_SPLITS", os.path.join(ROOT, "third_party", "FaceForensics", "dataset", "splits"))
# outputs of this repository
REGISTRATION = os.path.join(EXPERIMENTS, "registration")
RUNS = os.environ.get("RUNS_DIR", os.path.join(EXPERIMENTS, "runs"))
PAPER = os.path.join(ROOT, "paper")
RELEASE = os.environ.get("RELEASE_DIR", os.path.join(ROOT, "release"))
PYTHON = os.environ.get("PYTHON", sys.executable)
