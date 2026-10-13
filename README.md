# Protecting Generalization in Continually Updated Deepfake Detectors

Code, pre-registrations and paper sources for the study *Protecting Generalization in Continually Updated Deepfake
Detectors* (N. Medarametla, R. D. S. Sumanth, K. M; Vellore Institute of Technology, Chennai).

The study asks what few-shot, exemplar-free updates cost a deepfake detector that already generalizes, and whether
aligning the previous model's features of **genuine** faces protects detection of forgeries that are never trained.
The base detector is [GenD](https://github.com/yermandy/GenD) (LayerNorm-tuned CLIP ViT-L/14). The genuine-sample
alignment objective was introduced for continual fake-audio detection by DFWF (Ma et al., Interspeech 2021); this
repository adapts it to a visual foundation-model detector and evaluates it under a pre-registered protocol, with
mechanism controls, a re-implementation of GPL (ICCV 2025), five training seeds, a nine-update stream, matched-plasticity
comparisons and an unseen-forgery panel (WildDeepfake, DeepFakeFace) that is never used for training or selection.

## Repository layout

```
setup_gend.sh                 clone GenD at the pinned commit into third_party/GenD
requirements.txt              Python package versions used for the reported results
experiments/
  paths.py                    all filesystem locations (overridable with environment variables)
  common.py                   data loading, GenD model loading, video-level scoring, unseen panel
  stream.py                   continual training and evaluation over a task stream (all methods)
  make_splits.py              identity-disjoint split manifests
  preprocess_unseen.py        WildDeepfake / DeepFakeFace preparation for the sealed panel
  analyze_*.py, analysis_f.py development, confirmation and first-extension analyses
  j2_*.py, j3_*.py, j4_*.py   extension stages J2-J4 (robustness, GPL, seeds, long stream, matched plasticity, ...)
  make_release.py             assembles the archive of runs, predictions and checkpoints
  registration/               pre-registrations, SHA256 files, frozen code snapshots, split manifests, rule outputs
  scripts/                    shell chains that ran each stage; preprocessing/ holds the face-cropping scripts
paper/                        LaTeX sources (main paper, supplement), figure scripts, figure PDFs, fonts
results/                      per-stage result notes and summary JSON files behind the paper's tables
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./setup_gend.sh
pip install -r third_party/GenD/requirements.txt
```

Locations are set in `experiments/paths.py` and can be overridden: `GEND_ROOT` (GenD clone), `GEND_DATASETS`
(face crops, default `$GEND_ROOT/work/datasets`), `GEND_INIT_CKPT` (initial detector), `RAW_DATA`, `EXTERNAL_DATA`,
`FF_SPLITS` (official FaceForensics++ split lists), `RUNS_DIR` and `PYTHON`. The shell scripts read the same variables.

## Data

No dataset content is distributed here. Obtain the datasets from their owners under their licences:
FaceForensics++ c23 (including DeepFakeDetection and FaceShifter, with the official manipulation masks), DFDC-Preview,
Celeb-DF v2, WildDeepfake and DeepFakeFace. Faces are cropped with GenD's detector using the scripts in
`experiments/scripts/preprocessing/` (RetinaFace, five-landmark alignment, margin 1.3, 32 frames per video). Manipulated
FF++ and DFD videos are cropped with the official masks; `preprocess_ff_test_largestface.sh` produces the
label-independent largest-face crops used in the paper's label-independence check. The identity-disjoint splits are in
`experiments/registration/splits_v1.1.json` (used for every reported result); `splits_v1.json` is the superseded version
with the Celeb-DF source confound described in the paper.

## Initial detector

The confirmation and extension stages start from GenD retrained on FF++ following the GenD README, with checkpoint
selection on FF++ validation data only (epoch 8, `best_mAP.ckpt`); point `GEND_INIT_CKPT` to it. Development runs used the
released checkpoint `yermandy/GenD_CLIP_L_14`, and the backbone replication uses `yermandy/GenD_PE_L`.

## Running

All Python entry points are run from `experiments/`. Method names in the code and in the paper:

| code (`--method`) | paper |
|---|---|
| `none` | no update |
| `seq` | sequential fine-tuning |
| `lwf`, `ewc`, `lnmerge`, `head`, `replay`, `joint` | LwF, EWC, LayerNorm merging, head-only FT, exemplar replay, joint training |
| `featkd`, `featkd_lwf` | symmetric KD, symmetric KD + LwF |
| `realkd` | GA (genuine-sample alignment) |
| `realkd_lwf` | GA+LwF |
| `fakekd`, `randkd` | fake-only KD, random-subset KD (mechanism controls) |
| `gpl` | GPL, re-implemented on GenD |
| `sdc`, `radial`, `radial_lwf` | prototype-replay variants evaluated during development |

Example, stream S1 with sealed test splits and the unseen panel:

```bash
cd experiments
python stream.py --method realkd_lwf --seed 0 --tag c1 --init "$GEND_INIT_CKPT" --eval-role test --panel
python stream.py --method realkd --lam 3 --seed 0 --tag j4_matched --run-name realkd_l3_seed0 --init "$GEND_INIT_CKPT" --eval-role test --panel
python stream.py --method realkd_lwf --stream FF,DFDCP#1,DFDCP#2,DFDCP#3,DFDCP#4,DFD#1,DFD#2,DFD#3,DFD#4,CDF --eval-light \
                 --seed 0 --tag j3_long --init "$GEND_INIT_CKPT" --eval-role test --panel
```

Each stage was run by the script or queue named below; outputs go to `experiments/runs/<tag>/<run>/` and to
`experiments/analysis_*/`.

| stage | runs | analysis |
|---|---|---|
| Gate 1, Gate 2, E1, E2 (development) | `scripts/chain_gate1*.sh`, `chain_gate2.sh`, `chain_e1.sh`, `chain_e2.sh` | `analyze_gate2.py`, `choose_mstar.py` |
| C1 confirmation | `scripts/chain_c1*.sh` | `analyze_c1.py` |
| J1 extension | `scripts/chain_j1.sh`, `chain_post_j1.sh` | `analyze_j1.py`, `analysis_f.py` |
| J2 (robustness, recalibration, statistics, qualitative) | inference only | `j2_robust.py`, `j2_analyze_robust.py`, `j2_recal.py`, `j2_stats.py`, `j2_qual.py`, `j2_margin.py` |
| J3 (GPL, five seeds, nine updates) | `j3_queue.py` | `j3_analyze.py`, `j3_stagewise.py` |
| J4 (matched plasticity, operating points, label-independent crops, retention by family) | `j4_queue.py` | `j4_infer.py`, `j4_analyze.py` |

## Paper

```bash
cd paper
python make_legacy_figures.py && python make_journal_figures.py && python make_j2_figures.py && python make_j3_figures.py
latexmk -pdf paper.tex && latexmk -pdf supplement.tex && latexmk -pdf paper.tex
```

The figure scripts read the outputs in `experiments/runs` and `experiments/analysis_*`; the figure PDFs used in the paper are
committed so that the LaTeX sources build without rerunning the experiments. The main paper and the supplement reference
each other through the `xr` package, so compile the supplement after a first pass of the paper.

## Pre-registration records

`experiments/registration/` holds every registration (development protocol, exploratory E1/E2, confirmation C1, extensions
J1-J4), the SHA256 files written when each was registered, frozen copies of the code at registration time, and the
outputs of the registered selection rules (`C1_choice.json`, `J3_gpl_choice.json`, `J4_choice.json`). Run
`sha256sum -c registration/<name>.sha256` from `experiments/`. Known deviations, disclosed rather than corrected:

* hash files also list `common.py` and `stream.py` as they were at that stage; those files have changed since, which is
  why the frozen snapshots exist;
* `confirmation_C1.sha256` holds two hashes of the registration, before and after the recorded amendment C1-A; the second
  matches;
* `exploratory_E2.md` received a one-line disclosure note (the last line) two minutes after its hash was recorded, so its
  hash no longer matches;
* the registrations were hashed and stored locally; they were not deposited with an independent timestamping service,
  so their dates cannot be verified independently.

## Results and archive

`results/` contains the per-stage notes and the summary JSON files from which the paper's tables are built. Per-run outputs,
per-video predictions and the trainable-parameter checkpoints after every update (about 0.5 GB, no images or videos) are
assembled by `experiments/make_release.py` and distributed as a separate archive.

## Licence

No licence has been chosen for this repository yet. GenD is MIT-licensed; IEEEtran.cls and IEEEtran.bst are distributed under the LaTeX
Project Public License; the TeX Gyre Termes fonts are under the GUST Font License. Datasets remain under their owners' terms.
