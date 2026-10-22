# Protecting Generalization in Continually Updated Deepfake Detectors

Code and paper sources for a study of how to update a generalizable deepfake detector with a few examples of new
forgeries without losing its ability to detect forgeries it has never seen. The detector is
[GenD](https://github.com/yermandy/GenD); updates align the previous model's features of genuine faces
(genuine-sample alignment, introduced for fake-audio detection by DFWF), optionally with logit distillation.

## Layout

```
experiments/     training (stream.py), data and model utilities (common.py), analysis scripts
  registration/  pre-registrations, split manifests and their SHA256 hashes
  scripts/       scripts that ran each stage, and face-cropping scripts
paper/           LaTeX sources of the paper and supplement, figures
results/         result summaries behind the paper's tables
```

## Setup

```bash
pip install -r requirements.txt
./setup_gend.sh                                   # clones GenD into third_party/GenD
pip install -r third_party/GenD/requirements.txt
```

Datasets (FaceForensics++, DFDC-Preview, Celeb-DF v2, WildDeepfake, DeepFakeFace) are not included; download them from
their owners and crop faces with the scripts in `experiments/scripts/preprocessing/`. Paths are set in
`experiments/paths.py` and can be changed with environment variables (`GEND_ROOT`, `GEND_DATASETS`, `GEND_INIT_CKPT`).

## Usage

Run from `experiments/`:

```bash
# update the detector over the stream FF++ -> DFDC-P -> DFD -> Celeb-DF and evaluate
python stream.py --method realkd_lwf --seed 0 --init "$GEND_INIT_CKPT" --eval-role test --panel
```

Methods: `realkd` = genuine-sample alignment (GA), `realkd_lwf` = GA + LwF, `seq` = plain fine-tuning, `lwf`,
`featkd` = symmetric distillation, `gpl` = GPL re-implementation; see `stream.py` for the full list.

## Paper

Two versions are built from `paper/`:

```bash
cd paper
python build_tifs.py                       # IEEE TIFS submission: paper.pdf (13 pages) + supplement.pdf
latexmk -pdf access_paper.tex              # IEEE Access version: 15 pages, 15 references
```

The IEEE Access version needs the official IEEE Access LaTeX template, which is not redistributed here. Download it
from IEEE and copy its files into `paper/` (`ieeeaccess.cls`, `spotcolor.sty`, the logos, `bullet.png` and the font
files), keeping the `IEEEtran.cls` already in `paper/`.

The plots are drawn by `paper/make_figures.py` in the style of
[figures4papers](https://github.com/ChenLiu-1996/figures4papers); it reads the experiment outputs from `experiments/`.

## Demo

`demo/` scores face images with the original detector and the two updated versions (plain fine-tuning and GA+LwF);
see `demo/README.md`.

## Report

`report/` builds the project report (Word) from the university template; see `report/README.md`.
