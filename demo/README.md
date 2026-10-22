# Demo

Scores face images with three versions of the GenD detector and shows which ones each version gets right.

| Model    | What it is |
|----------|------------|
| `base`   | GenD trained on FaceForensics++ only (our clean retrain, never updated) |
| `seq_ft` | GenD after the updates DFDC-Preview → DFD → Celeb-DF with plain fine-tuning |
| `ga_lwf` | GenD after the same updates with genuine-sample alignment + LwF (the update studied in the paper) |

`weights/` holds only the parameters that differ from the released GenD model (LayerNorm and head, 0.4 MB each);
the rest of GenD is downloaded from Hugging Face on the first run. Requires the setup of the main README
(requirements and GenD in `third_party/GenD`).

## Run

```bash
cd demo
python make_samples.py /path/to/DeepFakeFace_aligned   # copies 11 sample faces into samples/ (not redistributed)
python demo.py                                         # scores samples/ with all three models
python demo.py --images my_images --detect             # your own photos: detects and aligns the face first
```

The output is a table of P(manipulated) per image and model, `results/results.csv` and `results/results_grid.png`.
It uses the GPU when one works and the CPU otherwise. Images in a folder called `real` or `fake`, or with a name
starting with `real` or `fake`, are checked against that label. `run_my_image.sh` runs the `my_images` case.

On the samples (DeepFakeFace, never trained by any model) the original detector is right on all 11 images;
after plain fine-tuning it misses 4 of the 7 manipulated faces, and after GA+LwF it still detects them. The samples
were chosen to show this effect and are not representative of overall accuracy.
