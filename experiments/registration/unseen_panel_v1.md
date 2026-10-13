# Unseen test panel v1 (registered 2026-10-04 ~12:50 IST, before any model was run on these data)

Purpose: confirmation of generalisation-retention claims. These datasets were not used by GenD's training
or checkpoint selection (paper validation pool: CDFv3, DeepSpeak v1/v2, FFIW), are not in our task stream
(FF++, DFDC-P, DFD, Celeb-DF v2) and are not the FaceShifter dev probe. **Sealed:** no method, hyperparameter,
threshold or preprocessing decision may use them. They are scored only in the confirmation stage (C1),
which will be specified after the E1 exploratory runs.

| Dataset | Source / licence | Content | Unit | Evaluation subset |
|---|---|---|---|---|
| WildDeepfake (Zi et al., ACM MM 2020) | HF `xingjunm/WildDeepfake`, Apache-2.0, not gated | Internet deepfake videos; provided as 224×224 face-crop sequences | face sequence (folder) | entire official test split (real_test + fake_test) |
| DeepFakeFace (Song et al., 2023) | HF `OpenRL/DeepFakeFace`, OpenRAIL, not gated | IMDB-WIKI real photos; fakes from SD-Inpainting, InsightFace swap, SD v1.5 text2img | single image | fixed random subsample, seed 20261004: 1,000 real + 1,000 per fake type |

Preprocessing (fixed now): GenD RetinaFace (det_thres 0.1) + `detector.align_face(scale=1.3)` on every
provided image, largest face. Images without a detection are dropped and counted per class. WildDeepfake
sequence score = mean of frame probabilities (≤ 32 evenly spaced frames per sequence).
Metrics: AUROC per dataset and per DeepFakeFace generator (each fake type vs the same 1,000 reals),
ACC@0.5, real FPR@0.5. Cluster bootstrap for WildDeepfake over source archives (one archive = one source video).

## Amendment P1 (2026-10-04 ~15:20 IST; preprocessing only, before any model was run on the panel)
WildDeepfake crops are tight 224×224 faces; RetinaFace on them detected faces in only 7,794 / 25,792 sampled
frames (30%), dropping 99 / 806 sequences. That is a selection bias, unrelated to any model output. On a 200-frame
sample, a 25% black border restored detection to 100% (0.39 without). Therefore, for WildDeepfake:
- **primary:** pad each provided crop by 25% (black, cv2.BORDER_CONSTANT) → RetinaFace → align_face(scale 1.3);
- **sensitivity:** the provided crops unchanged (≤ 32 evenly spaced frames per sequence).
Both are scored and reported. DeepFakeFace is unchanged (detection 96.9–100% without padding).
The first WildDeepfake output (no padding) is deleted and not used.
