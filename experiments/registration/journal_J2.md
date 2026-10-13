# Journal extension J2, Tier 1 (registered 2026-10-06, before any J2 computation)

Purpose: inference-only additions for the journal version: robustness of the retained unseen-forgery performance
to image degradations, post-update recalibration, per-manipulation breakdown, multiplicity correction of all
registered tests, test-split intervals, and qualitative analyses. No training. No method, hyperparameter or
preprocessing choice is made from J2 results; the method, λ = 1 and T = 2 stay as fixed before C1.
Models: the final (after CDF) checkpoints of the S1 runs from C1/J1 (seed 0, clean retrained init).

**Data re-use statement.** The unseen panel and the stream test splits were scored in C1/J1. J2 re-uses them under
new conditions (degradations, recalibration) and for corrected statistics. The calibration set of R below is new
(Celeb-DF dev pool, never used for scoring or selection in C1/J1).

## RB: robustness to degradations (unseen panel)
Models: none, seq, lwf, featkd, realkd, featkd_lwf, realkd_lwf, lnmerge, ewc.
Panel: WildDeepfake (aligned, amendment P1) with 8 evenly spaced frames per sequence (compute budget; the clean
condition is re-scored under the same subsampling) and all DeepFakeFace images; U defined as in C1.
Degradations, applied to the stored face crop (PIL, RGB) before the model's own preprocessing, deterministic
(noise seeded by file path and condition):
JPEG quality {75, 50, 30}; Gaussian blur σ {1, 2, 3} px; additive Gaussian noise σ {5, 10, 20}/255;
down-then-up resize (bicubic) to scale {0.5, 0.35, 0.25}; chain "social": resize 0.5 then JPEG 50.
Primary robustness score U_deg = mean of U over the 13 degraded conditions.
Hypotheses (criterion of C1: difference > 0 and paired cluster bootstrap 95% CI, 2000 resamples, excluding 0; the
same resample indices are used in every condition):
RB1 U_deg(realkd_lwf) > U_deg(lwf); RB2 U_deg(realkd_lwf) > U_deg(seq); RB3 U_deg(realkd) > U_deg(featkd).
Descriptive: U per condition and family, retention relative to no update under each condition.

## R: post-update recalibration (descriptive)
Exemplar-free constraint: only data of the current (last) task are available. Calibration set: 25 real + 25 fake
Celeb-DF dev videos (seeded sample, seed 20261006), 32 frames per video, disjoint from fit and test.
Two maps on the video-level fake probability p, fitted by NLL on the calibration set:
(a) bias-only: p' = σ(logit p + b); (b) Platt: p' = σ(a · logit p + b).
Evaluation: ECE (15 bins), ACC@0.5 and genuine FPR@0.5 on the four test splits (32-frame predictions from J1
block F) and on the unseen panel. AUROC is unchanged by construction (one monotone map per model).
Models: the ten S1 models with 32-frame predictions.

## P: per-manipulation breakdown (descriptive)
From the saved 32-frame test predictions: FF++ AUROC of real vs each of DF, F2F, FS, NT; DFDC-P method A and B.
FaceShifter probe AUROC per method from the stored stage results (development probe, reported as descriptive).

## S: statistics
S1 Multiplicity: Holm-Bonferroni over all 18 registered tests (C1: 3; J1: 15) at family-wise α = 0.05, with one-sided
bootstrap p-values p = (1 + #{Δ* ≤ 0}) / (B + 1), B = 10,000 resamples, seed 20261006, same cluster scheme as C1.
S2 Test-split intervals: paired bootstrap 95% CI (2000 resamples, videos resampled within class per task) of the
difference in AA (32-frame predictions) between realkd_lwf and every other model.

## Q: qualitative (descriptive)
t-SNE of the J1 probe features (base, seq, lwf, featkd_lwf, realkd_lwf); per-patch drift maps (1 − cos between
base and updated last-layer patch tokens) and fake-logit gradient saliency for fixed example images chosen by a
seeded rule (first 4 units of each panel set under the sorted order, plus the 4 DeepFakeFace text2img fakes with the
largest drop of p_fake under seq relative to no update); failure cases of realkd_lwf (the panel fakes with the
lowest p_fake that no update detects at 0.5).

All outcomes are reported, including failures.
