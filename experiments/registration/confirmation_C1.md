# Confirmation stage C1 (registered 2026-10-04 ~16:00 IST, before any C1 run; no model has been scored on
# the test splits of the stream tasks or on the unseen panel by the continual code)

## Setting (everything fixed here)
- Initial model: our FF-val-selected GenD retrain, epoch 8, `/root/deepfake-research/third_party/GenD/work/runs/phase0/phase0-retrain-GenD_CLIP-FFval-seed0/checkpoints/best_mAP.ckpt` (SHA256 3aba975d37efa4df314a208b4cff9a8737e7784591eb7ba4404866b80ac06f85).
  Not the released checkpoint used during development.
- Stream S1 FF → DFDCP → DFD → CDF; splits_v1.1; few-shot 25 real + 25 fake per task; seeds 0, 1, 2; all training
  defaults unchanged (10 epochs, Adam 1e-4 cosine, batch 16, λ = 1, T = 2).
- Methods: none (no-update reference), seq, lwf, featkd, realkd, M*. M* is chosen by the rule registered in
  exploratory_E2.md: realkd_lwf if its E2 dev FSh is within 1.0 of radial_lwf (82.84) and AA within 0.5 (98.26),
  else radial_lwf. The choice is made automatically from E2 dev results before C1 starts.
- Evaluation after every stage: **test** splits of FF, DFDCP, DFD, CDF (video AUROC, 8 evenly spaced frames/video)
  + FSh dev probe. After the last stage (and for 'none' at the base stage): the sealed unseen panel
  (unseen_panel_v1.md + amendment P1). Per-video predictions are saved for bootstrap CIs.

## Primary endpoint
U = mean( WDF_aligned AUROC , mean over the 3 DeepFakeFace generators of DFF AUROC ), after the last stage.

## Confirmatory hypotheses (each: seed-mean difference > 0 AND positive in all 3 seeds)
- C-H1: M* > lwf on U.
- C-H2: M* > seq on U.
- C-H3: realkd > featkd on U (genuine-only vs symmetric feature distillation).
All three are reported whatever the outcome; no correction is claimed beyond the all-seeds requirement.

## Guardrails / secondary (reported, M* claims qualified if a guardrail fails)
- Test AA of M* ≥ best other method's test AA − 0.5; M* mean gain-when-learned ≥ 0.5 × seq's.
- Retention vs no-update: U(method) − U(none); per-dataset panel AUROCs; WDF_provided sensitivity;
  test AF; FF ACC@0.5; FSh probe.
- Identity/source-cluster bootstrap 95% CIs on saved predictions (WDF: source archive; DFF: image).

## Not allowed
Any change of method, hyperparameter, preprocessing, panel subset or endpoint after the first C1 run starts;
dropping seeds; re-running a failed hypothesis with new settings and reporting it as confirmatory.

## Amendment C1-A (2026-10-04 17:45 IST; requested by the user for time reasons)
Made after the runs none, realkd_lwf, lwf (seed 0) had completed and seq seed 0 had started, but **before any C1
output was inspected or analysed**. C1 is reduced to **seed 0 only** (M* = realkd_lwf per C1_choice.json).
Because "positive in all 3 seeds" is no longer possible, each confirmatory hypothesis C-H1..C-H3 now requires:
(i) seed-0 difference in U > 0, and (ii) a paired bootstrap 95% CI of that difference excluding 0
(2,000 resamples; WildDeepfake units resampled by source archive, DeepFakeFace by image, separately per
dataset, recomputing U each time). Guardrails unchanged (seed-0 values). Single-seed results are weaker evidence
(no training-seed variance), and the paper must state this.
