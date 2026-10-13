# RADIAL development protocol v1 (registered 2026-10-03, before any continual run)

Code: `radial/stream.py`, `radial/common.py`. Splits: `splits_v1.json`
(SHA256 68ddbb25…), CDF dev subsample `eval_subsample_v1.json`. Test roles stay sealed; every
decision below uses **dev** splits only.

## Setting
- Stream S1: FF (base, already learned) → DFDCP → DFD → CDF. Orders S2/S3 only after Gate 2.
- Initial model for development: the released `yermandy/GenD_CLIP_L_14`. Final runs swap in the
  FF-val-selected retrain (Phase 0 step 4) once it finishes.
- Incremental data: few-shot, 25 real + 25 fake fit videos per task, seeds 0/1/2 (GPL's regime),
  all 32 crops per video, GenD training augmentations.
- Trainable: LN affine + LinearNorm head (104,450 params). Loss: GenD (CE + 0.5 unif + 0.1 align).
- Defaults (not tuned): 10 epochs/task, Adam lr 1e-4 cosine to 0, batch 16, λ = 1, λ_rep = 1,
  64 replayed features per step, K = 8 prototypes per class per task, LwF T = 2.
- Base-task memory/Fisher: 100 real + 100 fake FF fit videos (seed 1000+s), 8 frames each, used once.
- Evaluation after every stage: dev split of each task + FaceShifter probe (FSh val fakes vs FF dev
  reals; never trained). Video AUROC (mean of 8 evenly spaced frame probs) and ACC@0.5.
- Summary metrics (on AUROC and on ACC@0.5): AA = mean over seen tasks after the last stage;
  AF = mean over old tasks of (best earlier − final); plasticity = mean AUROC on each task right after
  learning it; GR = FSh-probe AUROC after the last stage / at the base stage.

## Gate 1 — does forgetting exist? (method `seq`, 3 seeds)
Pass if seed-mean AF ≥ 2 AUROC points **or** the FSh-probe AUROC drops ≥ 2 points from the base stage.
Fail → stop the method track and write up the "LN-tuned foundation detectors barely forget" analysis.

## Gate 2 — RADIAL vs controls (3 seeds each, all at the defaults above)
Controls: seq, lwf, ewc, featkd (symmetric distillation), lnmerge, sdc (prototype replay with
all-sample drift; radial without the real anchor). Pass requires all of:
1. RADIAL's final AA (AUROC) exceeds the best exemplar-free control, with a paired difference > 0 in
   all 3 seeds;
2. RADIAL's FSh-probe retention is not worse than that control (seed-mean);
3. Ablations radial vs featkd (asymmetric vs symmetric distillation) and radial vs sdc (real-anchored
   vs all-sample drift) are both reported, whatever their sign.
One sensitivity pass is allowed, applied identically to every regularised method: λ ∈ {0.1, 1, 10},
selected per method by seed-0 dev AA, then rerun at seeds 1–2. No other tuning; failures are reported.

## Not allowed
Using test splits for any choice; changing splits after results; per-task hyperparameters; dropping
seeds or tasks; changing the gate thresholds after seeing results.

## Amendment A1 (2026-10-03, approved by the user after Gate 1 v1)
Splits `splits_v1.1.json` (SHA256 c8317bc9…): CDF fit reals restricted to the 30 identity-clean Celeb-real
videos (v1 samples were 72–96% YouTube-real against 100% Celeb-synthesis fakes, a source confound). Nothing
else changes. Gate 1 is rerun under v1.1 (tag `gate1_v1.1`); v1 results stay recorded in
GATE1_RESULTS_2026-10-03.md. Code changes: `common.load_registration` selects the split file
(default v1.1); `stream.py` records the split version in args.json; spherical k-means uses farthest-point
initialisation (fixed before any prototype-method run, after the correctness check).

## Amendment A2 (2026-10-04 01:15 IST, approved by the user after Gate 1 v1.1, before any Gate 2 run)
Gate 1 v1.1 showed seen-task AA near ceiling (seq 97.78 vs no-update 97.20), so AA cannot separate
methods. The cost of updating is unseen-forgery generalisation (FSh probe −7.8) and the base-task operating
point (FF ACC@0.5 95 → ~80). Gate 2 criteria are replaced by:
- **Primary:** final FSh-probe dev AUROC (after the last stage). RADIAL must exceed the best exemplar-free
  control (lwf, ewc, featkd, lnmerge, sdc; seq is also a control) on the seed mean, with a positive
  paired difference in all 3 seeds.
- **Guardrails (both required):** (a) RADIAL final AA ≥ (best control's AA) − 0.5; (b) RADIAL's mean
  gain-when-learned on new tasks ≥ 50% of seq's (seq: +2.10). A method that does not learn cannot pass.
- **Secondary (reported, not gating):** final FF ACC@0.5; AF; ACC@0.5 AA; plasticity per task.
- **Reference rows:** no-update (released checkpoint, the base stage) and seq (gate1_v1.1 runs, reused).
- Ablations radial vs featkd and radial vs sdc reported whatever their sign. The λ sensitivity pass (if run)
  selects by seed-0 primary endpoint subject to the guardrails, identically for every method.
- Because the FSh probe now guides decisions, it is a development set. Generalisation claims in the paper
  need a separate, never-consulted unseen test panel.
Runs: tag `gate2`, splits_v1.1, methods radial, sdc, featkd, lwf, ewc, lnmerge × seeds 0–2, defaults unchanged.
