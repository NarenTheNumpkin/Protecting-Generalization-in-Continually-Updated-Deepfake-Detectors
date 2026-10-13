# Journal revision J4 (registered 2026-10-11, before any J4 run or J4 analysis)

Context: second external review. The genuine-only alignment + LwF objective was published earlier for fake audio
(DFWF, Ma et al., Interspeech 2021); the paper is repositioned as a controlled study of that idea in visual
foundation-model detectors. J4 adds the inexpensive analyses the review asked for. A new generator family and a second
detector family are deferred (user decision). Timing disclosure: J4 is registered after all C1/J1/J2/J3 results were
known. Registrations of this project are hashed locally (SHA256 files) and have file-system dates only; they were not
deposited with an independent timestamping service, so their dates cannot be independently verified.

Naming used from J4 on: GA = genuine-only feature alignment (method `realkd`; "positive sample alignment" in DFWF);
GA+LwF = method `realkd_lwf` (the DFWF objective on top of the GenD loss). Code is unchanged (`stream.py` = J3 snapshot).

## A: matched plasticity (new runs)
Question: does a stronger LwF, a weaker sequential fine-tuning, or GA alone reach a similar amount of learning with
equal or better retention than GA+LwF at its pre-set weights?
Development stage (S1, clean init, development splits, seed 0, no panel):
GA+LwF (λg = λl = 1, target); LwF with λ_lwf ∈ {1, 2, 4, 10}; Seq FT with lr ∈ {1e-4 (existing run
j3_gpl_dev/seq_lr1e-4), 3e-5, 1e-5}; GA with λ ∈ {1, 3, 10}.
Rule: plasticity = mean gain (as in the paper: mean AUROC improvement on a task when it is learned). For each family,
the setting whose development gain is closest to GA+LwF's development gain is chosen (ties: the more regularised
setting). Written to `registration/J4_choice.json` before any test run of block A.
Test stage: each chosen setting on S1 (clean init, sealed test splits + unseen panel) with seeds 0–4; settings equal to
an existing configuration re-use the existing runs.
Hypotheses (one-sided paired t-test over five seeds, Holm over the two): MA1 U(GA+LwF) > U(LwF matched);
MA2 U(GA+LwF) > U(Seq FT matched). Descriptive: GA matched vs GA+LwF (two-sided), test gain and AA of every
matched setting (to check that matching on development data carried over).

## B: low false-positive operating points (inference only)
For the S1 seed-0 final models (none, head, seq, ewc, replay, lwf, featkd, featkd_lwf, realkd, realkd_lwf, gpl):
score the development splits of the four trained datasets with 32 frames per video; thresholds are set so that 5%
(primary) and 1% (secondary) of the pooled development genuine videos exceed them. Report, at these thresholds, the
detection rate (TPR) and realised FPR on each test split (32-frame predictions) and on the unseen panel (per set).

## C: label-independent face selection (inference only)
Manipulated FF++ and DFD videos were cropped with the official manipulation masks, genuine videos with the
largest-face rule. Re-score the same models on the FF++ and DFD test splits with largest-face crops for every video
(DFD: existing crops; FF++ test fakes: re-cropped with the same detector and settings, without masks), 8 frames per
video, and compare test AUROC, AA over the four datasets, and the ranking of methods with the mask-based protocol.

## D: retention by family and non-inferiority (analysis of existing runs)
For GA+LwF and the main comparators: change relative to the never-updated detector of U, WildDeepfake AUROC and each
DeepFakeFace generator AUROC (worst family = smallest change), seed 0 (paired cluster bootstrap as in C1) and five seeds
(two-level bootstrap: seeds with replacement, units as in C1; 10,000 resamples). Non-inferiority of GA+LwF to the
never-updated detector on U with margin 1.0 AUROC point (lower 95% bound of the difference > −1.0). The margin is fixed
here, after the point estimates were known; it is reported as a post hoc analysis.

## E: identity-aware intervals for trained-task AA (analysis of existing predictions)
Paired bootstrap of AA differences (32-frame test predictions) resampling identity clusters within class: clusters are
connected components of the identity graph of a dataset's test videos (FF++ "a_b", DFD "a_b__…", Celeb-DF "ida_idb_…",
DFDC-P "a_b_…" identifiers); if the largest component holds more than half of a dataset's test videos, that dataset is
clustered by the first identifier instead. 2000 resamples.

All outcomes are reported, including failures.
