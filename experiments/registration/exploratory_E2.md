# Exploratory E2 (registered 2026-10-04 ~15:30 IST, after E1; post-hoc)
`realkd_lwf` = real-only feature distillation (λ=1) + LwF logit distillation (λ=1, T=2), **no** prototype memory.
Purpose: check that dropping the (E1: ineffective) memory from radial_lwf changes nothing material before carrying
the simpler method into confirmation C1. Same settings as E1 (released init, splits v1.1, few-shot, defaults, seeds 0-2, dev).
Decision rule (fixed now): carry realkd_lwf into C1 if its seed-mean dev FSh is within 1.0 AUROC of radial_lwf (82.84)
and its AA within 0.5 of radial_lwf (98.26); otherwise carry radial_lwf.
E2 note: stream.py gained default-off flags (--eval-role, --panel, method none) while E2 seed 0 ran; seeds 1-2 use that file; realkd_lwf code path unchanged.
