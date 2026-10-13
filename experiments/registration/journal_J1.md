# Journal extension J1 (registered 2026-10-05, before any J1 run)

Purpose: extend the confirmed result (C1) with the experiments a journal review requires: complete baselines,
mechanism controls, replication across task orders and backbones, sensitivity, data regime, and analyses.
Training seed 0 throughout (user decision: no multi-seed); replication across settings and paired bootstrap
intervals stand in for seed repetition. Code: `stream.py` after the J1 patch (C1 code kept as
`registration/stream_c1_snapshot.py`, `common_c1_snapshot.py`; existing methods unchanged, verified by
`check_j1_paths.py`).

**Data re-use statement.** The unseen panel (WildDeepfake, DeepFakeFace) and the stream test splits were scored once
in C1. J1 re-uses them for additional comparisons and analyses. No method, hyperparameter or preprocessing choice
is made from J1 results; the method and its λ = 1 remain as fixed before C1.

## Common settings
Clean retrained init (`best_mAP.ckpt`, epoch 8) unless stated; splits_v1.1; few-shot 25 + 25 videos per task unless stated;
10 epochs, Adam 1e-4 cosine, batch 16, λ = 1, T = 2; evaluation on sealed test splits (8 frames/video) + unseen panel.

## Blocks (run in this order)
| Block | Runs | Purpose |
|---|---|---|
| A: baselines, S1 | featkd_lwf, head, ewc, lnmerge, replay (5 videos/class/task), joint (oracle) | complete main table |
| M: mechanism controls, S1 | fakekd, randkd (distils as many samples per batch as GAD, chosen regardless of label) | is it the genuine class or just a weaker constraint? |
| O: task orders | S2 = FF→DFDCP→CDF→DFD, S3 = FF→DFD→DFDCP→CDF; seq, lwf, featkd, realkd, featkd_lwf, realkd_lwf | replication across orders |
| B: backbone | released GenD-PE-L init, S1; none, seq, lwf, featkd, realkd, featkd_lwf, realkd_lwf | replication across backbones |
| L: sensitivity, S1, light eval | realkd λ∈{0.3,3,10}; featkd λ∈{0.3,3,10}; realkd_lwf (λg,λl)∈{(0.3,1),(3,1),(1,0.3),(1,3)} | stability–plasticity frontiers |
| S: data regime, S1 | shots ∈ {5,10} for seq, lwf, realkd_lwf | few-shot sensitivity |
| F: analyses (inference only) | feature drift, calibration/ECE and score distributions, stage-wise panel score, 32-frame rescoring, t-SNE, cost | mechanism and reporting |

## Replication hypotheses (blocks O and B; each setting tested separately)
For each of S2, S3 and the PE backbone, with U defined as in C1 and the C1 criterion (difference > 0 and paired
bootstrap 95% CI, 2000 resamples, excluding 0):
R1 realkd_lwf > lwf; R2 realkd_lwf > seq; R3 realkd > featkd; R4 realkd_lwf > featkd_lwf.
## Mechanism hypotheses (block M, S1)
M1 realkd > randkd on U; M2 realkd > fakekd on U (same criterion).
All outcomes are reported, including failures. Blocks A, L, S and F are descriptive.
