# Journal extension J3 (registered 2026-10-09, before any J3 run)

Purpose: address the three main weaknesses identified in review: (G) no comparison with the closest specialised
method, GPL (Zhang et al., ICCV 2025); (S) single training seed; (L) only three updates.
No method, hyperparameter or preprocessing choice for GAD is made from J3 results; λ = 1, T = 2 stay as fixed before C1.
Code: `stream.py` after the J3 patch (pre-patch copy `registration/stream_j2_snapshot.py`). Existing methods are
unchanged: plain stream tokens draw exactly the same few-shot sets as before (checked), and the GPL code path is
separate. Smoke tests used 4+4 development videos and one epoch only; no J3 result was looked at before this file.

## G: GPL re-implemented on GenD
GPL has no public code; we re-implement it from the paper and its supplement on the same detector and protocol.
Faithful elements: backbone and classifier frozen during updates (here: GenD LayerNorm and head frozen); a lightweight
watermark generator (three 3×3 conv layers 3→8→16→32, each with 2×2 max pooling, 1/8-resolution three-channel mask,
nearest-neighbour upsampling, sample-shared watermark θ_W, x_w = x + λ_w·attn(x)⊙θ_W) is the only trainable module
(156,659 parameters); loss L_cls + λ_nce·L_nce with λ_nce = 0.1 (GPL's best value); L_nce is the hyperbolic contrastive
loss to the nearest same-class base cluster centre against all opposite-class centres, N_c = 25 per class; the
frozen previous generator provides p_fix and the task gradient is projected against ∇KL(p_fix‖p_learn) when their
angle exceeds 90°. Choices GPL leaves unspecified, fixed here: θ_W initialised at zero (the initial model equals the
base detector, verified), sigmoid attention mask from a final 1×1 conv, λ_w = 1 (θ_W is learned, so only a
parametrisation), λ_g = 1 (orthogonal removal of the conflicting component), τ = 0.1, features mapped to the
Poincaré ball by the exponential map at the origin (curvature −1) of the unit-norm GenD feature, base cluster centres
by spherical k-means on 8 frames of each of 100+100 FF fitting videos (the base subset already used for EWC), and the
GenD augmentations and training budget of every other method (10 epochs, batch 16, Adam, cosine decay).
Learning rate: the watermark generator is a newly initialised module, so the shared rate 1e-4 may be too small.
It is selected on development data, as GAD was: runs gpl at lr ∈ {1e-4, 1e-3} and seq (lr 1e-4) on S1 with the clean
init, development splits and seed 0. Rule: among the candidates whose mean gain is at least half of seq's, take the
one with the higher FaceShifter-probe AUROC after the stream; if neither qualifies, take the one with the higher gain.
The chosen rate is written to `registration/J3_gpl_choice.json` before any test-split GPL run.

Test runs (seed 0): gpl on S1 (clean init), S2, S3 (clean init) and S1 with the released GenD-PE-L init.
Hypotheses (C1 criterion: ΔU > 0 and paired cluster bootstrap 95% CI, 2000 resamples, excluding 0):
G1 U(realkd_lwf) > U(gpl) on S1; G2 on S2; G3 on S3; G4 on PE.

## S: five training seeds (S1, clean init)
Seeds 1–4 added to seed 0 for seq, lwf, featkd, realkd, featkd_lwf, realkd_lwf and gpl (28 runs; seed 0 runs exist
from C1/J1 and block G). A seed changes the few-shot samples, the base subsets and all training randomness; the
initial GenD checkpoint is the same (retraining GenD per seed is out of scope and stated as such).
Report mean ± SD over the five seeds of U, AA, AF, gain and ACC_FF.
Hypotheses on U, each with paired differences per seed: S-H1 realkd_lwf > lwf; S-H2 realkd_lwf > seq;
S-H3 realkd > featkd; S-H4 realkd_lwf > featkd_lwf; S-H5 realkd_lwf > gpl.
Criterion: mean ΔU > 0 and one-sided paired t-test over the five seeds p < 0.05, Holm-corrected over S-H1..S-H5.
Also reported: number of seeds with ΔU > 0, and per-seed bootstrap intervals.

## L: long stream (nine updates, seed 0, clean init)
FF → DFDCP#1 → DFDCP#2 → DFDCP#3 → DFDCP#4 → DFD#1 → DFD#2 → DFD#3 → DFD#4 → CDF, where "#k" is the k-th of four
disjoint 25+25-video sessions drawn from the dataset's fitting pool (seeded shuffle); the CDF pool supports one
session. Methods: seq, lwf, featkd_lwf, realkd, realkd_lwf, gpl (chosen lr). Light evaluation at intermediate stages,
full test evaluation and the unseen panel after the last update; checkpoints after every update.
Hypotheses on the final U (C1 criterion): L1 realkd_lwf > lwf; L2 realkd_lwf > seq; L3 realkd_lwf > gpl.
Descriptive: AA and forgetting over the four datasets; stage-wise U after every update (post hoc from checkpoints,
WildDeepfake at 8 frames per sequence as in J2).

All outcomes are reported, including failures.
