# Gate 1 results — does naive sequential LN fine-tuning forget? (2026-10-03)

Protocol: `registration/dev_protocol_v1.md` (SHA256 efd3af85…), splits `splits_v1.json` (68ddbb25…).
Method `seq`, released `yermandy/GenD_CLIP_L_14`, stream FF → DFDCP → DFD → CDF, few-shot 25 real +
25 fake videos per task, 10 epochs, Adam 1e-4 cosine, batch 16. Dev splits only; video AUROC from 8
frames/video. Runs: `runs/gate1/seq_seed{0,1,2}/` (≈34 min each, sharing the GPU with the retrain).

## Verdict: **PASS** (both criteria; either suffices)

| Seed | Final AUROC FF / DFDCP / DFD / CDF | AA | AF (FF, DFDCP, DFD) | FSh probe base → final | Final ACC@0.5 FF / DFDCP / DFD / CDF |
|---|---|---:|---|---|---|
| 0 | 93.23 / 96.46 / 98.98 / 92.44 | 95.27 | 2.30 (5.28, 0.91, 0.71) | 84.85 → 69.37 | 69.1 / 84.3 / 89.1 / 78.0 |
| 1 | 92.15 / 97.26 / 99.49 / 91.32 | 95.05 | 2.50 (6.35, 0.70, 0.46) | 84.85 → 68.92 | 68.1 / 91.9 / 93.1 / 73.7 |
| 2 | 93.27 / 97.46 / 98.99 / 96.21 | 96.48 | 2.01 (5.24, −0.14, 0.95) | 84.85 → 68.72 | 67.5 / 90.8 / 92.1 / 80.7 |
| **Mean** | | **95.60** | **2.27** (gate ≥ 2) | **−15.85** (gate ≥ 2) | |

Base stage (released checkpoint, before any update), dev AUROC: FF 98.50, DFDCP 95.70, DFD 98.07,
CDF 96.52, FSh probe 84.85.

## Reading

1. **Generalisation forgetting dominates.** The never-trained FaceShifter probe loses about 16 AUROC
   points, about 7× the seen-task forgetting. Within seen tasks, the base task FF loses 5–6 points;
   newer tasks barely forget. This supports the proposal's G2(b): the main cost of updating a
   foundation detector is its zero-shot generalisation, which existing continual benchmarks do not track.
2. **Threshold collapse.** FF ACC@0.5 ends at about 68% while FF AUROC is still about 93%, so updates
   shift the operating point even more than the ranking.
3. **Little plasticity headroom on S1.** The released checkpoint is already at 95.7–98.5 dev AUROC on
   every S1 task, and learning a task raises its AUROC by only −0.1 to +1.6 points. On this stream,
   the value of continual updating lies in *not* losing FF/FSh, not in gaining on new tasks. A harder
   stream (P-B in the proposal) remains needed to show plasticity benefits.
4. **Seen-task AF passes the gate by a small margin** (2.27 vs 2.00). The FSh criterion alone carries
   the decision robustly.

## Defect found: CDF few-shot sample is source-confounded (split design error)

The CDF fit pool (identity-clean non-test videos) contains 230 YouTube-real, 30 Celeb-real and 45
Celeb-synthesis videos. Fakes are always Celeb-synthesis (made from Celeb-real-style videos), but
18–24 of the 25 sampled reals per seed are YouTube-real. A detector can learn "Celeb-style → fake";
the CDF dev set uses Celeb-real reals only. Consistent with this, learning CDF *lowered* CDF dev
AUROC in seeds 0 and 1 (96.5 → 92.4 / 91.3), while seed 2 (7 Celeb-real reals) improved (→ 96.2).
GenD's own finding (paired real/fake from the same sources is essential) predicts this.

Proposed fix (needs approval; it changes a registered split after results): `splits_v1.1` draws CDF
few-shot reals only from the 30 identity-clean **Celeb-real** videos. Everything else stays
unchanged, and Gate 1 is rerun under v1.1 before Gate 2. The v1 results above are kept as recorded.

---

# Gate 1 rerun under splits v1.1 (Amendment A1, approved) — 2026-10-03, finished 14:00 UTC

Identical protocol; only the CDF few-shot reals changed (now 25 Celeb-real per seed). Runs:
`runs/gate1_v1.1/seq_seed{0,1,2}/` (args.json records `splits_v1.1`).

## Verdict: **PASS on the FSh criterion only** (registered: either criterion suffices)

| Seed | Final AUROC FF / DFDCP / DFD / CDF | AA | AF (FF, DFDCP, DFD) | Gain when learned (DFDCP, DFD, CDF) | FSh probe per stage | Final ACC@0.5 FF / DFDCP / DFD / CDF |
|---|---|---:|---|---|---|---|
| 0 | 95.93 / 98.19 / 99.13 / 97.73 | 97.74 | 0.77 (2.58, −0.81, 0.56) | +0.97, +1.47, +2.73 | 84.9 → 81.3 → 78.6 → 76.4 | 80.1 / 91.4 / 93.6 / 90.7 |
| 1 | 95.87 / 98.63 / 99.79 / 96.97 | 97.81 | 0.70 (2.64, −0.69, 0.17) | +2.23, +3.41, +1.69 | 84.9 → 77.2 → 78.4 → 77.3 | 81.0 / 94.8 / 95.5 / 87.7 |
| 2 | 95.84 / 98.24 / 99.12 / 97.95 | 97.79 | 0.88 (2.66, −0.83, 0.82) | +1.26, +3.48, +1.64 | 84.9 → 79.7 → 77.9 → 77.4 | 78.7 / 92.5 / 94.6 / 91.0 |
| **Mean** | | **97.78** | **0.79** (gate ≥ 2: not met) | | **−7.79** (gate ≥ 2: met) | |

Base ACC@0.5 (released checkpoint): FF 95.0 / DFDCP 86.5 / DFD 92.1 / CDF 82.7.

## Reading (supersedes the v1 reading where they differ)
1. **The confound fix worked.** Learning CDF now raises CDF dev AUROC (+1.6 to +2.7), and every task
   gains when learned (+1.0 to +3.5 AUROC). The v1 seen-task forgetting (AF 2.27) was largely caused by
   the confounded CDF step.
2. **Seen-task forgetting in AUROC is small** (AF 0.79). Only the base task FF loses ranking quality
   (−2.6, very consistent across seeds).
3. **Two real costs remain, both consistent across seeds:**
   - *Unseen-forgery generalisation:* FaceShifter probe −7.8 AUROC (76.4–77.4 final, sd ≈ 0.5).
   - *Base-task operating point:* FF ACC@0.5 falls 95.0 → 78.7–81.0 (−15), far more than its AUROC (−2.6),
     so updates shift the decision threshold of the base task.
4. **Implication for Gate 2:** the registered primary endpoint (final AA in AUROC) sits at 97.8 with a
   ceiling near 99, so it can hardly separate methods. The problem RADIAL must solve is the FSh drop
   and the FF threshold shift. Re-targeting the Gate 2 primary endpoint *before* any Gate 2 run is a
   protocol amendment requiring approval (proposal: primary = FSh-probe retention; AA non-inferiority
   within 0.5 of the best control; FF ACC@0.5 secondary).
5. **Caveat:** the FSh probe pairs FaceShifter fakes with FF dev reals, so it is a single unseen
   manipulation on FF sources. A broader unseen panel (other datasets/generators) is needed before any
   generalisation claim.
