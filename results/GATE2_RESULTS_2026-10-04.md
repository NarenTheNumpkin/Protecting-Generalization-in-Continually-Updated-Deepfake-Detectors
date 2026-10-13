# Gate 2 results — RADIAL vs controls (2026-10-04)

Protocol: `registration/dev_protocol_v1.md` incl. Amendments A1 (splits v1.1) and A2 (primary endpoint =
final FSh-probe dev AUROC; guardrails). Launch hashes: `registration/gate2_launch.sha256`. 6 methods × 3
seeds (`runs/gate2/`), seq reused from `runs/gate1_v1.1/`. Released GenD-CLIP init, few-shot 25+25 per task,
defaults untouched. Analysis: `analyze_gate2.py` (output reproduced below). All runs rc=0;
the chain ran 01:13–10:30 IST, sharing the GPU with the GenD retrain.

## Verdict: **FAIL** (primary endpoint not met; guardrails met)

| Method | FSh final (mean) | FSh per seed | AA (AUROC) | AF | Gain when learned | FF ACC@0.5 | AA ACC@0.5 |
|---|---:|---|---:|---:|---:|---:|---:|
| no-update (released) | 84.85 | – | 97.20 | – | – | 95.0 | 89.1 |
| seq | 77.06 | 76.44 / 77.35 / 77.39 | 97.78 | 0.79 | +2.10 | 79.9 | 89.3 |
| **radial** | **80.69** | 80.94 / 79.54 / 81.58 | **98.42** | **0.13** | +1.38 | 87.6 | **92.4** |
| sdc | 76.94 | 75.66 / 76.50 / 78.65 | 97.70 | 0.86 | +2.05 | 78.9 | 88.8 |
| featkd | 78.66 | 78.94 / 77.57 / 79.48 | 98.18 | 0.31 | +1.51 | 85.1 | 91.0 |
| lwf | 80.49 | 80.56 / 80.23 / 80.69 | 97.95 | 0.30 | +1.39 | 86.3 | 91.0 |
| ewc | 76.67 | 75.52 / 77.34 / 77.14 | 97.73 | 0.84 | +2.10 | 79.7 | 89.2 |
| lnmerge | 80.38 | 81.01 / 79.94 / 80.18 | 97.57 | 0.30 | +0.85 | **88.1** | 90.7 |

- **Primary:** RADIAL 80.69 vs best control LwF 80.49 (+0.20); paired seed differences +0.38 / **−0.69** /
  +0.89. The registered rule needs all three positive → **FAIL**. RADIAL and LwF are statistically tied
  on unseen-forgery retention.
- Guardrail AA: 98.42 ≥ 98.18 − 0.5 → pass. Guardrail learning: +1.38 ≥ 0.5 × 2.10 → pass.

## What the data do support (secondary / ablation; not the registered claim)
1. **Asymmetric (real-only) distillation beats symmetric distillation in every seed.** radial − featkd
   on FSh: +2.01 / +1.97 / +2.10 (mean +2.02), with higher AA (+0.24) and lower AF (0.13 vs 0.31). This is
   the most consistent effect in the study and directly supports hypothesis H3 (generalisation lives in
   the genuine-face manifold). Caveat: radial also has prototype replay, which featkd lacks, so the
   contrast is "real-only KD + replay" vs "symmetric KD"; a clean "real-only KD without replay" run is
   needed to isolate it.
2. radial − sdc: FSh +5.29 / +3.04 / +2.93. **Not a clean test of the real-anchored drift idea**: sdc has
   replay with all-sample drift but *no* distillation, so the contrast mixes two components. Replay
   alone (sdc) does nothing for FSh (76.94 vs seq 77.06).
3. RADIAL is best of all methods on seen-task metrics: highest AA (98.42), lowest AF (0.13), highest
   ACC@0.5 AA (92.4); FF ACC@0.5 87.6, second to lnmerge (88.1, which learns least: gain +0.85).
4. Every method still loses ≥ 4 FSh points versus no update (84.85). **Nothing tested preserves
   unseen-forgery generalisation fully**; the best methods recover about half of the seq loss.
5. Methods that keep FSh (radial, lwf, lnmerge, featkd) all learn new tasks less (+0.85 to +1.51 vs +2.1):
   a stability–plasticity trade-off is visible.

## Allowed next steps under the protocol
- The registered λ sensitivity pass (λ ∈ {0.1, 1, 10}, identical for every regularised method, chosen by
  seed-0 primary subject to guardrails, then seeds 1–2). About 20 runs, roughly 12 h.
- Any new variant (e.g. radial + LwF, real-only KD without replay) is **post-hoc** and must be labelled
  exploratory, then confirmed on a fresh, never-consulted unseen panel and the retrained init.
