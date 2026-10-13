# Exploratory runs E1 (registered 2026-10-04, AFTER Gate 2 results; post-hoc by construction)

Status: **exploratory**. Designed after seeing Gate 2 (which FAILED its registered primary). No result from E1
can be reported as a confirmatory claim; anything promising must be re-confirmed on a never-consulted unseen
panel and on the FF-val retrained initialisation (option 3).

Variants (added to stream.py; existing methods unchanged, snapshot `stream_gate2_snapshot.py`):
- `realkd`: real-only feature distillation (λ = 1), **no** prototype replay. Isolates the radial-vs-featkd effect
  (Gate 2: +2.02 FSh in all seeds), which in Gate 2 was confounded with replay.
- `radial_lwf`: radial + LwF logit distillation (T = 2), both λ = 1 (LwF and radial tied on FSh in Gate 2 and
  act on different parts of the model).
Settings identical to Gate 2: splits_v1.1, released init, few-shot 25+25, defaults, seeds 0–2, dev splits.
Reported against the Gate 2 table with the same metrics (FSh final, AA, AF, gain, FF ACC@0.5, AA ACC@0.5).
Questions fixed in advance:
  Q1 realkd vs featkd (asymmetric vs symmetric KD, both without replay): FSh paired differences, 3 seeds.
  Q2 realkd vs radial: does replay add anything?
  Q3 radial_lwf vs max(radial, lwf) on FSh, with AA and gain guardrails as in A2.
