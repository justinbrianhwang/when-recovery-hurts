# Amendment 015 (execution, run2) — Vision-CENTER: B1 passed, B2 `NO_VISION_GATE_PASS`

Date: 2026-11-17 (UTC). Work id `vision_center_001`. Plan SHA256
`72b750057e31da493983f123a3cafd19cceed979d443f3daacfe76595da47e75` (unchanged). One retry approved by the user with an
additional cap of US$4.00 after the first attempt was stopped by the budget gate (see the execution amendment of run1).

## What differed from run1

Only what the approval allowed: another host, and `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, `BLIS_NUM_THREADS=1` exported before every Python process. Before stage B1 the unchanged perception
code was run with this setting on three stored frames; decisions and numbers equalled the existing outputs within 1e-9.
Plan, code, policy, constants, criteria and the gate formula are those of the frozen plan.

## Results

- Checks: environment, 321 server tests, placement check, 3-scene audit: passed. Budget gate: US$1.09 projected (0.011 s per
  environment step, 0.30 s per reading), cap US$4.00.
- **Stage B1, 200 used scenes (development data; decides only whether B2 runs): `SHADOW_PASS`.** 32 moves, 0 unnecessary,
  rescue 13, harm 0, offset error at a move median 0.9 mm, maximum 2.8 mm. C0 162, Vision-CENTER 175.
- **Stage B2, 100 new scenes 50800–50899, one pass, ledger written first: `NO_VISION_GATE_PASS`.**

| Arm | Successes of 100 | Rescue / harm against C0 |
|---|---:|---|
| C0 | 91 | — |
| VISION_CENTER | 92 | 1 / 0 |
| CENTER_ORACLE (reference, reads simulator positions) | 96 | 5 / 0 |

- Gate: rescue ≥ 4 (1), net ≥ +3 (+1), harm ≤ 3 (0).
- Vision-CENTER moved in 11 scenes and abstained in 87 (aligned 43, target not estimated 35, held-cube fit not accepted 8,
  held cube too low 1); 2 scenes had no trigger. No unnecessary move. Offset error over 55 estimates: median 0.9 mm,
  maximum 2.1 mm.
- The 4 oracle rescues it missed: target not visible 2, held-cube fit not accepted 2.
- One move (50892) pushed the target cube 5.5 mm and was stopped by the controller; the scene failed in all three arms.
- No cube position was read outside `CENTER_ORACLE`. The result recomputed locally from the scene records equals the
  server's.

## Consequences fixed by the plan

No extension, no tuning, no subset search, no second pass. The set `vision_center_pilot` is used. Estimator, thresholds and
abstain rules were not changed after the results.

## Cost

Run2 US$0.36 (cap US$4.00); run1 US$0.47; cumulative US$0.83 (limit US$4.47). No instance is left.

## Deviations

- Two test assertions that assumed the pilot had not been executed were updated after the run; no code under test changed.
- Helper files outside the plan: the server script of run2 and the thread-limit check with its reference.
- Post hoc tables (`run2/posthoc_analysis/`) were made after the outcomes were known and are descriptive.
- Rollouts are not identical across hosts (C0 on the used scenes differs from earlier runs in 9 of 200 scenes).

## Records

`results/vision_center_001/run2/`.
