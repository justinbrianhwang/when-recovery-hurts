# Amendment 014 (execution) — Recovery-V1 oracle XY pilot run once

Date: 2026-11-17 (UTC). Work id `recovery_v1_oracle_xy_pilot_001`.
Plan SHA256 `b7078afbe705a5ffd8a7128951f9c9e7618bb7c2fed5c044a5474c2fc895730b` (the local plan file), unchanged.
The preparation is described in `AMENDMENT_014_recovery_v1_oracle_xy_preparation_DRAFT.md`, which is unchanged.

## Approval

The user approved Phase B in a separate message that stated the cap (US$2.50) and named the plan hash above. The same message
decided the three open points of the readiness report:

1. The 3 cm bound is the maximum displacement of the commanded target from the TCP position at the trigger (as planned), not a
   bound on the sum of commands.
2. Exact simulator cube positions may be used for control only inside the diagnostic control part of `CENTER_ORACLE`. The
   policy, the trigger, the monitor and the other arms receive none. This is a stated exception for one diagnostic arm; the
   project rule for the monitor and the policy is unchanged.
3. `R5` calls the unchanged `Runtime.retreat()` at the first pending release command and skips the Stage 2 entry grid check
   (step multiple of 4, inside a chunk, 48 steps left). In this run 17 of the 95 trigger states were on that grid. `R5` here is
   therefore not the Stage 2 retreat condition: same motion, different entry states.

`authorization.json` records the approval text for this plan hash. Trigger, bounds, analysis and gate were not changed.

## What was run

- Server checks: tests (294 passed, 5 skipped), file hashes against the plan, placement check (0 identical of 100 against
  1,747 reference placements).
- Technical audit on 5 already-used scenes (50400–50404): controller units as assumed; all four arms and every check on 5 of 5;
  20 tracking probes towards synthetic targets 3 cm away: 2.0–2.2 cm achieved in 10–12 steps, per-step gain 0.37–0.41, direction
  cosine at least 0.999, vertical change at most 2.0 mm, grasp kept in 20 of 20. Passed all criteria fixed beforehand.
- Budget gate: US$1.03 projected at the worst case; passed. Re-projection after two scenes: US$0.74.
- One pilot pass on `recovery_v1_pilot_reserved` (50700–50799), four arms, 100 of 100 scenes completed without error.
  The ledger was written before the first rollout.

## Result (all 100 scenes in every denominator)

| Arm | Successful scenes |
|---|---:|
| C0 (no intervention) | 82 |
| R5 (5 cm retreat) | 73 |
| HOLD_MATCHED | 82 |
| CENTER_ORACLE | 92 |

- CENTER_ORACLE against C0: rescue 10, harm 0, both succeed 82, both fail 8, net +10.
- CENTER_ORACLE against HOLD_MATCHED: rescue 10, harm 0. HOLD_MATCHED against C0: rescue 1, harm 1.
- R5 against C0: rescue 5, harm 14, net −9 (not duration-matched).
- Trigger: fired in 95 scenes, 0 false triggers by ground truth; 5 scenes without a trigger fail in every arm.
- Of the 95: 48 already within 1.0 cm (no move; CENTER_ORACLE and HOLD_MATCHED identical, outcomes equal to C0 in 48 of 48),
  47 moved (TCP 0.05–2.3 cm, median 0.6 cm; 40 ended within 1.0 cm, 6 at the 12-step limit, 1 at the end of the episode).
- Follow-up gate fixed beforehand (rescue ≥ 4, net ≥ +3, harm ≤ 3, CENTER > HOLD, audits pass): `ORACLE_GATE_PASS`.
- Simulator positions read for control: 516 reads, all in CENTER_ORACLE; 0 in the other arms.

Post hoc descriptions (made after the outcomes, not fixed rules): after getting control back the policy released within 1–6
steps and in no moved scene farther from the centre than where the control part ended; all 40 moved scenes released within
1.0 cm succeeded. In 2 of the 8 remaining failures the carried cube was low enough to push the target cube along, so the offset
did not shrink.

## Deviations

- One instance never left image loading and was deleted after 193 s; nothing was uploaded to it.
- Attempt 1 on the working instance stopped at the test suite: one assertion in `tests/test_recovery_v1_oracle.py` used
  `Path.glob('*/')`, which also returns files before Python 3.13 (server: 3.10, development machine: 3.14). The assertion was
  rewritten with `iterdir()`; the test file is not among the files hashed in the plan, no code under test changed, and the
  pilot set had not been touched. The logs of attempt 1 are kept under `run1/failed_attempts/`. Attempt 2 re-ran every phase.
- The audit's trunk runs gave the same outcome as the Vision-V1 record on 5 of 5 scenes but the same step count on only 1 of 5
  (differences of one step). Rollouts are therefore not bit-identical across hosts; within this host C0 reproduced the
  uninterrupted run exactly in 95 of 95 triggered scenes. The plan reports this item without blocking on it.

## Status and limits

`recovery_v1_oracle_xy_pilot: EXECUTED_ONCE`. `recovery_v1_pilot_reserved` (50700–50799) is used.
CENTER_ORACLE reads simulator ground truth and is not deployable. The result is feasibility evidence for this controller,
trigger and bound on one task, one policy and one noise repeat; it says nothing about perception and carries no significance
claim. The width window and run length of the trigger were chosen on the consumed diagnostic set. No policy training, image
estimator, calibration or final Test was run. Earlier results are unchanged.

Cost: US$0.44 (credit difference) of the US$2.50 cap. No instance remains.
Records: `results/recovery_v1_oracle_xy_pilot_001/run1/`. The archive
(SHA256 `72930038a97e0675662aebfb41f02720756eab1a77812b89e93eb5262d002ada`, 235 files checked one by one) is kept locally.
