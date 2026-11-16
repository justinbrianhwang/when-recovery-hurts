# Amendment 013 (preparation, draft) — Recovery-V1 diagnostic of the Vision-V1 base policy

Date: 2026-11-16. Work id `recovery_v1_diagnostic_001`. Plan SHA256 `26c9374041e143439ea1588da89e21170d68743c732bff0aeeae93f4a066fd35`.
Status: `PREPARATION_ONLY`. No simulator episode was run and no paid compute is authorised.

## What changes

- The base policy for recovery work is the Vision-V1 policy of train seed 1, 80k EMA (weights `a37be25603d6c7b7…`, 79/100 on
  `vision_v1_qualify`). It is frozen; it is not retrained. The legacy policy and all earlier results are unchanged.
- One diagnostic pass is prepared: the frozen policy without intervention on 100 new scenes, with a per-step ground-truth
  trace, and one mechanism label per failed episode by rules fixed before the pass.

## New seed sets (`configs/recovery_v1_splits.json`)

| Set | Seeds | Use |
|---|---|---|
| `recovery_v1_diag` | 50600–50699 | the one diagnostic pass |
| `recovery_v1_pilot_reserved` | 50700–50799 | reserved for a later recovery pilot; no command of this work resets or runs it |

No seed overlaps the sets of `splits.json`, `policy_eval_splits.json`, `vision_v1_splits.json`, the 550 demonstration scenes or the
scenes of Stage 2A / 2B / 2D / H1. Cube placements are compared on the server before any rollout. Qualification, calibration and
Test sets are not reused.

## Why a new pass is needed

Vision-V1 stored one summary row per episode (outcome, steps, furthest stage). The per-step trace was computed in memory and not
kept, so the stored records cannot show when the gripper opened, where the cube was at that step, or what happened after contact.

## Recording and rules

Recorded after every control step, for logging only: the executed action, TCP pose, finger opening, pose and velocity of both
cubes, the task's own flags, and contact forces (cube–cube, cube–finger, cube–table). Camera frames are stored for failed
episodes. Nothing recorded is given to the policy.

Labels (`src/hci/recovery_v1.py`, fixed by hash in the plan): `NO_APPROACH`, `NO_GRASP`, `GRASP_NOT_LIFTED`,
`NOT_RELEASED_AT_PLACE`, `HOLDING_NOT_PLACED`, `GRASP_LOST_WITHOUT_OPEN_COMMAND`, `RELEASE_OFF_TARGET_XY`, `EARLY_RELEASE_ABOVE`,
`RELEASE_BELOW_REST_HEIGHT`, `POST_RELEASE_UNSTABLE`, `IN_PLACE_NOT_STATIC`, `UNKNOWN`. Position tolerances are the task's own
(3.3 cm horizontal, 0.5 cm vertical). An episode no rule explains is `UNKNOWN`. A rule changed after traces are seen is reported
as post hoc.

## Guards

- Server commands need `authorization.json` with the plan hash and a cap; none exists.
- Order enforced by the driver: environment checks, placement check, probe on 3 already-used selection scenes, budget gate,
  then the pass. The set is marked used in a ledger before the first rollout; no resume, no second pass; an error leaves the pass
  incomplete and no episode is dropped.
- The list of files read on the server is in the plan and is checked against the driver source by a test (the Vision-V1 run
  stopped once because such a list was incomplete).

## After the pass (not part of this preparation)

One local recovery action for the most frequent mechanism, and a pilot plan comparing no intervention, the existing 5 cm retreat
and the new action from the same state with the same noise keys, with a trigger that does not use the outcome. Both need their own
approval. No policy training, calibration or final Test.

Estimated cost of the pass: US$0.60–1.00 on an RTX 4090 at about US$0.45/h (from the measured 19 s per episode of this policy);
proposed cap US$2.00, not yet approved.
