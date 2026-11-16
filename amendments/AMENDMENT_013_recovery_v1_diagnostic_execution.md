# Amendment 013 (execution) — Recovery-V1 diagnostic run once

Date: 2026-11-16 (UTC). Work id `recovery_v1_diagnostic_001`.
Plan SHA256 `26c9374041e143439ea1588da89e21170d68743c732bff0aeeae93f4a066fd35`, unchanged.
The preparation is described in `AMENDMENT_013_recovery_v1_diagnostic_preparation_DRAFT.md`, which is unchanged.

## Approval

After the readiness report asked for a cap (the request had proposed US$2.00), the user answered "proceed" without a number.
This was taken as approval of the US$2.00 cap and written into `authorization.json` for this plan hash.

## What was run

- One pass of the frozen Vision-V1 policy (train seed 1, 80k EMA) without intervention on `recovery_v1_diag` (50600–50699),
  with the per-step ground-truth recorder. 100 of 100 episodes completed without error. The ledger was written before the first
  rollout.
- Server checks passed first: tests (245 passed, 5 skipped), file hashes against the plan, placement check (0 identical of 100),
  probe on 3 used selection scenes (outcome and steps equal to the Vision-V1 record in 3 of 3), budget gate (US$0.84 projected).

## Result

79 of 100 successful. 21 failures by the labels fixed beforehand: `NO_APPROACH` 5, `EARLY_RELEASE_ABOVE` 5,
`POST_RELEASE_UNSTABLE` 5, `RELEASE_OFF_TARGET_XY` 3, `GRASP_NOT_LIFTED` 1, `HOLDING_NOT_PLACED` 1, `IN_PLACE_NOT_STATIC` 1,
`UNKNOWN` 0. The largest label is a three-way tie.

Post hoc reading (made after seeing the traces, not a fixed rule): the 14 failures of the four release labels share one
property. The horizontal offset between the cubes at the last step before the open command was 1.7–4.7 cm in all 14, against a
median of 1.0 cm (maximum 2.3 cm) in the 79 successes. Up to 1.5 cm: 62 episodes, 0 failures; above 2.5 cm: 8 episodes, 8
failures. Height above the resting height did not differ (median 0.27 cm against 0.28 cm). This is an association in one set of
100 scenes; no intervention has tested it.

## Proposal (not implemented, not run)

One local action, "centre before release": at the policy's first open command while carrying, hold the gripper closed, make a
bounded horizontal move that removes the offset to the target cube, and hand control back. The source of the offset is an open
decision: a wrist-camera estimate (deployable; a first offline check on 158 stored frames gave a median error of 0.9 cm, not yet
sufficient) or the simulator (an oracle reference arm, which would need explicit permission because ground-truth pose is not
otherwise used for control). The pilot plan compares no intervention, the existing 5 cm retreat and the new action from the
same restored state with the same noise keys on `recovery_v1_pilot_reserved`.

## Deviations

- Three instances were created and deleted before a usable one: one host that does not expose the graphics libraries (tried
  twice; the first time the driver option had also been omitted), one host stuck in image loading, one host whose GPU was gone.
  No diagnostic scene was touched on them. The failed environment-check log is kept under `run1/failed_attempts/`.
- The archive download was interrupted once; the partial file was deleted and the archive was fetched in hash-checked parts.

## Status and limits

`recovery_v1_diagnostic: EXECUTED_ONCE`. `recovery_v1_diag` is used; `recovery_v1_pilot_reserved` is unused.
The labels describe logged mechanisms, not causes proven by intervention. One task, one policy, 100 scenes, development data.
No policy training, recovery action, calibration or final Test was run. Earlier results are unchanged.

Cost: US$0.39 (credit difference) of the US$2.00 cap, of which about US$0.16 on the unusable instances. No instance remains.
Records: `results/recovery_v1_diagnostic/run1/`. Frames of failed episodes and the archive
(SHA256 `7301502aa4462e3a5d98d0dfa8f26213851ec2e4833944b14fc914a57529f4a6`) are kept locally.
