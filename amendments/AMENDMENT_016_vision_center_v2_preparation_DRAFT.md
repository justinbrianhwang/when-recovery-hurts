# Amendment 016 (preparation, DRAFT) — Temporal Vision-CENTER v2

Date: 2026-11-18 (UTC). Work id `vision_center_v2_001`. Status: **Phase A, prepared, not run.** No paid compute, no simulator,
no new scene. Plan SHA256 `1b4ea3bb65bd4cfdffc13a610fee010d568a5837e632701d4b0d265e6ba66e86`. The plan, its criteria and
its cap (US$5.00 proposed) are not approved.

## Why

Vision-CENTER v1 ended `NO_VISION_GATE_PASS` (Amendment 015, execution, run2): it recovered 1 of 5 oracle rescues and
abstained on the other 4 (target hidden 2, held-cube fit rejected 2). v1, its records and its result stay as they are. v2
is a new cycle with one question: can positions that a single frame does not give be recovered from earlier RGB
observations and the robot state history.

## What the offline audit found (used scenes 50600–50899; development data)

- Full video exists only for 21 failed diagnostic episodes and the demonstrations. For scenes without video the hand view
  was reconstructed from logged states; at the trigger it agrees with 288 of 289 real frames on the target fit.
- The target moved more than 1 mm before the trigger in 18 of 289 scenes, always while the gripper was near it.
- Of 14 uninterrupted failures in which v1 had no target, the hand camera had seen the target while holding in 2, before
  the grasp only in 4, never in 8. In the 6 that have video the base camera saw it in all 6.
- The held cube moves in the gripper by at most 1.6 mm between lift and trigger, so a fit rejected at the trigger was the
  same image all along. Rejected fits are worse than accepted ones (xy error median 1.6 mm, max 8.3 mm; 0.8 and 2.8 mm).
- Scenes 50813 and 50872 (hidden target): the target was in the hand view earlier and did not move. Scenes 50817 and 50846
  (rejected held-cube fit): not recoverable by memory.

## What v2 is

`src/hci/vision_center_v2.py` (v1 imported unchanged, kept as an arm):

- target memory from the hand-camera fit or a new base-camera estimate (largest connected lit-green region), usable after
  three consistent observations, restarted by any conflicting observation, widened by 5 mm after unobserved steps with the
  gripper near it, refused when the current hand image contradicts it;
- held-cube memory valid only while the finger opening is unchanged and no open command ran;
- an error bound on every estimate that enters the act threshold and the stop tolerance;
- no move when the held cube is beside the target and not clearly above it (scene 50892), and earlier stops during the move.

Replayed on stored video: v2 would move in 4 of the 6 hidden-target episodes (offset error 0.6–2.8 mm, all inside their
bound) and makes no unnecessary move in 167 demonstration releases. No v2 move has been executed; there is no outcome.

## Plan

Stage B1: 300 used scenes, arms C0, v1, v2, which also records both cameras up to the trigger. Stage B2, only if B1 passes:
200 new scenes 50900–51099, arms C0, v1, v2, CENTER_ORACLE, one pass, ledger first. Proposed criteria are in the plan and in
the report; approving the plan by hash fixes them before any result.

## Not done, not claimed

No training, no neural model, no added camera, no viewpoint motion, no calibration, no Test. The used scenes are not
evidence for v2. Constants were chosen on used data; the side-contact rule rests on one pushed target.

## Deviations

- One assertion of `tests/test_recovery_v1_oracle.py` now also accepts the offline analysis files of this work, which name
  used scenes.
- v2 reads the base camera, which v1 did not use for control. It is part of the policy's observation.

## Records

`results/vision_center_v2_001/preparation/` (analyses, scripts, figures, plan copy).
