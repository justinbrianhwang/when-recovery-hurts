# Amendment 015 (preparation, DRAFT) — Vision-CENTER

Date: 2026-11-17. Work id `vision_center_001`. Status: prepared, **not run**. No simulator step, no policy inference, no new
scene, no paid compute. Plan SHA256 `72b750057e31da493983f123a3cafd19cceed979d443f3daacfe76595da47e75` (local plan file).

## Why

The Recovery-V1 oracle pilot passed its gate (`ORACLE_GATE_PASS`): a bounded horizontal centring move at the policy's first
release command turned 10 of 18 failures into successes and harmed none, but it read simulator cube positions. This work asks
whether the same move can be driven by RGB images and robot state only.

## What is added (nothing existing is changed)

- `src/hci/vision_center.py`: perception by colour segmentation and projection with fixed camera constants (no trained
  model); an act / abstain decision; the Vision-CENTER controller; a three-arm scene harness (`C0`, `VISION_CENTER`,
  `CENTER_ORACLE`). The oracle module, the policy and all earlier results are untouched.
- `scripts/32_vision_center.py`, `scripts/vision_center/run.sh`: guarded driver. `configs/vision_center_splits.json`: new set
  `vision_center_pilot`, seeds 50800–50899, no recorded use.
- `tests/test_vision_center.py` (27 tests: synthetic images, three stored real frames, mock physics) and a fixture with
  three hand-camera frames of the consumed diagnostic set.
- Two assertions in `tests/test_recovery_v1_oracle.py` said that no pilot ledger exists. That was true before the oracle
  pilot ran; they now accept the ledger of the executed plan. No code under test changed.

## Inputs of Vision-CENTER

Hand-camera RGB and TCP pose from the policy's own observation, plus fixed camera constants. It reads no cube pose, grasp
flag, task flag or outcome. Cube positions are used only as labels in the offline evaluation and in after-the-fact error
columns of the records. `CENTER_ORACLE` stays the only control code that reads cube positions, as allowed for the oracle pilot.

## Frozen design

- Trigger: the oracle pilot's, unchanged.
- Estimate at the trigger from the current hand image only (no memory of earlier images): target cube from its visible top
  face, held cube from its camera-facing face and outline, offset = (TCP + hold offset) − target.
- Abstain (the policy's queue is left untouched, so the episode is the uninterrupted one) when the held cube or the target is
  not estimated, the held cube is estimated below 0.058 m, fewer than 20 steps are left, or the estimated offset is below
  1.5 cm.
- Act: the oracle pilot's bounds (5 mm per step, target within 3 cm, 12 steps, no vertical command, held gripper command);
  stop at an estimated 1.0 cm, or when the estimate is lost, the target appears to have moved, or the held cube is too low.
- The base camera is not used: about 1 cm per pixel at the cubes, and errors of several centimetres pass single-image checks.
- All constants were chosen on used data (traces of the 200 used scenes, 21 failed diagnostic episodes with frames, training
  demonstrations).

## Offline evaluation (stored data only)

- Frames exist for 21 of the 200 used scenes: the failed diagnostic episodes. No frame of a successful policy episode and no
  frame of 50700–50799 is stored.
- 15 of those 21 reach the trigger: ACT 7 (offset error median 1.0 mm, maximum 1.3 mm), target not estimated 6, held cube not
  accepted 2. No unnecessary act.
- The 6 without a target estimate all have the target on the far side of the held cube as seen from the hand camera.
- 167 demonstration releases (motion planner, mostly centred): 1 act (true offset 1.8 cm), 166 abstentions, 0 unnecessary acts.

## Plan (needs a separate approval naming a cap and the plan hash)

1. Stage B1, shadow pass on the 200 used scenes (re-run because their frames are not stored): `C0` and `VISION_CENTER`.
   Fixed criteria decide whether stage B2 runs; a failure stops the work and nothing is tuned and re-tried under this plan.
2. Stage B2, one pass on the 100 new scenes: `C0`, `VISION_CENTER`, `CENTER_ORACLE`. Gate for Vision-CENTER against C0:
   rescue ≥ 4, net ≥ +3, harm ≤ 3.

Expected cost US$0.90–1.40; worst-case projection of the budget gate about US$2.5 at US$0.44/h; proposed cap US$3.00
(a proposal, not an approval).

## Limits stated in advance

Simulator images with flat colours and exact camera parameters; one task, one policy. From the stored frames about half of
the cases that need a correction are not observable from the hand camera, so the expected number of rescues on 100 new scenes
is close to the gate threshold. No training, calibration or final Test is part of this work.
