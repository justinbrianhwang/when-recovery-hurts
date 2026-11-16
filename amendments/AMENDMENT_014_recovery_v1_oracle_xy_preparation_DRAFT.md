# Amendment 014 (draft) — Recovery-V1 oracle XY pilot: preparation, and a diagnostic-only use of ground truth

Date: 2026-11-16 (UTC). Work id `recovery_v1_oracle_xy_pilot_001`. Status: **Phase A, preparation only. Nothing was run on a
simulator, no policy inference, no paid compute.** This draft becomes final only with the execution record.

## Why

The Recovery-V1 diagnostic (amendment 013) found, after the run, that the 14 release failures all let go of the cube
1.7–4.7 cm off-centre. That is an association in one set of 100 scenes. This pilot tests it by intervention: does a bounded
horizontal correction just before the first release raise task success, beyond what interruption, holding and re-planning do?

## The exception, stated narrowly

Project rule (AGENTS.md, unchanged): object ground-truth pose, task success and future labels are not given to the monitor or
the policy.

- One new arm, `CENTER_ORACLE`, computes its horizontal correction from the simulator's exact cube positions.
- The value is read in one function (`hci.recovery_v1_oracle.oracle_offset`), only while the control part of that arm runs. A
  gate object refuses the read anywhere else, and every read is counted per arm and stored.
- The policy observation whitelist (RGB, joint positions and velocities, TCP pose) is unchanged. The trigger, `C0`, `R5` and
  `HOLD_MATCHED` receive no object pose. No monitor or selector is part of this work.
- `CENTER_ORACLE` is a diagnostic reference arm. It is **not deployable**, and no result of it is evidence about perception.
- The rule's wording covers the monitor and the policy; the oracle value goes to a scripted reference controller. AGENTS.md is
  therefore not edited. The user asked for this arm explicitly (request of 2026-11-16). If the rule is meant to forbid ground
  truth in any controller, this work must not run; that reading is the user's to make.

## Fixed before any outcome

- **Scenes:** `recovery_v1_pilot_reserved`, seeds 50700–50799, unused. One policy-noise repeat (0). No replacement.
- **Trigger (observable only):** the first step at which the pending action is an open command (> 0), the three or more most
  recent executed commands are close commands (< 0), and the measured finger opening is 0.02–0.06 m. At most one per episode.
  The state is captured before the pending action is executed. Scenes without a trigger stay in every denominator with the
  uninterrupted outcome in all arms. False triggers are recorded from ground truth after the fact and are not excluded.
  The width window and the run length were chosen on the stored traces of the consumed diagnostic set (development data),
  where the rule fires in 94 of 100 episodes, always with the cube grasped and lifted.
- **Arms from the one captured state:** `C0` (pending action executed; must reproduce the uninterrupted run), `R5` (the
  existing retreat code, unchanged), `HOLD_MATCHED` (TCP held for the number of steps `CENTER_ORACLE` used on that state),
  `CENTER_ORACLE`.
- **`CENTER_ORACLE` bounds:** stop at an offset of 1.0 cm or less; at most 5 mm commanded per control step; the commanded TCP
  target stays within 3.0 cm of the TCP at the trigger; at most 12 control steps; no vertical command; the gripper command of
  the trigger (a close command) is held; then the queue is cleared and the unchanged policy re-plans. No forced opening, no
  state write, no steps beyond the 200-step episode.
- **Zero-step case:** offset already within 1.0 cm → both `HOLD_MATCHED` and `CENTER_ORACLE` discard the pending queue and
  re-infer at the trigger step with the same noise key; both are run and must be identical.
- **Matched handoff:** the two arms return control at the same step with the same seed, queue and history handling. Any other
  difference stops the run as a technical fault.
- **Primary reading:** paired success on all 100 scenes, `CENTER_ORACLE` against `C0`. Secondary: against `HOLD_MATCHED`;
  `R5` against `C0` (not duration-matched); measured motion.
- **Follow-up gate (engineering, not a significance test):** rescue ≥ 4, net ≥ +3, harm ≤ 3, `CENTER_ORACLE` successes above
  `HOLD_MATCHED`, all technical audits passed. Otherwise `NO_ORACLE_GATE_PASS` and the candidate is closed for this cycle.

## Deviations from the request, and readings that had to be chosen

- **`R5` does not go through `Runtime.force(2)`.** That entry point requires the Stage-2 eligibility grid (step multiple of 4,
  inside a chunk, 48 steps left); on the diagnostic traces only 13 of 94 trigger states are on it. The three statements of
  `force()` are executed without the grid and `Runtime.retreat()` is called unchanged. A test shows both give the same run on
  a state that is on the grid.
- **"3 cm" is read as a cap on the commanded target displacement,** as `commanded_distance_m` of the existing retreat, not as
  the sum of per-step commands. The stored traces show that one control step achieves about 0.39 of its commanded increment.
  Under the chosen reading the 12-step limit allows about 2.3 cm of real correction. Under the other reading (sum of commands
  ≤ 3 cm) the move would stop after 6 steps and about 1.2 cm. The sum is recorded. Changing the reading needs a new plan hash.
- **The gripper command in the three control arms is the last executed command, not a fixed −1.** The trigger guarantees it
  is a close command; holding it leaves the grip force as it was, and it is what the existing retreat does.
- **No camera frames are stored** in this pilot; per-step ground-truth traces of every arm are.

## Not verified locally

No ManiSkill installation exists on the local PC. Controller scaling and gripper sign were read from the pinned upstream
source and are consistent with the stored traces; they are read again from the live controller on the server (audit step).
Real tracking of the bounded move, grasp retention and vertical drift are unknown until the audit; its pass criteria are
fixed. The server-only code has never been executed.

## Status

`recovery_v1_oracle_xy_pilot: PHASE_A_PREPARATION_ONLY`, `paid_compute: NOT_AUTHORIZED`. Phase B needs a separate explicit
instruction that states a USD cap and names the plan hash. Earlier results, the base policy, `report.md` and AGENTS.md are
unchanged. No training, calibration or final Test.
