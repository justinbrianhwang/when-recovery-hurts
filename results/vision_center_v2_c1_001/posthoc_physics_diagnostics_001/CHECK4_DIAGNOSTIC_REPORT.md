# Trajectory-audit Check 4 — post-hoc diagnosis (POST_HOC_EXPLORATORY)

**The original Check 4 results stay as they are: V2 FAIL in 9 of 41 moves (train_seed_1) and 5 of 39 (train_seed_2).**
Thresholds unchanged (`DIR_COS_MIN = 0.9`, `PROGRESS_MIN_M = 0.0002`). V2 rescue/harm against C0 stays 14/1 and 16/0.

## 1. Definitions (from the code)

- Check 4 (`hci.trajectory_audit.check_arm`): for every control step j of a move, the actual TCP xy step from row t0+j−1
  to t0+j must have cosine ≥ 0.9 with the commanded xy step and length ≥ 0.2 mm.
- The command is the xy part of a `pd_ee_delta_pos` action (`configs/stackcube.yaml`): an end-effector **delta target**
  relative to the TCP **measured when the action is set** (pinned ManiSkill a4a4f92: `use_target=False`,
  `interpolate=False`; IK → joint targets held by joint PD drives, stiffness 1e3, damping 1e2, force limit 100, for 5 PhysX
  sub-steps). `recovery_v1_oracle.arm_scale` refused to run unless `use_target` was False, so every C1 move ran with it.
  It is not a torque and not a promised displacement. Not logged: IK targets, drive targets, IK failures, sub-step states.
- Alignment of command and step was verified on the raw trace (`test_check4_command_and_step_alignment`): command j's
  recorded `tcp_after` is the TCP of row t0+j, and the logged action of that row × 0.1 m equals the command.

## 2. Reproduction

| Policy | V2 moves | FAIL in audit CSV | FAIL re-running the audit function | FAIL recomputed independently |
|---|---|---|---|---|
| train_seed_1 | 41 | 9 | 9 | 9 |
| train_seed_2 | 39 | 5 | 5 | 5 |

All 715 moved branches of all arms: 0 disagreements between the three. The failing scenes were taken from
`trajectory_audit_001/per_scene_checks.csv`, not from any external list.

## 3. Which condition, which step

- In all 14 V2 failures the **only failing step is step 1**, and only the **cosine** condition fails (first-step cosine
  0.63–0.88, median 0.79); the progress condition never fails (smallest first step 0.47 mm ≥ 0.2 mm).
- Steps 2+ of the same 14 moves: minimum cosine 0.934–0.999 (median 0.995). **POST_HOC_STEP2PLUS: 14/14 PASS.**
- Over every V2 step: step 1 — 14 cosine failures in 80; later steps — 0 in 529. Over every arm: all 110 failing moves
  (V1 8, single-frame 8, V2 14, oracle 28, ablations 52) fail at step 1 only; POST_HOC_STEP2PLUS fails in 0 of 715 moves.
- The first-step failures had no robot–target or robot–table impulse in the logged contact list.

## 4. Command vs motion (quantities kept apart)

| V2 moves | FAIL (n = 14) median | PASS (n = 66) median |
|---|---|---|
| (1) path: Σ|cmd step| / Σ|TCP step| (mm) | 32.5 / 11.2 | 37.5 / 13.4 |
| (2) net: |Σ cmd| / |Σ TCP| (mm) | 32.5 / 11.1 | 37.5 / 13.4 |
| (3) first-step cosine | 0.79 | 0.99 |
| (3) first-step TCP length (mm) | 1.08 | 1.65 |
| (4) TCP step just before the trigger (mm) | 1.04 | 0.65 |
| (4) cos(command 1, TCP step before trigger) | **−0.57** | **+0.66** |
| (4) cos(actual step 1, TCP step before trigger) | 0.08 | 0.74 |
| (4) cos(command 1, the policy's last command) | −0.56 | +0.89 |
| arm joint speed at the trigger (rad/s) | 0.063 | 0.050 |
| (5) true held-cube offset after the move (mm) | 9.0 | 8.9 |

Achieved fraction |TCP step| / |command| for V2: step 1 median 0.33, step 2 0.36, steps 3–12 0.37–0.38 (the command
is 5 mm and the TCP moves about 1.9 mm per step once the move is under way).

## 5. Outcome and harm

The 14 failures: 6 rescues, 8 both-succeed, **0 harm**, 0 both-fail. The intersection of Check 4 failures with harm is
empty for every arm: all 7 harm moves (seed 1 scene 51157: V2, oracle, 4 V2 ablations; seed 2 scene 51253: oracle)
passed Check 4. In 51157 the V2 move followed its command closely (cosine 1.00 on all 12 steps, about 2.03 mm per 5 mm
command) and ended at the 12-step budget with the held cube still 30.4 mm off; the harm is not a direction-following
failure. Whether the limited per-step progress or the episode horizon caused that harm is not determined.

## 6. Exploratory models (associational; leave-one-scene-out over all 715 moves of all arms)

- First step, M0: act₁ = a·cmd₁ → RMSE 0.43 mm. M1: act₁ = a·cmd₁ + b·(TCP step before the trigger) → RMSE 0.34 mm,
  a = 0.26, b = 0.31 (b in 0.29–0.33 across held-out scenes). M1 predicts a first-step cosine < 0.9 for 35 moves; 25 of them
  are among the 110 actual failures (85 actual failures not predicted, 10 predicted but passing). Correlation of predicted
  and actual first-step cosine 0.61.
- Later steps, M2: act_j = a·cmd_j + b·act_{j−1} → RMSE 0.044 mm vs 0.078 mm with the command alone (a 0.28, b 0.27).

## 7. Hypotheses and their status

| Hypothesis | Status | Evidence / what would falsify it |
|---|---|---|
| H-C4-1 The failures are a first-step effect, not a general tracking failure | **Supported (observation)** | 110/110 failing moves fail only at step 1; 0 failures from step 2 on |
| H-C4-2 The motion in progress at the trigger bends the first step (inertia / the arm's state when the target changes) | **Supported in part (association)** | Failing moves command against the motion before the trigger (cos −0.57 vs +0.66); adding that motion improves first-step prediction, but explains only 25 of 110 failures |
| H-C4-3 A PD transient response of the joint drives is the cause | **UNKNOWN / not established** | Consistent with the steady 0.37 achieved fraction and the lag term, but drive targets, IK solutions and sub-step states are not logged; a regression is not evidence of PD dynamics |
| H-C4-4 Accumulated previous targets | **Refuted by the configuration** | `use_target=False`: each step's target is measured TCP + delta; there is no accumulated target |
| H-C4-5 Contact disturbs the first step | **Not supported** | No robot–target or robot–table impulse in any failing first step |
| H-C4-6 The failures cause harm | **Not supported in C1** | 0 of 14 V2 failures (and 0 of 110 over all arms) are harm; all harm moves passed Check 4 |

Other unresolved candidates: IK solution changes or joint coupling at the first step, the z motion of the policy before
the trigger, and differences between the finite-difference TCP velocity and the instantaneous velocity. Check 4 FAILs
are not declared irrelevant to safety: in C1 they did not coincide with harm.

## 8. Files

`CHECK4_FAILURE_CASES.csv` (14), `CHECK4_ALL_MOVES.csv` (715 moves), `CHECK4_STEPWISE.csv` (every control step),
`CHECK4_FIRST_STEP_MODEL_PREDICTIONS.csv`, `CHECK4_SUMMARY.json`, `CHECK4_CASE_NOTES.md`; figures
`check4_v2_stepwise.png`, `check4_first_step_vectors.png`, `check4_first_step_model.png`.
