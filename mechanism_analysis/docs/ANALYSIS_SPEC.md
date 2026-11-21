# Analysis specification: C1 mechanism decomposition

Work `vision_center_v2_c1_001`. Reference plan `6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7`. Code: `mechanism_analysis/`.

This specification fixes the analysis. It changes:
- no collection gate;
- no primary test, Holm procedure or per-policy safety rule;
- no arm, sensor, policy or threshold.

## 1. Sources (never pooled)

| `source_kind` | What | Where | Outcomes? |
|---|---|---|---|
| `PHASE_B_ORIGINAL` | The executed Phase B run `vision_center_v2_001` on the server.<br>B1 shadow: 300 scenes, arms C0, v1, v2.<br>B2 pilot: 200 scenes, arms C0, v1, v2, oracle.<br>Policy train_seed_1 only. | `runs/vision_center_v2_001/server_backup/extracted/runs/vision_center_v2_001/{shadow,pilot}` | Yes, for the executed arms |
| `OFFLINE_DECISION_REPLAY` | Decisions recomputed from stored frames for the 9 C1 arms (v2s excluded). | `results/vision_center_v2_c1_001/preparation/offline_replay_r4.json` | No: motion and outcome `NOT_RUN` |
| `LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS` | Stored Phase B actions of 6 cases replayed on a local CPU with the spatial logger. No RGB. | `logging_instrumentation_option_a/local_cpu_replay_of_recorded_actions` | Not used as performance. Used for contacts and S6 only |
| `MOCK` | Synthetic tables and the mock C1 collector. | `mechanism_analysis/tests` | Never |
| `C1_ORIGINAL` | A future, separately approved C1 run. | absent | – |

**Rules for combining sources:**
- A table holds one source kind per aggregate; `check_keys` refuses mixed kinds.
- The analysis key is (source, run, stage, policy, scene, arm, branch).
- A figure from `PHASE_B_ORIGINAL` draws only that run's frames and trace.
- The local replay, which ran on another host and differs by up to 5.6–11 mm on full trunks, is linked, never overlaid.
- No contact is back-filled into a Phase B record.

## 2. Time axis

- **Trace row `t`.** The state after `t` environment steps. Row 0 is the record's `initial_state`. Row `t` carries the action executed at step index `t−1`.
- **Observations.** The observation used at step index `t` is the state of row `t`.
- **Policy history.** The policy's history at a call holds the observations up to that call.
- **Memories.** Memories are updated once per trunk step, from the image of that step.
- **Contacts.** In the spatial records, the contact list belongs to the last of the 5 physics sub-steps of the control step that produced row `t`.

| Event | Definition | Bounds | Null when |
|---|---|---|---|
| `t_trigger` | `T` = `record.trigger.step`; the decision uses trunk row `T` | – | not triggered |
| `t_control_start` | `T`; the first control action is step index `T` | inclusive | `K` = 0 |
| `t_control_end` | `T + K`; the state after the last control action | inclusive | `K` = 0 |
| `t_policy_resume` | the recorded `first_inference_step` of the branch | – | no call recorded |
| `t_first_open` | the first branch row `t > T` whose executed gripper command is > 0 | – | never opened |
| `t_release_gt` | the recorded first row whose grasp flag is off after being on (post hoc) | – | not found |
| `t_success` | the first branch row with `success` | – | never |
| `t_end` | the last branch row | – | no branch trace |

Checked by `check_time_order`:
- `t_trigger ≤ t_control_start < t_control_end`;
- `t_policy_resume ≥ t_control_end`;
- `t_first_open > t_control_end`.

## 3. Counts and denominators (`opportunity_funnel.csv`)

**The funnel, per run, stage, policy and arm.** Each step's denominator is its parent:
1. scenes;
2. triggered;
3. decision ACT;
4. control steps executed;
5. actual movement (TCP XY ≥ 1 mm).

**Other counts:**
- **Abstained:** triggered and not acted.
- **ACT with zero steps:** an immediate stop.
- **Technical error or incomplete:** error records or `completed: false`, listed and never dropped.

**Oracle.**
- `ACT` means at least one control step.
- An engagement with zero steps and the recorded end reason `within_tolerance` is `ORACLE_WITHIN_TOLERANCE_AT_START`.
- A zero-step engagement with any other recorded end reason (for example `episode_ended`, `target_cap`) is `ORACLE_ZERO_STEPS:<reason>` and is counted separately.

**Harm and rescue.** These use C0 of the same run, stage, policy and scene:

| Count | Numerator | Denominator |
|---|---|---|
| Harm, all | harm | all C0 successes |
| Harm, moved | harm | C0 successes the arm moved in |
| Protected by abstention | – | C0 successes where the arm abstained and kept success |
| Kept with movement | – | C0 successes where the arm moved and kept success |
| Rescue | rescue | all C0 failures |
| Oracle rescues also rescued | – | oracle rescues; computed only where the oracle ran (B2) |
| Rescued where the oracle failed | – | – |

**Rules:**
- A zero denominator gives NA.
- A missing outcome stays missing; it is never counted as success or zero.
- An abstention is not a safety pass.
- `check_funnel` asserts that ACT + abstained = triggered, and that the miss classes sum to the misses.

## 4. Decomposition A: observation, estimate, memory

**Coverage.** Every triggered branch with a control record has a status:
- `ESTIMATED:<source>`;
- `NO_TARGET_ESTIMATE`, with `why` and `memory_reason`;
- `NO_HOLD_ESTIMATE`.

Error statistics are always reported with their `n` and the coverage.

**Target (v2, memory source).** A vector identity in world XY:

`e_now = e_store − Δ`

- `e_now` = estimate − true target at `T`;
- `e_store` = the stored fused memory value − true target at the memory's `t_last`;
- `Δ` = true target at `T` − true target at `t_last`.

The magnitudes are reported separately and never added. The fused memory is used as stored.

**Held cube.**
- the estimate error at `T` (XY and Z);
- the true held-cube XY shift relative to the TCP between the last accepted hold fit (`memory.hold_steps`) and `T`.

The shift is in the world frame, an approximation without a hand-frame rotation. A refused hold estimate is `NO_HOLD_ESTIMATE` and is not lumped with occlusion.

**Own bound.** `error_within_bound` is the recorded post-hoc label against the arm's own `u`.

## 5. Decomposition B: decisions and opportunities

**Opportunities.** These are computed only where C0 and the oracle exist for the same policy, scene and branch start (B2).

**Single label `miss_class`.** It applies to a vision branch where C0 failed, the oracle succeeded and the arm failed. It is the first block in code order (`hci.vision_center_v2.estimate` then `decide`, then execution):
1. `NO_OBSERVATION` (v2 `never_seen`; v1 no target fit);
2. `HOLD_ESTIMATE_NOT_ACCEPTED`;
3. `TARGET_ESTIMATE_NOT_ACCEPTED` (v1: a fit present but not accepted);
4. `MEMORY_UNUSABLE_OR_INVALIDATED`;
5. `ABSTAIN_HEIGHT` (`LOW_CARRY`);
6. `ABSTAIN_CONTACT_RISK`;
7. `ABSTAIN_TIME`;
8. `JUDGED_ALIGNED` (`ALIGNED`, `ALIGNED_WITHIN_UNCERTAINTY`);
9. `ACT_WITHOUT_MOVEMENT`;
10. `MOVED_ALIGNMENT_INSUFFICIENT` (true offset at `T + K` > 10 mm);
11. `ALIGNED_BETTER_BUT_FAILED_AFTER_RESUME`;
12. `UNCLASSIFIABLE_MISSING_RECORD`.

**Auxiliary flags** (`miss_flags`, which may overlap):
- estimate outside its own bound;
- target or hold from memory;
- target moved during control;
- push-assisted;
- offset grew by more than 5 mm before opening;
- ended during control.

Neither the label nor the flags are a unique physical cause.

**The oracle.** It is the recorded `CENTER_ORACLE` controller, not an optimum or an upper bound of all recoveries.

## 6. Decomposition C: command, motion, alignment, resume, opening

**Inputs.** For each branch with `K > 0`, from trunk row `T` and branch row `T + K`:
- the per-step commanded XY (`control.steps[].commanded_xy_m`): path length, net vector, maximum step, and the recorded target distance from the start;
- the actual TCP XY net displacement and path;
- the held and target cube displacements;
- the held-minus-TCP displacement;
- the true offset (target − held) before and after.

**Contributions.** Each cube's contribution to closing the offset is its displacement projected on the unit starting offset:
- held: `+ΔA·û`;
- target: `−ΔB·û`.

**`push_assisted`.** The target's contribution is > 1 mm and at least half of the positive reduction.

**Direction cosines.** These are NA when either vector is shorter than 1 mm.

**The three limits are checked separately** and are never merged into one bound:

| Limit | Value |
|---|---|
| Per step | 5 mm |
| Target from the start | 3 cm |
| Steps | 12 |

**After control:**
- steps from `t_control_end` to `t_first_open`;
- the offset change from the control end to the row before the first open command;
- the offset at GT release.

A failure after resume is reported as an event order, not attributed to the resume.

## 7. Decomposition D: space, contact, S6

**Phase B records** have no contact list or link poses. Their contacts are `NOT_OBSERVABLE`.

**The local CPU replay** gives:
- contact existence (last sub-step);
- reconstructed clearances;
- the S6 decomposition.

These come from another host and are labelled so.

**S6.**
- The official verdict and its 1 mm tolerance are kept.
- `s6_validity.py` decomposes the differences by pair kind, phase, relative speed, normal direction and rotation.
- It fits an explanatory term (`EXPLANATORY_ONLY`).

**Impulse over dt** of the last sub-step is not read as an average, peak or total force.

## 8. Decomposition E: ablation and policies

**From the decision replay only.** For each arm:
- the removed element;
- the change of information source;
- the change of decision;
- ACT lost or gained versus v2.

Motion, outcome and side effects are `NOT_RUN`.

**`V2_SINGLE_FRAME`** changes two things together: memories removed and confirmation count.

**Interpretation limits:**
- Leave-one-out differences overlap and do not add.
- Counting an ACT's source is not its causal contribution.

**Policies.** For C1, all tables are per policy. `policy_paired_outcomes.csv` keeps the scene pairing of the two policies. The 200 shared scenes are not 400 independent samples.

## 9. Fixed constants (`mechanism.py`; also in `analysis_manifest.json`)

| Constant | Value | Use |
|---|---|---|
| `MOVE_MIN_M` | 1 mm | actual movement threshold |
| `PUSH_MIN_M` | 1 mm | target moved; the Phase B summary's own threshold |
| `NA_NORM_M` | 1 mm | cosine NA threshold |
| `RESIDUAL_TOL_M` | 10 mm | the controllers' residual tolerance |
| `STEP_CMD_M`, `TARGET_CAP_M`, `MAX_STEPS` | 5 mm, 3 cm, 12 | the separate control limits |

## 10. Case selection (fixed rule, `run_analysis.select_cases`)

1. **The six explanatory cases:** 50632, 50892, 51004, 51009, 50959, 50702.
2. **Rule-selected cases**, each the lowest seed not already selected:
   - a v2 rescue that is push-assisted;
   - a v2 move with an estimate outside its own bound;
   - one case per miss class, for v2 and for v1.
3. **Every harm event.**
