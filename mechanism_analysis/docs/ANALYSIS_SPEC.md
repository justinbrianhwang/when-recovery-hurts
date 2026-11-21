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

**Harm and rescue.** These use C0 of the same run, stage, policy and scene. There are three scopes. They are never mixed, and each is named in its step (review F1, targeted fix v1.0):

| Scope | Step | Numerator | Denominator |
|---|---|---|---|
| **Overall** (all scenes with both outcomes, triggered or not) | `overall_success` | arm successes | scenes with complete outcomes |
| Overall | `overall_rescue_of_c0_failure` | rescue | **every** C0 failure |
| Overall | `overall_harm_of_c0_success` | harm | **every** C0 success |
| Overall | `overall_net_rescue_minus_harm` | rescue − harm | scenes with complete outcomes |
| Not triggered | `not_triggered_rescue_of_c0_failure` / `not_triggered_harm_of_c0_success` | rescue / harm | C0 failures / successes without a trigger |
| **Trigger-conditional** | `triggered_rescue_of_c0_failure` | rescue | C0 failures among triggered scenes |
| Trigger-conditional | `triggered_harm_of_c0_success` | harm | C0 successes among triggered scenes |
| Trigger-conditional | `triggered_c0_success_kept_by_abstention` | abstained and kept success | C0 successes among triggered scenes |
| **Move-conditional** | `moved_harm_of_c0_success_where_arm_moved` | harm | C0 successes in which the arm moved the TCP ≥ 1 mm |
| Move-conditional | `moved_c0_success_kept_where_arm_moved` | kept success | the same |
| Oracle opportunity | `oracle_rescue_also_rescued`, `oracle_rescue_missed[:class]` | – | all scenes where C0 failed and the oracle succeeded |
| Oracle opportunity | `overall_rescued_where_oracle_failed` | – | every C0 failure |

`not_triggered` (the arm never ran) and `abstained_or_no_action` (triggered, then abstained) are counted apart, and both stay in every overall count. The Phase B check values are B1 V2 rescue 22/47 overall and 22/34 conditional on the trigger; B2 19/35 overall and 19/32 conditional.

**Rules:**
- A zero denominator gives NA.
- A missing outcome (arm or C0) is the completeness error `outcome_missing_completeness_error`. It is excluded from every scope, never counted as success, failure or zero, and makes the analysis INCOMPLETE.
- An abstention is not a safety pass.
- `check_funnel` asserts the following:
  - ACT + abstained = triggered;
  - the miss classes sum to the misses;
  - triggered + not triggered = scenes;
  - C0 success + C0 failure = complete scenes;
  - overall = triggered + not-triggered, for both rescue and harm;
  - arm success = C0 success + rescue − harm;
  - net = rescue − harm;
  - each arm's C0 counts equal the C0 arm's own counts.
- **Independent recount.** The overall counts are compared with the collector's own summary (`tables/crosscheck_against_collector_summary.csv`):
  - Phase B: `shadow/shadow_result.json` and `pilot/result.json`;
  - C1: `result.json`, with `per_policy.<policy>.against_C0`.

  Any mismatch makes the analysis INCOMPLETE.

## 3a. Completeness gate of a C1 run (review F2)

`run_analysis.py c1` checks the run folder against an **independent expected set** before any analysis. The gate (`completeness.py`) only reads; it never writes a ledger or changes the run.

**Expected set, `C1_ORIGINAL`:**
- The plan **file** given by `--plan`. Its own SHA256 must equal the frozen hash; a SHA string from the CLI is never copied as proof.
- The seed and arm registries the plan hashes, with their SHA256 checked: 200 scenes 51100–51299, the 9 included arms in registry order, and the policies in plan order.
- The run's provenance records (`scenes.json`, probe check, audit, both budget gates, the `ledger.json` copy, `authorization.json`) must name the plan, and no `budget_stop.json` may exist.
- The collector summary `result.json` must exist.

**Expected set, `MOCK`:**
- A separate `MOCK_EXPECTED_MANIFEST` (`--expected`) gives policies, scenes, arms and the synthetic plan hash. The records must carry that hash.
- The same completeness rules apply; MOCK is not an exception to them.
- A MOCK manifest may not name C1 research scenes or the frozen plan, and a MOCK run folder may not hold a ledger.

**Blocked, each with its own problem code** (exit code 3, status INCOMPLETE, no report or claims):
- an empty `scenes/` folder;
- a missing or extra policy pass;
- missing or extra scenes, including 199 of 200;
- the same number of scenes per policy with different IDs;
- a duplicate scene;
- a scene file name that differs from its scene ID;
- `arms`, or `cells`, that differ from the registry, including the same arm missing from both;
- an error, incomplete or unreadable record;
- a record of another policy or plan;
- a missing trace, or a trace whose SHA256 differs from the record;
- missing provenance, a stopped run or a ledger that differs from the plan;
- a missing collector summary.

**After the analysis**, every expected (policy, scene, arm) must have exactly one row.

A complete input passes. The status is `ANALYSIS_COMPLETE` only when every output listed in `run_analysis.C1_REQUIRED` exists.

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
