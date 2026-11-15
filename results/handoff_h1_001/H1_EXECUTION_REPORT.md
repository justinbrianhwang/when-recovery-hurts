# H1 execution report — one-shot latest-observation padding at the handoff after R5

**Status: completed.** In this sample, giving the policy the latest observation twice at its first re-inference (`R_PAD_ONCE`) was **worse** than resuming on the real last two observations (`R_KEEP`): 11 against 17 successful blocks of 180, with 8 blocks turned from success to failure and 2 from failure to success. This is a descriptive result on 12 scenes; it does not establish a cause and no novelty is claimed.

Run id: `handoff_h1_001`. Date: 2026-11-15 (UTC).
Task and policy: StackCube-v1 (pinned ManiSkill `a4a4f927…`), policy `stack_1d_s1_d500_clean_v1_i080000` (no retraining).
Host: Vast.ai instance 60740152, RTX 3090 24 GB, driver 580.126.09, AMD EPYC 7532, 04:41–05:50 UTC, destroyed. A first instance (60737590) was abandoned during installation and ran nothing (see Cost).
Code: repository `883a72a` plus one list entry in `scripts/26_handoff_h1.py`; every file hash is fixed in the plan.
Records: `amendments/AMENDMENT_011_handoff_h1_execution.md`, plan `execution_plan.json` (SHA256 `b381e24fa4902345…`), approval `authorization.json`.

```text
handoff_h1: EXECUTED_ONCE (completed)
legacy_stage3b: COMPLETED_DEV_NO_GATE_PASS
legacy_model_search: CLOSED_FOR_THIS_CYCLE
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
new_training: NOT_RUN
calibration: NOT_RUN
final_test: NOT_RUN
```

## 1. What was compared

| Arm | What happens |
|---|---|
| `C0` | Existing Continue. Context only. |
| `R_KEEP` | Existing 5 cm retreat (R5); the policy resumes on the real last two observations `[o_(tau-1), o_tau]`. |
| `R_PAD_ONCE` | The same R5; right before the first inference, once, the history becomes `[copy(o_tau), o_tau]`. |

12 new train scenes (10060–10071), checkpoints 12 / 52 / 92 / 132, 4 noise repeats. 45 snapshots, **180 blocks, 540 continuations** (upper bound 576; three checkpoints did not exist because the trunk had already succeeded). In addition 12 trunks, 45 R5 control-only reference runs, 16 reruns, and the audit on 5 scenes.

## 2. Results

### Successful blocks

| 180 blocks, 12 scenes | C0 | R_KEEP | R_PAD_ONCE |
|---|---:|---:|---:|
| All | 19 | 17 | 11 |
| Step 12 (48 blocks, 12 scenes) | 5 | 6 | 4 |
| Step 52 (48 blocks, 12 scenes) | 9 | 4 | 1 |
| Step 92 (44 blocks, 11 scenes) | 5 | 7 | 6 |
| Step 132 (40 blocks, 10 scenes) | 0 | 0 | 0 |

Scene-mean success fraction: C0 0.146, R_KEEP 0.111, R_PAD_ONCE 0.068.

### Primary comparison: R_PAD_ONCE against R_KEEP (same snapshot, same R5, same return step, same seed)

| | KEEP fail → PAD success | KEEP success → PAD fail | both succeed | both fail |
|---|---:|---:|---:|---:|
| Blocks (of 180) | 2 | 8 | 9 | 161 |
| Scenes in which it occurs | 2 | 5 | 3 | 12 |
| at step 12 / 52 / 92 / 132 | 2 / 0 / 0 / 0 | 4 / 3 / 1 / 0 | 2 / 1 / 6 / 0 | 40 / 44 / 37 / 40 |

- `Y_PAD − Y_KEEP`: block mean −0.033; **scene-equal mean −0.043**; by scene 1 positive, 5 negative, 6 zero.
- Scenes with at least one failure-to-success block: 10060, 10063. Scenes with at least one success-to-failure block: 10060, 10065, 10069, 10070, 10071. Scene 10060 has both (1 and 2) and is a net loss; 10063 is the only scene with a positive scene mean.
- The two arms had the same outcome in 170 of 180 blocks.

| Scene | Trunk | Checkpoints | C0 | R_KEEP | R_PAD_ONCE | PAD − KEEP (0→1 / 1→0) |
|---|---|---|---:|---:|---:|---|
| 10060 | fail | 4 | 1/16 | 2/16 | 1/16 | 1 / 2 |
| 10061 | fail | 4 | 0/16 | 0/16 | 0/16 | 0 / 0 |
| 10062 | fail | 4 | 0/16 | 0/16 | 0/16 | 0 / 0 |
| 10063 | fail | 4 | 0/16 | 0/16 | 1/16 | 1 / 0 |
| 10064 | fail | 4 | 0/16 | 0/16 | 0/16 | 0 / 0 |
| 10065 | fail | 4 | 1/16 | 4/16 | 2/16 | 0 / 2 |
| 10066 | fail | 4 | 0/16 | 0/16 | 0/16 | 0 / 0 |
| 10067 | fail | 4 | 0/16 | 0/16 | 0/16 | 0 / 0 |
| 10068 | fail | 4 | 0/16 | 0/16 | 0/16 | 0 / 0 |
| 10069 | success at 91 | 2 | 5/8 | 2/8 | 0/8 | 0 / 2 |
| 10070 | success at 106 | 3 | 12/12 | 7/12 | 6/12 | 0 / 1 |
| 10071 | fail | 4 | 0/16 | 2/16 | 1/16 | 0 / 1 |

### What the padding did to the input and to the first action chunk

| | Blocks |
|---|---:|
| Policy resumed after R5 (real handoff) | 180 |
| Episode ended during R5 | 0 |
| Padding applied (`R_PAD_ONCE`) | 180 |
| Input actually changed | 180 |
| No-op (the two real observations were identical) | 0 |
| First chunk identical in the two arms | 0 |
| First chunk differs, outcome equal | 170 |
| Outcome differs | 10 |

- The two real observations at the return step differ: the largest difference in the state vector is 0.003–0.014 (median 0.004), and 36–43% of the RGB values differ.
- First resumed chunk, PAD against KEEP, in **normalised controller units (not metres)**. Arm command: mean absolute difference per block 0.0002–0.0079 (median 0.0005); largest single difference per block 0.0005–0.028 (median 0.0017). Gripper command: largest difference 0–0.004; **the gripper command changed sign in 0 blocks.**
- So the padding always changed the input and always changed the generated actions, by small amounts; in most blocks the outcome was the same.

### Each R5 arm against Continue (context; these also contain the retreat itself)

| | C0 fail → arm success | C0 success → arm fail | Scene-equal mean difference | Scenes +/−/0 |
|---|---:|---:|---:|---|
| R_KEEP vs C0 | 9 of 161 | 11 of 19 | −0.035 | 3 / 2 / 7 |
| R_PAD_ONCE vs C0 | 5 of 161 | 13 of 19 | −0.078 | 3 / 2 / 7 |

### Blocks in which the cube was grasped at the checkpoint (all of them: 36 blocks, 6 scenes)

| Of 36 blocks | C0 | R_KEEP | R_PAD_ONCE |
|---|---:|---:|---:|
| Grasp flag kept through R5 | n/a | 36 | 36 |
| Open command found after the return (C0: after the checkpoint) | 32 | 33 | 33 |
| First grasp-flag release found | 32 | 33 | 33 |
| First release with cube A within 2 cm (xy) of cube B | 12 | 7 | 7 |
| First release with the on-cube flag set | 20 | 9 | 9 |
| Final success | 13 | 9 | 7 |

Positions and flags are simulator ground truth read from the step traces; they were used for logging only. No cause is attached to any row. The 2 cm label is stricter than the environment's own on-cube test (about 3.3 cm).

## 3. Was the manipulation what it was meant to be?

| Check | Result |
|---|---|
| Restore audit on seeds 100–104 (18 checkpoints), including the control-only reference, `R_KEEP` and `R_PAD_ONCE` | pass; every compared difference 0.0 |
| Every R5 branch at its return step against the control-only reference of its snapshot, before the history was touched (simulator state, step counter, history, queue, last action, action log, TCP history, flags, counters, records, RNG digests, raw observation, controller state) | 360 of 360 equal; largest difference 0.0 |
| After padding, anything changed other than the history | 0 of 180 |
| Latest history entry unchanged, earlier entry equal to it | 180 of 180 |
| Policy called during R5 | 0 times |
| First inference of the two R5 arms at the same step with the same seed | 180 of 180; seeds follow `seed_for(scene, step, repeat, 'dp')` |
| Continue with repeat 0 reproduces the never-restored trunk | 45 of 45 |
| `R_KEEP` and `R_PAD_ONCE` of repeat 0 rerun after all other branches (first two scenes) | 16 of 16 identical |
| Technical check of the first two scenes | 12 of 12 items |
| Scene placement check | 12 scenes as planned, none replaced; nearest placement on record 11.7 mm away |
| CPU tests | local 243 passed, 3 skipped; server 201 passed, 2 skipped (skips are tests whose libraries are absent on that machine) |

What differed between `R_KEEP` and `R_PAD_ONCE` at the first inference was the earlier history entry and nothing else. What was the same: the physical state, the return step (R5 took 22–28 steps), the remaining budget, the empty queue, the last action, the noise.

Scale decision: taken from measured time and credit only (10.9 s per run on this host; projected total $0.55 against the $1.00 cap), before any outcome table existed: 12 scenes.

## 4. Cost, backup, instance, Drive

| | |
|---|---|
| Credit before / after | 7.9852 / 7.6553 |
| **Total cost** | **US$0.33** (cap US$1.00) |
| of which first instance 60737590 | about $0.05: its network was far slower than advertised (about 0.5 MB/s), so it was destroyed during installation, after 22 minutes. No policy or simulator command ran on it and there was nothing to back up |
| of which second instance 60740152 | about $0.28 for 68 minutes |
| Backup | `handoff_h1_out.tgz`, SHA256 `dab6767e1b0d10ee…`; 101 of 101 files match the server-side hash list after extraction. Copy in `data_backup/handoff_h1_001/` |
| Instances | both destroyed; `show instances` returned "No instances found" after each |
| Drive | files placed in `Robot Harm/results/handoff_h1_001/` through the local sync folder. **Remote upload not verified through the Drive API** (see `DRIVE_UPLOAD_STATUS.json`) |

No key or token is in any log, in Git or on Drive.

## 5. Limits

- **12 scenes.** Blocks of one scene share the scene and its past. The 10 blocks whose outcome differs come from 6 scenes. No p-value, interval or pass criterion was computed, and none was planned.
- **Outcomes in this setup also differ under a change of noise alone.** In Stage 2B (other scenes), Continue with another future noise key had a different outcome from Continue in 31 of 336 blocks. The 8-to-2 split observed here is a count from one run, not an estimate of an effect size.
- **Not a statement about cause.** The result does not show that the real pair of observations is "contaminated" or out of distribution, nor that it is fine. It shows that replacing the earlier one by a copy of the latest did not help here and went with fewer successes.
- **Only this manipulation was run.** Nothing is known about other history lengths, repeated padding, waiting steps, or other policies, and none of these was tried.
- **The base policy is weak on these scenes** (trunk success 2 of 12; 161 of 180 blocks fail in both R5 arms), so most blocks carry no information about the contrast.
- **Novelty is not claimed.** Three papers on policy history at recovery found during preparation have not been read in full.

## 6. Not done

Training, any other condition, qualification, calibration, final Test, video inspection, post hoc subgroup tables, additional scenes. The run was executed once and is closed.

## 7. Files in this folder

| Requested name | File here |
|---|---|
| `H1_EXECUTION_REPORT.md` | this file |
| `execution_plan.json` | copy of `runs/handoff_h1_001/scope_plan.json` |
| `commands.status` | local, first-host and second-host command ledgers |
| `technical_audit_summary.json` | audit, technical check and collection checks |
| `paired_outcomes.csv` | one row per block (the runner's `pair_outcomes.csv`) |
| `scene_summary.csv` | per scene |
| `first_chunk_comparison.csv` | the chunks actually executed first in both R5 arms, with step, seed and differences |
| `handoff_inputs/` | per scene: the real two observations at the return step and the padded pair (`INDEX.csv` lists arrays and hashes); saved before any continuation |
| `backup_sha256.txt` | server-side hash list of the backup archive |
| `cost_and_instance_status.json`, `DRIVE_UPLOAD_STATUS.json` | cost, instances, upload status |
| also | `authorization.json`, `summary.json`, `pair_summary.csv`, `event_rows.csv`, `skip_reasons.csv`, `dataset.json`, `restore_audit.json`, `techcheck.json`, `scale_decision.json`, `scenes.json`, `scenes/` (12 records), `videos/` (6, fixed rule: first two scenes, step 12, repeat 0) |

Kept locally only: per-step traces, pre-intervention inputs, server logs and the archive (`runs/handoff_h1_001/`, `data_backup/handoff_h1_001/`).
