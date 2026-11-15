# Protocol change record 011 (DRAFT) — Handoff H1: one-shot latest-observation padding after R5

**Status: draft. Preparation only. No policy inference, no simulator rollout and no paid compute has been run or authorised under this record.** It becomes effective for execution only if the user approves a run; that approval is recorded separately (section 7).

Previous version: v0.3 + AMENDMENT_001–010 (repository HEAD `874d26e`)
Date of draft: 2026-11-15 (UTC).
Reason: an external feedback review suggested that the policy's input at the moment control is handed back after a recovery motion may matter. The user's instruction `prompts/AGENT_PROMPT_HANDOFF_H1_v1_0.md` asks for the literature to be checked against the original papers and for one narrowly defined contrast to be implemented and tested locally.

This is a follow-up hypothesis, separate from Stage 3B. It was designed after the Stage 2B–3B results had been seen. It does not reopen the selector search.

## 1. Status

```text
legacy_stage3b: COMPLETED_DEV_NO_GATE_PASS
legacy_model_search: CLOSED_FOR_THIS_CYCLE
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
handoff_h1: PREPARATION_ONLY
new_training: NOT_RUN
new_policy_inference: NOT_RUN
new_simulator_rollouts: NOT_RUN
paid_compute: NOT_AUTHORIZED
calibration: NOT_RUN
final_test: NOT_RUN
```

## 2. Question

With the fixed Diffusion Policy, right after R5 ends, and with the physical state, the return step, the remaining budget and the policy noise held equal: does changing the observation history of the first re-inference from `[previous, latest]` to `[latest, latest]` change the generated action and the later task outcome?

Neutral name: **one-shot latest-observation padding at handoff**. "History contamination" is an untested hypothesis. No improvement is assumed and no out-of-distribution cause is taken as given.

## 3. Fixed conditions

Environment and policy as in Stage 2: pinned StackCube-v1, D500-clean 80k EMA policy (same weight and metadata hashes), horizons 2 / 8 / 16, 100 DDPM steps, 200 control steps, `success_once_absorbing`. R5 unchanged: 5 cm up, command increment at most 5 mm per step, arrival tolerance 5 mm, at most 40 steps, last gripper command held. Checkpoints 12 / 52 / 92 / 132, block repeats 0–3. Restore by reset plus action replay, never by `set_state_dict`. Audit tolerances unchanged.

## 4. Arms

| Arm | What happens |
|---|---|
| `C0` (`continue`) | Existing Continue. Queue and history untouched. Context only. |
| `R_KEEP` (`r_keep`) | Existing R5 through `Runtime.force(2)`; the policy resumes on the real last two observations. |
| `R_PAD_ONCE` (`r_pad_once`) | The same R5. Immediately before the first inference, and only once, the earlier history entry is replaced by an independent copy of the latest entry. |

Primary comparison: `R_PAD_ONCE` against `R_KEEP`. Comparisons with `C0` also contain the retreat, the discarded queue, another inference schedule and the spent budget; they are context.

Rules of the padding:

- The whole converted observation entry the policy receives is copied (both tensors: RGB of all cameras, and qpos / qvel / TCP pose). No partial ablation.
- The latest entry is left untouched, byte for byte. The earlier entry is an independent copy with no shared memory.
- Nothing else is written: not the simulator, the controller, the raw observation, any object pose, the time, the queue (already empty after R5), the last action, the intervention record, the inference count or any RNG.
- qvel is not set to zero. A duplicated observation is not a physical stop and is not described as one.
- No environment step, waiting or holding is added.
- `policy.reset()` is not used, because it also clears the queue, the last action and the counters. Only `policy.history` is assigned.
- After the first inference the ordinary `observe()` FIFO continues with real observations.
- If the episode ends during R5, nothing is padded and the policy is not called; the block is kept with `padding_applied=false`, `policy_resumed=false`.
- If the last two observations are identical, the padding is a no-op; such blocks are kept.
- Padding is never decided by an outcome, an object state or a scene property.

Noise: all arms use `seed_for(scene, physical step of the call, block repeat, 'dp')` through the unchanged `generate()`. `R_KEEP` and `R_PAD_ONCE` make their first inference at the same step with the same seed. Equal actions for different inputs are allowed.

## 5. Checks built into the run

- For every snapshot the R5 control part is run once alone, without inference, as a reference. It fixes the return state and the handoff inputs before any outcome of that snapshot exists.
- Every `R_KEEP` and `R_PAD_ONCE` branch is compared with that reference at the return step, before the history is touched: runtime tree (simulator state, step counter, history, queue, last action, action log, TCP history, flags, counters, intervention and retreat records, RNG digests), raw observation, controller state. A difference stops the scene as a technical fault.
- After the padding the runtime is compared exactly with what it was: only the history entries may differ, the latest entry must be unchanged and the earlier one equal to it.
- First inference step and seed are equal in the two R5 arms; no inference during R5.
- Continue with repeat 0 reproduces the never-restored trunk at every snapshot. In the first two scenes `R_KEEP` and `R_PAD_ONCE` of repeat 0 are run a second time after all other branches.
- Server audit on seeds 100–104: the Stage 2A audit plus restore and trace after the reference and after both R5 arms.

## 6. Planned run (not started)

- At most 12 new train scenes with no recorded use, in ascending order: by the usage ledger these are 10060–10071 (to be confirmed against stored cube placements at the server step). Role `handoff_h1_exploratory_dev`; never moved to calibration or Test. Earlier harm or rescue cases are not picked as the main sample.
- At most 4 checkpoints × 4 repeats × 3 arms per scene: **at most 576 continuations**, plus 12 trunks, at most 48 reference control runs, at most 16 reruns, and the 5 audit scenes.
- The first two scenes (at most 96 continuations) are a technical and throughput check. Reduction order 12 → 6 → 2 by time and cost only, never by outcomes.
- Proposed total cap US$1.00, including install, transfer, audit and backup. This is a proposal, not an approval. Caps of earlier stages are not reused.
- Calibration and Test are compared by seed number only. Qualification sets are not rerun.

Analysis fixed in advance: `Y_PAD − Y_KEEP` within a block, averaged within a scene and then with equal weight per scene; the four cells with blocks and scenes; per-scene +/−/0; first resumed chunk differences in normalised controller units; counts of blocks with a real handoff, a changed input, a no-op; rescue and harm of each R5 arm against `C0` with denominators; for all blocks grasped at the checkpoint, first open command, first grasp-flag release and outcome in all three arms. No p-value, no pass criterion, no risk bound.

Stopping: one run and one report. No other history length, padding frequency, waiting step, model or distance is added afterwards, whatever the result. A difference would be a difference under a controlled manipulation of the resume input; it would not establish OOD, mode collapse or any single cause.

## 7. Authorisation

Execution commands of `scripts/26_handoff_h1.py` (`scenes`, `audit`, `collect`, `techcheck`, `scale`, `finalize`) refuse to run unless the run folder contains `authorization.json` with `approved_by: "user"`, `paid_compute_authorized: true`, a positive `cap_usd` and the SHA256 of the plan. No code in the repository writes that file. The agent does not create its own approval.

## 8. Code

| File | Content |
|---|---|
| `src/hci/handoff_h1.py` (new) | the padding, the return-state checks, the three-arm collector, audit, tables |
| `scripts/26_handoff_h1.py`, `scripts/handoff_h1/run.sh` (new) | `plan`, `inspect-inputs`, `dry-run`, and the guarded execution commands |
| `tests/test_handoff_h1.py` (new) | 20 mock tests |

`simulation.py`, `policies.py`, `pilot.py`, `pilot2b.py`, `pilot2d.py` and all completed stage files are unchanged (same hashes as in the Stage 2D plan).

## 9. What preparation established and what it did not

- The saved Stage 2B and 2D records contain pre-intervention inputs only. The two observations at the return step were never saved, so the contrast cannot be read from existing files: `REPLAY_REQUIRED`.
- None of the six priority papers contains this contrast. Three uncatalogued papers on history at recovery were found and not read in full; novelty is not claimed.
- Nothing about the effect of the padding is known. Mock tests show that the code does what this record says; they say nothing about the simulator or the policy.
