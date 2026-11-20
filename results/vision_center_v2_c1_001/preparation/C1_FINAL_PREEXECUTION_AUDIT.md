# Phase C1 — Final Pre-Execution Audit

Date: 2026-11-20 (UTC). Work id `vision_center_v2_c1_001`.

**Scope.** Free, local work only. There was no paid GPU, no GPU instance, no simulator step, no new scene (51100–51299
untouched), no calibration and no Final Test. The plan, policies, arms, thresholds and controller code are unchanged.

## 12. Final verdict: `READY_FOR_APPROVAL`

The verdict rests on four findings:

- The plan `dd3af27e…` is unchanged and still binds the working tree: 43 of 43 frozen items are equal.
- The run entry point now has end-to-end tests on Linux:
  - approval and plan refusal;
  - start and five-scene budget gates;
  - the per-scene budget check;
  - SIGALRM inside a scene;
  - ordinary errors;
  - ledger and no re-run;
  - no summary of a partial run.

  All pass.
- The post-hoc C1 trajectory audit runner exists, handles mock C1 folders, and received GO from Codex review 15.
- All CPU tests pass: Windows **417 passed, 13 skipped, 0 failed** (exit 0, 1 h 18 min; log sha256 `ac700db5…f49091`); Linux (WSL) 24 of 24 for the new tests.

The verdict is not unconditional. It needs the user's explicit approval naming a total cap **and** plan SHA256
`dd3af27e5484b9ba395820470d5adf5c7865a820514dd23964483c06ecd699a6`. The limits listed in §9 and §10 are accepted, not
removed.

## 1. Commits and changed files

| Item | Value |
|---|---|
| Before | `ac909960880056011b94b6152afdd8d52cac5a30` (`main`) |
| After | the commit that adds this file (hash in the commit log; reported in the Drive copy) |

| File | Change | Why | Hashed by the plan? |
|---|---|---|---|
| `tests/test_c1_run_integration.py` | new | Task A: `cmd_run` from its entry point, real SIGALRM on Linux | no |
| `scripts/35_c1_trajectory_audit.py` | new | Task B: post-hoc C1 trajectory audit runner | no |
| `tests/test_c1_trajectory_runner.py` | new | Task B: mock C1 folder, injected faults | no |
| `README.md` | +96 lines | Task C: top GIF and a "Visual evidence" section; one table-of-contents line | no |
| `results/vision_center_v2_c1_001/preparation/full_tests_preexec3.log` | new | Task D: raw output of the full CPU test run | no |
| `results/vision_center_v2_c1_001/preparation/C1_FINAL_PREEXECUTION_AUDIT.md` | new | this report, at the path the task names | no |

- No existing file in `src/`, `configs/` or `scripts/` was changed. No policy, registry, plan or manifest was changed.
- No file was deleted or overwritten.
- No git history was rewritten, and nothing was force-pushed.

## 2. Task A — the run entry point, end to end, with a real SIGALRM

**Call path traced** (`scripts/34_vision_center_c1.py`):

1. `cmd_run` checks `require_authorization`, which requires a user approval, a positive cap and the plan hash.
2. `bound` checks the plan's own checks, its 35 code hashes, both policies and both registries.
3. The gate files are checked: `scenes.json`, `audit/audit.json` and `budget_gate.json` must be passed and name this plan.
4. The ledger must be absent, and so must the pass folders.
5. The start projection runs.
6. The ledger is written exclusively, and copied to `run/ledger.json`.
7. `run_passes`:
   - `arm_deadline` computes the time left from the session clock: cap − reserve − earlier spend. It registers the SIGALRM
     handler that raises `BudgetDeadline` (a `BaseException`) and sets `setitimer`.
   - `budget_deadline.json` is written.
8. For each policy, `passes` handles each scene:
   - the per-scene check: spend + one scene at its bound + reserve must fit the cap; otherwise `budget_stop.json` and
     `BUDGET_STOP`;
   - `collect_scene` and `save_scene`;
   - on any `Exception`, an `<seed>.error.json` is written and the exception is re-raised;
   - after five scenes, the five-scene gate runs.
9. `BudgetDeadline` writes `budget_stop.json` (kind `deadline`, policy, scene, scenes finished) and raises a `SystemExit`
   whose message contains `INCOMPLETE`. `finally` disarms the timer.
10. `cmd_summarize` runs `check_run_gates`, which refuses on any `budget_stop.json`. `load_records` refuses on error
    files or missing records.

`run.sh` writes `RUN INCOMPLETE` into the status file when `run` fails.

**Test design.** Everything above runs unchanged. Only these are replaced:

- `LEDGER` points into the test's temporary folder;
- `runtime_for` returns the mock runtime (mock physics, synthetic renderer, scripted policy);
- `X.gt_probe` is the mock probe;
- `C1.pilot_seeds` returns synthetic seeds 7, 8, 9, …;
- `bound()` returns the test plan, a copy of `plan_v2.json`. The real `bound` is tested separately.

The placement check and the server audit need ManiSkill, so the test writes their result files as passed. The session
clock and the budget gate come from the real `cmd_clock` and `cmd_budget`. Every test checks that the real ledger and
the real run folder are unchanged.

Command (Linux, WSL2 Ubuntu, Linux 6.6.87.2, Python 3.12.3, numpy 2.4.6, torch 2.15.0+cpu, pytest 9.2.0):

```
C1_IT_EVIDENCE=<dir> python -m pytest -p no:cacheprovider -rA -v tests/test_c1_run_integration.py tests/test_c1_trajectory_runner.py
```

Result: **24 passed** in 16 min 51 s. Log sha256 `84f534887cc0bb09…c4fb119`. Each test's created files and their sha256
are in `evidence/<test>.json`.

| Case | Test | Expected | Observed | Status |
|---|---|---|---|---|
| 1. normal mock run | `test_1_normal_run_ends_with_a_summary_and_7_a_second_run_is_refused` | both passes, 2 scenes each, `result.json`; no stop or error file; timer disarmed | as expected. `result.json` from the real `C1.summarize`; `n = 2` for all 9 arms and both policies; deadline at start more than 1 h | PASS |
| 2. SIGALRM during the run | `test_3_4_…` | the run's own timer, armed from the session clock, interrupts a scene | timer at scene 8 equal to `deadline_seconds_left` from the clock (within 2 s); then shortened by the test to 4 s; `BudgetDeadline` raised **inside `hci` code** (5 frames deep in `src/hci`) | PASS |
| 3. deadline inside a scene | same | scene 8 interrupted; no record of it | `budget_stop.json`: kind `deadline`, scene 8, 1 scene finished; scene 7 kept with a valid trace hash | PASS |
| 4. no scene after the deadline | same | scene 9 and pass 2 never start | events exactly start 7, end 7, start 8, interrupted 8; no `pass_train_seed_2` | PASS |
| 5. recorded as `INCOMPLETE` | 2–4, budget check, five-scene gate | stop file and an `INCOMPLETE` stop message | as expected | PASS |
| 6. no summary of a partial run | 2–4, budget check, 9 | `cmd_summarize` refuses; no `result.json` | refused (`budget stop`, `INCOMPLETE: no record`, `INCOMPLETE: error files`) | PASS |
| 7. re-run with an existing ledger | 1, 2–4, 9, five-scene gate | refused | `… exists`. With the ledger removed, the pass folders still refuse (`never resumed`). Run folder unchanged. | PASS |
| 8. missing or wrong approval or plan | `test_8_no_approval_…`, `test_8_the_real_bound_…` | refused before any file is written | refused for each case: no approval; another plan hash; not the user; cap 0; not authorized; plan changed after approval; gate file of another plan. Real `bound` accepts the real plan and refuses one changed code hash. | PASS |
| 9. ordinary exception | `test_9_an_ordinary_error_in_a_scene_fails_closed` | same fail-closed rule | `8.error.json` (`completed: false`); scene 7 kept; no pass 2; timer disarmed; ledger present; summary refused; re-run refused | PASS |
| per-scene budget check | `test_the_budget_check_before_a_scene_stops_…` | after scene 7 the clock is moved 13 h back; scene 8 must not start | `pass/budget_stop.json`, before scene 8, 1 finished | PASS |
| gates versus deadline | `test_2_5_6_…`, `test_2_a_…` | see below | as described below | PASS |

**What was learnt about the deadline** (first attempt failed, kept in `superseded_attempts/`):

- **A pure clock deadline cannot be reached in a short test.** The budget gate's projection includes a fixed 900 s × 1.25
  overhead, so a deadline less than about 19 min after the gates is refused by the start gate (`test_2_a`).
- **A 20-minute attempt was stopped earlier.** With the clock deadline set 1200 s after the clock, the run was stopped
  first by the five-scene gate (`BUDGET_BLOCKED after five scenes … INCOMPLETE`), from the measured scene time
  (`test_2_5_6`).
- **The deadline is therefore a backstop.** With the times measured in the first five scenes, the gates stop a run before
  the clock deadline. The deadline fires only if later scenes are slower than those first five.
- **What is and is not shown.** Its in-scene handling is tested with the run's own handler and timer, shortened by the
  test (`test_3_4`). A natural clock expiry in the middle of a scene was **not** observed. The code path is the same; the
  start time of the timer is the only thing the test changes.

**Windows.** The 6 SIGALRM tests are **SKIPPED** there (no `setitimer`) and are not counted as passed. The 2 refusal
tests run and pass.

## 3. Task B — the C1 trajectory audit runner

`scripts/35_c1_trajectory_audit.py` is post-hoc only:

- It reads the folder layout written by `collect_scene` and `save_scene`, and nothing else.
- It runs nothing.
- It refuses an output folder that is the run folder, is inside it, or contains it, and writes nothing into the run.
- It is not hashed by the plan and not read on the server.

**What it audits**, for each policy, scene and arm:

- The ten behaviour checks of `C1_BEHAVIOR_EXPECTATION_SPEC.md`, through `hci.trajectory_audit.check_arm`, unchanged.
- Record checks:

| Check | What it requires |
|---|---|
| `R_files_and_hashes` | trace and frames files present, with matching sha256 |
| `R_record_identity` | completed, scene, role, policy, plan hash, the nine arms in order, `validate_scene` |
| `R_branch_and_snapshot` | `seed:policy:t<trigger>:arm` and one snapshot digest |
| `R_frame_index` | trunk frames = trigger + 1 (or = trunk steps without a trigger); exactly one control frame per reading; after-trigger frames listed by `post_frame_steps`, ordered, unique, ending at the last step; abstaining arms store none; duplicate consecutive frames counted |
| `R_abstain_pointer_to_C0` | the pointer, and the trace recomputed against C0 within `state_atol` |
| `R_events_match_trace` | first open command and success step from the record equal those from the trace and the timeline; outcome and length |
| `R_handback` | logged control steps; first policy inference exactly at trigger + k |
| `R_control_tcp_sync` | the TCP from the observation equals the trace |

It also reports the target displacement and the offset before and after each move, the image sync check and the case
lists.

**Outputs:**

- `audit_summary.json`, with no time stamp: equal inputs give an identical file (tested);
- `per_scene_checks.csv`;
- `trajectory_manifest.json`: every input with sha256 before and after, and every output;
- per selected scene: 3D path, XY view, action and gripper timeline, outcome strip, recorded RGB and POST-HOC overlay
  GIFs, and key frames. Steps with no stored frame are shown as NO RGB RECORDED.

**Status:**

| Status | When |
|---|---|
| `COMPLETE` | every planned seed of both policies, no stop or error file, every R check passed, every figure drawn, run folder unchanged |
| `SUBSET_ONLY` (exit 4) | `--seeds` narrows the plan |
| `INCOMPLETE` (exit 3) | anything else, including unreadable, missing or changed inputs, malformed records and a malformed ledger |

**Review.** Codex reviews 7–14 returned NO-GO, with 12 MAJOR and 3 MINOR findings in total on status soundness and crash paths.
Each was fixed and tested. Review 15 returned **GO**, with no CRITICAL, MAJOR or MINOR finding left. Prompts and outputs
are in `codex_reviews/`.

**Limit.** The RGB panels of `hci.trajectory_audit.rgb_panels` show C0 and the arms that moved. An abstaining arm has no
frames of its own: its pointer to C0 is checked and listed, not drawn as a separate column.

## 4. Mock data audit

Two kinds of mock folder were used:

- **Test fixture:** both policies, nine arms, synthetic seeds 7–8, with a mock plan whose seed range is 7–8.
- **Demonstration folder:** the real plan, seeds 51100–51299.

**Test results** (Windows 16/16 and Linux 16/16):

| Requirement | Test | Result |
|---|---|---|
| both policies and nine arms recognised | `test_complete_mock_run_…` | PASS: COMPLETE; 2 records per policy; every R check PASS; figures made |
| a complete mock run audits clean | same | PASS |
| missing arm or scene found | `test_a_missing_arm_or_scene_is_found` | PASS: arm dropped, record deleted, pass folder deleted and trace deleted each give INCOMPLETE |
| wrong snapshot digest found | `test_a_wrong_snapshot_digest_or_branch_id_is_found` | PASS |
| frame index mismatch found | `test_a_frame_index_mismatch_is_found`, `…control_frame_count…` | PASS: a step dropped, a step repeated, a reading removed |
| abstaining arm differing from C0 found | `test_an_abstaining_arm_that_differs_from_C0_is_found` | PASS: trace changed with a consistent hash; R check and behaviour check 9 both fail; INCOMPLETE |
| interrupted run not taken as complete | `test_an_interrupted_run_is_never_taken_…` | PASS: run-level stop, pass-level stop, error file, no ledger, record of another plan |
| same input gives the same summary | `test_equal_inputs_give_an_identical_summary…` | PASS: byte-identical `audit_summary.json` and CSV |
| originals not modified | same, plus every test | PASS: the mock folder's bytes are compared after each audit; the runner also hashes the run folder before and after |
| robustness (review findings) | malformed record, missing frames file, broken ledger, figure failure, run folder changed, unreadable file, listing error, missing run folder | PASS: each gives an INCOMPLETE summary, never a crash and never COMPLETE |

**Injected faults versus real defects:**

- All faults above were introduced on purpose into copies.
- The real defects found in the runner came from Codex reviews 7–14, not from these tests, and were fixed in the runner.
  - Partial-seed status; output inside the run; failed record checks still COMPLETE.
  - Loose frame and hand-back rules.
  - Crash paths: malformed record, ledger, figures, run-folder digest, listing, missing run folder.
  - Missing frames on scenes without a trigger.
- No defect was found in the run code (`34_vision_center_c1.py`, `vision_center_c1.py`).

**Demonstration** (`preexecution_validation/runner_mock_demo/audit_output_final/`, final runner):

- The status is INCOMPLETE because the folder's records (seeds 7, 8) lie outside the real plan's seeds, which is
  correct. Every R check passes.
- Behaviour check 8 (the correction kept after the hand-back) fails for several arms. This is a property of the mock's
  scripted policy, which drifts back after the hand-back. It is not a finding about v2.

## 5. Task C — README and GIF verification

**What was added.** `README.md` received:

- `recovery_success.gif` at the top only;
- a `## Visual evidence` section after `## Current status`, with a table-of-contents entry, describing scenes 50632
  (rescue), 51004 (unsafe move versus abstention) and 51009 (stale memory).

**What it states:**

- the GIFs come from stored records, with nothing re-simulated and no frame generated;
- single episodes, not statistics;
- the oracle is a privileged reference;
- post-hoc ground truth versus controller estimates (×);
- NO RGB RECORDED placeholders.

**Verification:**

- **Frames and events.** `assets/research_gifs/VERIFICATION.json` was re-checked, with `all_pass: true`. In it, all 144
  shown frames are closest to their own step's raw frame, all 154 placeholders are grey, and numbers and events equal the
  raw trace.
- **README numbers.** Every number in the README was compared with the raw record:
  - 50632: 25.6 → 4.6 mm, 11 steps, target 0.02 mm, resume 124, open and success 127, C0 opens 113;
  - 51004: 51.1 → 47.7 mm, push 5.69 mm, `SIDE_CONTACT_RISK`, identical to C0;
  - 51009: memory 37.69 mm, target moved 38.79 mm, estimate 14.4 mm, true 47.8 mm, oracle push 20.72 mm,
    47.8 → 47.1 mm.
- **One wrong statement found and corrected.** The draft said v2 used "the current frame each time" in 50632. The record
  shows `frame` for the readings at steps 112–114 and `memory_hand` for steps 115–123. The README now says so, and the
  Drive draft carries a correction note.
- **Plan.** The README is not hashed by the plan.

## 6. Full CPU tests

| Platform | Command | Result |
|---|---|---|
| Windows 11, Python 3.14.4 | `python -m pytest -q -p no:cacheprovider -rfs tests` → `full_tests_preexec3.log` | **417 passed, 13 skipped, 0 failed** (exit 0, 1 h 18 min; log sha256 `ac700db5…f49091`) |
| Linux (WSL2) | the two new files (above) | **24 passed** |

**Skipped on Windows:**

- 7 skips need a library that is not installed: h5py, imageio, torchvision, mani_skill, gymnasium. These are the same 7
  as before.
- 6 skips are the SIGALRM tests, which need Linux and passed on Linux.

**Earlier baseline.** The baseline was 399 passed and 7 skipped. No earlier test was deleted or weakened.

**The three earlier test fixes**, checked again:

| Test | Earlier wrong assumption | Status now |
|---|---|---|
| `test_session_clock_…` | a wrong expected spend | Now 0.25 + 10 h × 0.5 = 5.25, the clock rule; early fits, late refused. Still checks the requirement. |
| `test_validate_scene_rejects_inconsistent_records` | the altered field was not one `validate_scene` checks for abstaining arms | It now alters an abstaining arm's outcome away from C0, which `validate_scene` must reject. Still checks the requirement. |
| `test_scene_with_every_arm_…` | `V2_SINGLE_FRAME`'s `memory_target` can be `None` on the mock scene | The assertion holds trivially when it is `None`. The requirement (`confirm_min` 1, one base view) is checked directly by `test_single_frame_memory_holds_the_trigger_step_only_and_accepts_one_base_view`, so it is still covered, but not by that line. |

**Process note.** A first full Windows run gave 409 passed and 13 skipped with exit 0, before the Codex fixes to the
runner. Its log was then damaged by a faulty masking script of mine, which inserted characters when it found nothing
to mask.

- The damaged log was not committed. It is kept as superseded.
- The script now refuses when there is nothing to mask.
- One later full run (`preexec2`) was stopped because the runner changed during it.
- `full_tests_preexec3.log` is the only valid final log.

## 7. Plan SHA256

| Copy | sha256 | Equal to `dd3af27e…`? |
|---|---|---|
| `results/vision_center_v2_c1_001/preparation/plan_v2.json` | `dd3af27e5484b9ba395820470d5adf5c7865a820514dd23964483c06ecd699a6` | yes |
| `runs/vision_center_v2_c1_001/plan.json` (server copy) | same | yes |

**The plan was not regenerated and remains valid.**

## 8. Frozen artifacts, before and after

The plan's recorded hashes were compared with the files, at the start of this work, mid-way and after the final tests.
All **43 of 43** items are equal each time:

- the 35 code files of the plan (v1, v2, oracle, C1 module and driver, `run.sh`, statistics code, configs);
- policy files, 4 items:

  | Policy | `weights.pt` | `policy_meta.json` |
  |---|---|---|
  | train seed 1 | `a37be256…` | `4ccfba34…` |
  | train seed 2 | `52eff6a9…` | `c3606292…` |

- the arm registry `c732aad5…` and the seed registry `bb5416b9…`;
- both plan copies.

Script: `runner_mock_demo/frozen_check.py`. Outputs: `frozen_check_mid.json` and `frozen_check_final.json`.

## 9. Open problems

| Severity | Item |
|---|---|
| CRITICAL | none |
| MAJOR | none |
| MINOR | `ArmView` is an access guard, not a sandbox (ACCEPTED_LIMITATION, unchanged) |
| MINOR | The US$0.30 reserve does not bound spend after the deadline (ACCEPTED_LIMITATION, unchanged) |
| MINOR | A natural clock expiry in the middle of a scene was not observed. The handler path is tested with a shortened timer; the gates stop earlier when timing is as measured. Was OPEN_MINOR "no `cmd_run` deadline test": now **RESOLVED** with this caveat. |
| MINOR | **The new tests run on the paid server.** `run.sh env` runs the full test suite, so the new tests also run there. In WSL they take about 17 min; on the RTX 4090 host's CPU probably less. This adds roughly US$0.05–0.13 to the run. If any of them fails there, the run stops before any scene (fail-closed). |
| MINOR | The RGB panels draw no separate column for abstaining arms (pointer to C0 checked, not drawn) |
| MINOR | The trajectory runner has been reviewed by Codex (GO) but not yet run on real C1 data, which do not exist |

## 10. Checks possible only on the GPU server

- CUDA and Vulkan, rendering, and the frozen diffusion policies (`run.sh env`: `hci doctor`, versions, server pytest).
- Restore exactness and C0/trunk identity in ManiSkill (server audit on 50400–50402).
- Placement duplicates of the 200 new seeds (`run.sh scenes`).
- Real timing for the budget gate, and real SIGALRM behaviour inside ManiSkill code.
- Storage size and transfer cost of a real run.

## 11. Cost and approval status

| Item | Value |
|---|---|
| Expected | US$1.65–2.48 at US$0.4604/h, plus about US$0.05–0.13 for the new tests in the server test step |
| Worst case of the gate projection | US$4.60 (8.7 h) |
| Proposed cap | **US$6.00, a proposal only. Not approved.** No earlier approval carries over. |

**Status:** no `authorization.json` exists; paid compute is `NOT_AUTHORIZED`; no instance was created.
