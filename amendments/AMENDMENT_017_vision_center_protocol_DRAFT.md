# Amendment 017 (DRAFT, not approved) — Protocol for the Vision-CENTER research cycle

Date: 2026-11-18 (UTC), revised 2026-11-19 after the Phase C1 offline replay. Status: **draft for discussion.** It changes nothing until the user approves it. Approving the
Phase C1 run does not approve this amendment, and this draft does not open calibration or the final Test. Calibration
(30000–30399) and Test (90000–90399) stay locked and unused.

Template fields (`templates/AMENDMENT.md`) are answered in the last section.

## 1. Why an amendment is needed

Protocol v0.3 (`docs/00`–`06`) describes a learned selector that chooses among Continue, Refresh and Retreat–Retry on the
Stage 1 policy. Amendments 011–016 moved the work, step by step and each with its own approval, to a different object:

| Protocol v0.3 | Vision-CENTER cycle (Amendments 013–016, Phase C0/C1) |
|---|---|
| Actions: Continue, Refresh, Retreat–Retry | Continue, or one bounded horizontal centring move before the release |
| A learned selector (Q or Joint heads) decides | A fixed trigger and a rule-based controller on RGB and robot state; nothing is learned |
| Base policy of Stage 1 | Vision-V1 ResNet-18 diffusion policy, train seed 1 (and train seed 2 as a second policy) |
| Candidate registry of 19 selector settings | none of them applies |
| `hci freeze` / `evaluate` / `frozen-test` built for heads, registry and selection files | cannot run these arms |

Unchanged and kept: the task (StackCube, `panda_wristcam`, CPU physics), the outcome `success_once_absorbing`, the harm
definition `H = P(Yg = 0 | Y0 = 1)`, ε = 0.05, δ = 0.05, the exact binomial test with Holm, seed-level splits, at most one
intervention per episode, no privileged input to a deployed arm, the approval token for the Test.

### 1.1 What Phase C1 adds, compared with v0.3 and with Phase B

| | v0.3 | Phase B (Amendment 016) | Phase C1 (planned, not run) |
|---|---|---|---|
| Object | learned selector over three actions | one fixed controller (v2) against v1, C0, oracle | v2 taken apart: one arm per component, on two policies |
| Base policies | Stage 1 policy | Vision-V1 train seed 1 | train seed 1 **and** train seed 2 (both frozen, 80k EMA; nothing retrained or adapted) |
| Arms | Continue, Refresh, Retreat–Retry | C0, v1, v2, oracle | C0, v1, `V2_SINGLE_FRAME`, v2, oracle, `V2_NO_HAND_MEMORY`, `V2_NO_BASE_MEMORY`, `V2_NO_HOLD_MEMORY`, `V2_RULES_OFF` (nine) |
| Confirmatory claim | H1–H3 on calibration and Test | gate on development data | H_gen and H_mem (exact paired tests, Holm, development data) |
| Data role | calibration and Test | development | development; calibration and Test locked |

`VISION_CENTER_V2S` (a separate safety candidate) was built and evaluated offline for C1 and **excluded** by its pre-set
technical criterion: it refused the 37.7 mm stale memory of scene 51009, but in scene 50623 it acted, as v2 did, on a
base-camera estimate 5.2 mm off (target not moved). Its contact rule is aimed at stale memories, not at estimator error.
It also gave up 7 moves of v2 (6 recorded v2 rescues). It was not retuned. It is not a candidate under this amendment; a
revised version would be a new candidate developed and tested on new development scenes.

### 1.2 How far a C1 result about temporal memory can reach

- `V2_SINGLE_FRAME` differs from v2 in two ways at once: no observation before the trigger, **and** a single base-camera
  estimate is accepted (confirmation count 1 instead of 3). The relaxation favours the single-frame arm, so H_mem is
  conservative, but v2 minus single-frame is "memory together with the three-view confirmation", not memory alone.
- The offline replay on the 500 used scenes (484 triggered) shows the expected direction: v2 moves in 26 scenes where the
  single-frame arm abstains and never the reverse; 20 of the 26 are recorded v2 rescues. These are used scenes and are
  not evidence for C1.
- The leave-one-out arms describe what each memory source carries. They overlap (without hand memory the base memory
  stands in, 3 scenes one way and 2 the other), so their differences do not add up to the total and are not tested.
- A rejected H_mem would hold for this task, these two policies trained on the same 500 demonstrations with the same
  architecture, these cameras and this trigger. It says nothing about other tasks, real images or other policy families.
- Memory brings its own failure: a remembered position can be stale (51009: the target was pushed 38.8 mm after the last
  view). v2 has no rule against it; the safety criteria in C1 count such moves but do not prevent them.



Question: can a fixed, observation-only centring move, triggered at the release, raise the success rate of a frozen
manipulation policy while the rate at which it turns a success into a failure stays below ε?

- **HA (benefit):** the candidate succeeds in more scenes than the uninterrupted policy.
- **HB (bounded harm):** `P(candidate fails | policy alone succeeds) ≤ ε`.
- **HC (mechanism):** the gain over the single-frame controller comes from positions remembered from earlier frames.
- **HD (generality over policy seeds):** HA and HB hold for a second policy trained on the same data.

The v0.3 hypotheses H1–H3 concern learned selectors and are not tested in this cycle. They are neither confirmed nor
rejected by it.

## 3. Intervention

- Trigger: an open command pending directly after at least three close commands, fingers closed on something
  (`hci.recovery_v1_oracle.trigger_rule`). Observable quantities only.
- Action: horizontal move of at most 5 mm per step, at most 12 steps, commanded target within 30 mm of the TCP at the
  trigger, vertical command zero, gripper command held. Steps are charged to the 200-step episode. Then the queue is
  cleared and the unchanged policy re-plans.
- Abstention: nothing is touched; the episode is the uninterrupted one. This is checked in every scene.
- Inputs: both RGB images, TCP pose, finger joints, executed gripper commands, step counter. No cube pose, grasp flag,
  task flag, outcome or future information.
- `CENTER_ORACLE` reads simulator positions. It is a reference and is never a candidate.

## 4. Candidates and their selection

- Candidates are fixed controllers, identified by file hashes: `VISION_CENTER_V2` (plan `1b4ea3bb…`, commit `fd23aa2`).
  `VISION_CENTER_V2S` was excluded in the C1 preparation (section 1.1) and is not a candidate.
- Ablation arms (`V2_SINGLE_FRAME`, the three leave-one-out arms, `V2_RULES_OFF`) are diagnostics and never candidates.
- The candidate list for calibration is declared after Phase C1, from development data only, and is hashed before any
  calibration rollout. It cannot grow or change afterwards.
- Proposed default: one candidate. If two are declared, both are tested with Holm and, among those that pass, the one
  with the higher development success rate is selected, as in `docs/03`.
- A candidate changed after a calibration result turns that calibration into development data and needs a new set.

## 5. Risk test

- Unit: one initial scene. `n₊` = calibration scenes the policy solves alone; `k` = those the candidate loses.
- `p = BinomialCDF(k; n₊, ε)`; Holm over the declared candidates at δ. `n₊ = 0` does not pass.
- Y0 and Yg come from one snapshot of the same episode in one session on one host (uninterrupted outcomes differed
  between hosts in 16 of 300 scenes in Phase B).
- Each policy seed is calibrated and reported separately. A statement covering both policies counts both in the Holm
  family.
- Reported next to the certificate, without a guarantee: harm among the scenes in which the candidate moved, memory
  errors, target pushed, abstentions by reason.
- Benefit (HA) on the Test: exact paired one-sided test of the selected candidate against the uninterrupted policy,
  declared beforehand; separate from the risk certificate.

## 6. Use of the splits

| Set | Seeds | Use |
|---|---|---|
| Development, used | 50000–51099 | no further evidential use |
| Development, Phase C1 | 51100–51299 | ablation and second policy; development only |
| Calibration | 30000–30399 | one calibration of the frozen candidate list |
| Test | 90000–90399 | one final Test, after explicit approval |

Initial placements of the calibration and test scenes are compared with all placements on record before use (no policy).

## 7. Freeze

Before calibration a frozen folder records, by SHA256: this amendment as approved; the candidate list; controller,
runner and analysis code; both policy files; configuration; seed registries; ε and δ; the full `pip freeze` and driver
of the host; the upstream commit. The manifest is committed, pushed and copied to Drive before the first calibration
rollout. `V2_FREEZE_MANIFEST_DRAFT.json` (Phase C0) is the starting point and is not a freeze.

A new runner for these arms replaces `hci evaluate` / `hci frozen-test` for this cycle. It must keep every existing
refusal: no test split without the approval token, verification of all frozen hashes, and the started marker.

## 8. One Test

- The Test runs once, with `RUN_FROZEN_TEST` supplied only after the user's explicit approval.
- A set-level ledger for the test seeds is created exclusively before the first Test rollout; a second creation fails.
- `TEST_STARTED.json` is written before the first episode and is never deleted.
- The analysis script is part of the frozen manifest; a partial run is not summarised.
- A failed or interrupted Test is reported as it is. A changed setting is a new research version with a new held-out
  set; the old result stays in the record.
- These are operational safeguards against accidental contamination, not access control.

## 9. Procedural questions to settle before any calibration

1. **Candidate list.** Declare it from development data only (after C1) and hash it. Proposed: v2 alone.
2. **Which policy the certificate is for.** Train seed 1, train seed 2, or both (both: one Holm family, a separate
   calibration count for each, and roughly twice the calibration compute).
3. **Calibration size and power.** With 400 calibration scenes and about 330 that the policy solves alone, the risk test
   at ε = δ = 0.05 passes with at most 9 harms (2.7 %). If the true harm rate is 2 %, it passes with probability about
   0.87; at 3 %, about 0.47. If this power is too low, the set has to be enlarged before calibration. It cannot be
   enlarged afterwards.
4. **Runner.** A calibration and Test runner for these arms does not exist yet. `hci evaluate` and `hci frozen-test`
   cannot run them. The new runner must keep every refusal listed in section 7 and must be tested before the freeze.
5. **Placement checks.** Initial placements of 30000–30399 and 90000–90399 are compared with every placement on record.
   This needs the simulator (no policy) and is a prerequisite of the calibration run, as it is for C1's own 200 seeds.
6. **Host and session.** Uninterrupted outcomes differed between hosts in 16 of 300 scenes. Fix one host type, and run Y0
   and Yg in the same session from one snapshot.
7. **Memory-error reporting.** Under the C1 definition (a move on a remembered position with offset error over 5 mm or
   outside its own bound), v2 made 3 such moves in the 484 triggered used scenes (5.02, 5.08 and 5.23 mm). Decide before
   calibration whether this quantity is only reported or also limits the candidate.
8. **Revision of `docs/00`–`06`** to the text of this amendment, and approval of the amendment itself.

## 10. Not covered

Real images, other tasks, repeated interventions, learned selectors, and any claim about physical safety. H is the
worsening of a task outcome in simulation.

## 11. Template fields

- **Existing version:** v0.3 (2026-11-12) with Amendments 001–016.
- **New version:** v0.4-draft, this document. Not in force.
- **Time of change:** proposed after Phase C0, revised in the Phase C1 preparation, before any calibration.
- **Reason:** the object under study is no longer the one the v0.3 freeze and calibration code describe.
- **Files and settings affected:** `docs/00`, `docs/02`, `docs/03`, `docs/05`, `docs/06` (to be revised on approval),
  `configs/candidates.json` (superseded for this cycle), a new calibration and Test runner.
- **Results seen before the change:** development results only (Stages 0–3, H1, Vision-V1, Recovery-V1, Vision-CENTER v1
  and v2 on seeds 50000–51099, and the C1 offline replay on stored frames of 50600–51099). No calibration and no Test result exists.
- **Validity of earlier calibration:** none was run; nothing is invalidated.
- **Data to be generated:** Phase C1 development scenes; later one calibration and one Test.
- **Checks to re-run:** full local tests; restore and reproduction checks of every arm; placement checks of new sets.
- **Where earlier results are kept:** `results/`, `amendments/`, Drive `Robot Harm/results/`; unchanged.
- **Scope of approval requested:** none yet. This draft is submitted for comment.
