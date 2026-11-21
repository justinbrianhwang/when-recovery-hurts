# C1 analysis readiness

Reference plan `6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7`. No C1 run exists. **This is not an approval of a paid run.**

## 1. How a C1 run is read

**Command:**
```
python mechanism_analysis/run_analysis.py c1 --run <C1 run folder> --out <new folder>
```

**What the run folder holds.** The driver writes `pass_train_seed_1/` and `pass_train_seed_2/`, each with:
- `scenes/<seed>.json`;
- `traces/trace_<seed>.json.gz`;
- `frames/frames_<seed>.npz`;
- `<seed>.error.json` for a failed scene.

**How the command reads it:**
- **Per policy.** It reads each pass as its own `Run`; policies are never pooled.
- **Checks.** It checks every trace's sha256.
- **Outputs:**
  - `scene_arm_summary.csv`, `event_table.csv` and `opportunity_funnel.csv` per policy;
  - `policy_paired_outcomes.csv`, which pairs the two policies by scene;
  - `analysis_manifest.json`.
- **Failures.**
  - A missing pass, an error record, an incomplete record, a time-order violation or a funnel inconsistency makes the analysis `INCOMPLETE`, with exit code 3.
  - Nothing is dropped.

**Test.** `test_a_c1_format_mock_run_from_the_real_collector_is_read` runs the real `hci.vision_center_c1.collect_scene` on the mock runtime and the driver's `save_scene`:
- both policies, all nine arms;
- synthetic seeds 7 and 8;
- the command reads the result as `MOCK`.

## 2. Computed automatically for C1 (per policy)

| Item | From |
|---|---|
| Triggered, ACT, control steps, actual movement, abstention, immediate stop, errors (with denominators) | records and traces |
| Harm and rescue against C0, harm among moved C0 successes, protected by abstention | `cell.y` of C0 and the arm |
| Oracle rescues, and for **every** vision arm: also rescued / missed by the first block in code order, with flags | C1 runs the oracle in both passes |
| Target and held estimate error with coverage, and the memory vector identity | readings, memory target, trunk ground truth |
| Command vs TCP vs held / target motion, push assistance, the separate limits, offset at control end, at opening and at release | control steps and traces |
| Event table with the fixed time axis | traces |
| Per-policy tables and scene-paired policy comparison | two passes |

**What C1 adds over Phase B:**
- The six ablation arms are **executed** in C1. Their motion and outcome columns fill from the records instead of `NOT_RUN`.
- The ablation chain then reads: removed element → information → decision → motion → outcome → side effect.
- The S6 decomposition (`s6_validity.collect`) runs on C1 traces, since every C1 row carries the spatial fields.

## 3. Remaining data gaps

| Gap | Class | Why |
|---|---|---|
| Contacts in sub-steps 1–4 and between steps; continuous-time safety | interpretation limit | the step loop is not changed (Option A); last sub-step only |
| PhysX separation as a distance (S6 FAIL) | interpretation limit | use the reconstruction from logged poses for distances, and the list for contact existence |
| Held-cube in-gripper shift in the hand frame | interpretation limit | world-frame XY relative to the TCP is used |
| Policy-seed generalisation beyond two seeds | follow-up research | two fixed policies |
| Oracle as an upper bound of recoveries | interpretation limit | the oracle is one controller |
| Cause of a failure after resume | interpretation limit | event order is observed; the cause is a hypothesis |

**No mandatory blocker was found.** Every table above can be computed from the fields the C1 plan already records.

## 4. Not changed by this work

The following are unchanged:
- the collection code, control code, policies and records;
- the statistical plan (two primary tests, Holm procedure, per-policy safety);
- the plan `6b3f04fe…` and every file it hashes.

The analysis lives in `mechanism_analysis/`, which no plan hashes.
