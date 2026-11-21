# C1 analysis readiness

Reference plan `6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7`. No C1 run exists. **This is not an approval of a paid run.**

## 1. How a C1 run is read

**Command.** This is one command and it produces the whole chain (targeted fix v1.0, review F3):
```
python mechanism_analysis/run_analysis.py c1 --run <C1 run folder> --out <new folder> [--plan results/vision_center_v2_c1_001/preparation_v3/plan.json]
python mechanism_analysis/run_analysis.py c1 --run <mock run> --out <new folder> --source-kind MOCK --expected <MOCK_EXPECTED_MANIFEST.json>
```

**Expected-set gate.** Before anything is analysed, the run is compared with an expected set:
- `C1_ORIGINAL`: the plan file and the two registries it hashes;
- `MOCK`: a separate small manifest.

Details are in `ANALYSIS_SPEC.md` §3a. An empty, partial, duplicated, extra, wrong-plan, wrong-arm, error or changed input stops the command. It then writes:
- `INCOMPLETE` and exit code 3;
- `gate_report.json`;
- `INCOMPLETE_NOT_A_RESULT.txt`;
- no report and no claims.

**Required outputs** (`run_analysis.C1_REQUIRED`): the status is `ANALYSIS_COMPLETE` only when all of these exist:
- per policy: `scene_arm_summary.csv`, `event_table.csv` and `opportunity_funnel.csv` (overall, trigger-conditional and move-conditional);
- `tables/denominator_check.csv`;
- `tables/crosscheck_against_collector_summary.csv`, an independent recount against the collector's `result.json`; any mismatch blocks the result;
- `tables/target_estimate_and_memory_decomposition.csv` and `tables/held_estimate_decomposition.csv`;
- `tables/executed_ablation_chain.csv`: information → decision → executed motion → outcome, paired with v2 by scene;
- `tables/s6_validity_summary.json` and `tables/s6_pair_rows.csv.gz`, the S6 decomposition on the C1 spatial fields. A row without spatial fields blocks the result. Listed pairs and recorded nonzero impulses are counted apart;
- `case_index.csv`: rule-selected cases with their record, trace and frame paths and SHA256;
- `results.json`, `claims_to_evidence.csv`, `C1_MECHANISM_REPORT.md`, `START_HERE.md`, `policy_paired_outcomes.csv`, `metric_dictionary.json`;
- `analysis_manifest.json`, with the hashes of the code, settings, expected set, inputs and outputs.

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

**Tests** (`mechanism_analysis/tests/test_targeted_fix.py`):
- **The whole chain.** `test_f3_the_c1_command_produces_the_whole_chain` runs the real `hci.vision_center_c1.collect_scene` on the mock runtime, with:
  - the spatial mock logger;
  - the driver's `save_scene`;
  - the collector's own `summarize` for `result.json`;
  - both policies, all nine arms, and synthetic seeds 7 and 8.

  The command gives `ANALYSIS_COMPLETE`, and every required output exists.
- **Damaged inputs.** A disagreeing or missing collector summary, missing spatial fields and a missing arm are each blocked.
- **Synthetic counterexamples.** 13 counterexamples, plus the empty-folder reproduction and the MOCK-manifest and plan cases, are each blocked with their own code.

## 2. Computed automatically for C1 (per policy)

| Item | From |
|---|---|
| Triggered, ACT, control steps, actual movement, abstention, immediate stop, errors (with denominators) | records and traces |
| Overall success, rescue / all C0 failures, harm / all C0 successes, net; trigger-conditional and move-conditional values in separate, labelled steps | `cell.y` of C0 and the arm |
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
