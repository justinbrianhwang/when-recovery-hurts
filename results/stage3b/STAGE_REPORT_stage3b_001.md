# Stage report — Stage 3B: full saved feature instead of 8 principal components

**Neither candidate met the entry criteria, so the next rollout is not authorised and model search for the current configuration is closed for this cycle.** Giving the selector all 568 saved feature dimensions did not produce a gain over never intervening: over the 36 scenes the two candidates ended with 9 and 10 successful scenes against 11 for Never. This is a development result on scenes whose outcomes had already been seen; it does not show that observation-based selection is impossible.

Run ID: `stage3b_001`
Stage: Stage 3B — one representation comparison for the Continue / R5 selector (local CPU, no new rollouts)
Task / policy: StackCube-v1 / saved records of policy `stack_1d_s1_d500_clean_v1_i080000` (no inference, no retraining)
Environment: local PC, Windows 11, Python 3.14.4, numpy 2.5.4, scikit-learn 1.9.0, scipy 1.18.0 (identical to the Stage 3A record). 2026-11-15 01:56–01:57 UTC
Code: repository `2e58810` plus the new files of this stage (hashes fixed in the manifest)
Change record: `amendments/AMENDMENT_010_stage3b_direction_lock_full_feature.md`

## Status

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage3b: COMPLETED_DEV_NO_GATE_PASS
current_configuration_next_rollout: NOT_AUTHORIZED
current_configuration_model_search: CLOSED_FOR_THIS_CYCLE
new_simulator_rollouts: NOT_RUN
base_policy_retraining: NOT_RUN
calibration: NOT_RUN
final_test: NOT_RUN
```

## What was done

- One change against Stage 3A: the model input. The candidates `M_Q_full` and `M_pair_full` read all 568 saved `policy_feature` dimensions, standardised on the training snapshots of each fold, plus the checkpoint one-hot. No PCA.
- Everything else was kept: the 36 leave-one-scene-out folds, rows and weights, the logistic-regression settings, the score > 0 rule, both evaluators, the library versions.
- The Stage 3A models were refitted in the same run as controls. All 384 of their predictions and all their tables matched the stored Stage 3A files exactly (largest difference 0.0).
- Predictions were written and hashed before held-out outcomes were joined. The run was executed once.

## Commands

| Command | Exit code | Log | Start / end (UTC) |
|---|---:|---|---|
| `python -m pytest -q` | 0 (223 passed, 3 skipped) | terminal | 2026-11-15 01:55 |
| `codex exec …` (pre-run code review) | 0 (1 finding, fixed before the manifest) | session record | 2026-11-15 |
| `25_stage3b_full_feature.py manifest --out runs/stage3b_001` | 0 | `logs/s3b_manifest.log` | 01:56:06 / 01:56:34 |
| `25_stage3b_full_feature.py verify --label input_hash_check_before` | 0 | `logs/s3b_verify_before.log` | 01:56:34 / 01:56:34 |
| `25_stage3b_full_feature.py run --out runs/stage3b_001` | 0 | `logs/s3b_run.log` | 01:56:34 / 01:57:37 |
| `25_stage3b_full_feature.py verify --label input_hash_check_after` | 0 | `logs/s3b_verify_after.log` | 01:57:37 / 01:57:38 |

No command failed.

## Outputs

Originals are in `runs/stage3b_001/` (not in Git); copies are in this folder (first 16 characters of SHA256).

| File | Purpose | SHA256 |
|---|---|---|
| `direction_lock.json` | question, fixed items, the one change, forbidden items, entry and exit criteria | `7f05772fb1a0e17c` |
| `analysis_manifest.json` | hashes of 90 inputs, code and configs; library versions | `b6ed2d431b940967` |
| `model_config.json` | settings used and the single item changed against Stage 3A | `1846d304d923afa6` |
| `oof_predictions.csv` | 640 label-free predictions (128 snapshots × 5 models) | `30ba44cc49fb3fc9` |
| `snapshot_decisions.csv` | score and choice per snapshot | `3ec7cb7a8660bc13` |
| `cached_episode_decisions.csv` | decision checkpoint and branch read, per scene and method | `5f81eea6519d00bb` |
| `stage_separated_summary.json` | tables per stage and for the auxiliary pool | `6fb0b8e939960a07` |
| `gate_decision.json` | each entry criterion per candidate with its numbers | `86035518ce938e85` |
| `stage3a_reproduction.json` | controls against the stored Stage 3A files | `99ca49a8d9452e0b` |
| `fold_fit_diagnostics.json` | convergence and class counts per fold | `58a6c6834e706b15` |
| `input_integrity.json`, `input_hash_check_after.json` | recount and commonality; inputs unchanged after the run | `0194b18e5dbbe15c`, `ab410db3a22e6f6b` |
| `scripts/25_stage3b_full_feature.py`, `configs/stage3b_full_feature.json` | analysis code, lock and criteria | `d00e93fd27ec4d41`, `fce1b81576e1fd90` |
| `tests/test_stage3b_full_feature.py` | 19 synthetic checks | `110aa6011f162533` |

## Checks

| Item | PASS / FAIL / NOT RUN | Evidence |
|---|---|---|
| CPU tests | PASS | 223 passed, 3 skipped. The Stage 3B tests use synthetic records only |
| CUDA, Vulkan/RGB, policy rollout, restore, videos | NOT RUN | this stage uses no GPU, no simulator and no base-policy inference |
| Scene split | PASS | 36 folds, each holding out every checkpoint, repeat and label of one scene; identical to Stage 3A |
| Input integrity | PASS | recount 24 / 84 / 336 and 12 / 44 / 176; saved inputs match the snapshot hashes and the backup lists; 12 commonality checks pass |
| Feature | PASS | exactly the 568 saved dimensions; other dimensions and other columns are refused (tests) |
| Leakage | PASS | scaler, fit and timer choice use training scenes only; predictions hashed before labels were joined; tests change held-out labels and features without any effect |
| Stage 3A controls reproduced | PASS | 384 predictions, 0 differing choices, largest difference 0.0, no differing table |
| Convergence | PASS | 36 folds × 8 heads converged at `max_iter=2000`; no retry, no single-class head |
| Completion | PASS | `run_status.json` completed; 90 input files unchanged before and after |
| Scientific hypothesis | NOT RUN | this stage is not a hypothesis test |

## Results

Source episodes: 36 stored scenes (24 + 12). New rollouts: 0. Stored blocks read: 512. Time: 63 s on a CPU.

### Main reading: one intervention read from the stored repeat-0 paths (successful scenes)

| | Never | Timer chosen on training scenes | `M_time` | `M_Q` (3A) | `M_pair` (3A) | `M_Q_full` | `M_pair_full` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 2B (24 scenes) | 8 | 8 | 8 | 8 | 8 | 6 | 7 |
| Stage 2D (12 scenes) | 3 | 3 | 3 | 2 | 3 | 3 | 3 |
| Both (auxiliary) | 11 | 11 | 11 | 10 | 11 | 9 | 10 |

| Candidate | Scenes gained | Scenes lost | Scenes with an intervention |
|---|---|---|---:|
| `M_Q_full` | 10054 | 10027, 10032, 10051 | 18 of 24, 8 of 12 |
| `M_pair_full` | 10043, 10054 | 10027, 10046, 10051 | 17 of 24, 9 of 12 |

### Entry criteria

| Criterion | `M_Q_full` | `M_pair_full` |
|---|---|---|
| 1. Not below Never and the timer in each stage | no (6 < 8 in 2B) | no (7 < 8 in 2B) |
| 2. At least 2 more scenes than Never and than the timer over both stages | no (9 vs 11) | no (10 vs 11) |
| 3. At most 1 baseline success lost | no (3 lost) | no (3 lost) |
| 4. Net gain of at least 1 scene in each stage | no (−2, 0) | no (−1, 0) |
| 5. All folds and checks complete | yes | yes |
| **Result** | **not passed** | **not passed** |

### Snapshot-level reading (the same choice applied to the four repeats)

| | R5 chosen (snapshots) | Successful blocks vs Continue | Scene-mean difference | 0→1 (scenes) | 1→0 (scenes) |
|---|---:|---:|---:|---:|---:|
| 2B `M_Q_full` | 37 / 84 | 82 vs 93 | −0.055 | 2 (1) | 13 (3) |
| 2B `M_pair_full` | 37 / 84 | 89 vs 93 | −0.031 | 4 (2) | 8 (2) |
| 2D `M_Q_full` | 12 / 44 | 30 vs 25 | +0.019 | 9 (3) | 4 (1) |
| 2D `M_pair_full` | 16 / 44 | 27 vs 25 | −0.002 | 9 (3) | 7 (1) |

- On the Stage 2D scenes `M_Q_full` was above Continue at the snapshot level (30 vs 25 blocks), but this did not carry over to the main reading (3 scenes, equal to Never) and was accompanied by a loss on the Stage 2B scenes.
- Of the 17 snapshots where R5 did better than Continue, `M_Q_full` chose R5 at 6 and `M_pair_full` at 7 (Stage 3A: 2). Of the 32 snapshots where Continue did better, they chose R5 at 6 and 4 (Stage 3A: 4 and 5).
- Brier scores of the predicted success probabilities were worse than in Stage 3A on the Stage 2B scenes (Continue 0.23 and 0.21 against 0.13) and similar or worse on the Stage 2D scenes. Across folds, 170–172 of the 568 dimensions had zero variance in the training snapshots.

## What this does and does not say

- With these records and these fixed linear models, using the full saved feature did not give a gain over not intervening, and it lost more baseline successes than the 8-component version.
- It does not show that the PCA was or was not the reason for the Stage 3A result: the change also alters the geometry of the L2 penalty, and no other setting was tried.
- It does not show that observation-based selection is impossible, and it says nothing about new scenes or closed-loop behaviour: the table reads stored paths at four fixed checkpoints, and every fold has a different model.
- No p-value, interval, calibration or harm-rate bound was computed. The independent units are 24 + 12 scenes.

## Changes and failures

- New: `scripts/25_stage3b_full_feature.py`, `configs/stage3b_full_feature.json`, `tests/test_stage3b_full_feature.py`, `amendments/AMENDMENT_010_stage3b_direction_lock_full_feature.md`, `results/stage3b/`.
- Modified: `README.md` (Stage 3B entry), `.gitattributes`.
- `scripts/24_stage3a_selector.py` was imported unchanged (same hash as in the Stage 3A manifest). No earlier result file was written to.
- Codex pre-run review found one defect in the reproduction check (it compared only two fields of each stored prediction). It was fixed before the manifest: every prediction field and the probability tables are compared.
- Failed commands: none.
- Not done: GPU rental, simulator rollouts, base-policy inference or retraining, any search over C, dimensions, thresholds or seeds, stage-transfer fits, other features or models, calibration, final Test.
- Cost: $0.

## Next step

By the criteria fixed before the run, no candidate passed. The current configuration (StackCube, D500-clean 80k, Continue / R5) is closed for further model, distance, duration or training-amount search in this cycle, and no new rollout is authorised. Nothing further was started. What comes next is a decision for the user.
