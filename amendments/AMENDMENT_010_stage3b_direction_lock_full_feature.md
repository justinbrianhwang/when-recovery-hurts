# Protocol change record 010 — research direction lock and Stage 3B (one representation comparison, CPU only)

Previous version: v0.3 + AMENDMENT_001–009 (repository HEAD `2e58810`)
New version: v0.3 + AMENDMENT_001–010
Time of change: 2026-11-15 (UTC), before any preprocessing, fit or prediction of this stage.
Reason: Stage 3A found no gain for a selector that reads 8 principal components of the saved policy feature. The user's instruction `prompts/RESEARCH_DIRECTION_LOCK_AND_STAGE3B_v1_0.md` fixes the research direction for this cycle and allows one more development comparison in which only the input representation changes.

This plan was made **after the Stage 2B, 2C, 2D and 3A results had been seen**. The criteria below are operating criteria for deciding whether to spend budget on a next step. They were not pre-registered before the data existed, and they are not significance, generalisation or harm-rate criteria.

## 1. Direction lock

The research question is unchanged:

> With a fixed manipulation policy, can the decision to intervene be made from run-time observations alone so that task success increases while degradation of baseline-successful runs stays limited?

For this cycle the configuration is fixed to StackCube-v1, the D500-clean 80k policy, and the two choices Continue and the existing 5 cm retreat (R5).

| Item | Fixed |
|---|---|
| Task and physics | pinned StackCube-v1 / ManiSkill, existing runtime semantics |
| Base policy | D500-clean 80k EMA, existing weight and metadata hashes; horizons 2 / 8 / 16, 100 DDPM steps |
| Episode | 200 steps, `success_once_absorbing` |
| Choices | Continue and the existing R5 only; controller, held gripper command, arrival tolerance and step limit unchanged |
| Interventions | at most one per episode, decided at steps 12 / 52 / 92 / 132 |
| Data | the common C0 / R5 records of Stage 2B and Stage 2D only |
| Role of the evaluation | post hoc development evaluation; not an independent validation, not a final Test |
| Protected | no reuse of qualification sets, no access to calibration or Test |

Not done in this stage: GPU rental, simulator rollouts, base-policy inference or retraining, new data, new recovery actions (R2M, H0M, Refresh), another base policy or task, stage-transfer fits, new ablations. Earlier results, manifests, plans and models are not overwritten, and the Stage 3A result is not replaced.

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage3b: DEV_REPRESENTATION_COMPARISON
new_simulator_rollouts: NOT_RUN
base_policy_retraining: NOT_RUN
calibration: NOT_RUN
final_test: NOT_RUN
```

## 2. The one change: model input

| | Stage 3A (controls, refitted unchanged) | Stage 3B candidates |
|---|---|---|
| Input | StandardScaler → PCA 8 → StandardScaler, plus checkpoint one-hot (12 columns) | StandardScaler on all 568 saved `policy_feature` dimensions, plus checkpoint one-hot (572 columns). No PCA |
| Models | `M_time`, `M_Q`, `M_pair` | `M_Q_full` (score `q_R − q_C`), `M_pair_full` (score `p(Rescue) − p(Harm)`) |

- The scaler is fitted on the unique training snapshots of each fold only; the held-out scene is transformed, never fitted on.
- The feature dimension is checked against the saved schema (568). Another dimension is refused; nothing is cut or padded.
- No new RGB encoder, no fine-tuning, no hand-made features from `state_history` or `queue`, no privileged object state. Only what the saved `policy_feature` already contains.
- Everything else is that of Stage 3A: the same 36 leave-one-scene-out folds, the same rows and row weights (`1 / (4 · m_j)`, mean 1 in the training fold), `LogisticRegression(solver=lbfgs, C=1.0, fit_intercept=True, class_weight=None, tol=1e-7, max_iter=2000)` with one retry at `max_iter=10000` on a convergence warning, the same single-class rules, and the same library versions as recorded by Stage 3A.
- Decision: R5 if score > 0, otherwise Continue; a score of exactly 0 is Continue. Zero is the net-gain boundary, not a harm tolerance. No risk constraint is claimed.
- No search over C, dimensions, thresholds, penalties or seeds.
- Standardising 568 dimensions instead of 8 components also changes the geometry of the L2 penalty. If results improve, that is not evidence that the PCA caused the Stage 3A result.

## 3. Evaluation

- Predictions are written and hashed before held-out outcomes are joined. Scene id, host, stage, file names, outcomes and simulator ground truth are never model inputs.
- Snapshot-level reading: one decision per snapshot, applied to its four repeats.
- Main reading: the cached single intervention on repeat 0. Checkpoints are read in the order 12 → 52 → 92 → 132; at the first one where the model chooses R5 the stored R5 branch is read and the episode ends; otherwise the stored trunk is read. Later scores are never used to go back, and branches of different repeats are never joined into one episode.
- The Stage 2B and Stage 2D tables are reported separately; their sum is an auxiliary figure for the development decision.
- The Stage 3A models are refitted in the same run and must reproduce the stored Stage 3A predictions and tables. A difference stops the run before any candidate is read.

## 4. Operating entry criteria (fixed before the run)

For each candidate, on the cached repeat-0 single-intervention reading:

1. In Stage 2B and in Stage 2D separately: successful scenes not fewer than Never and not fewer than the training-selected timer.
2. Over both stages: at least 2 more successful scenes than Never, and at least 2 more than the training-selected timer.
3. Scenes turned from baseline success into failure: at most 1 over both stages. This limits a count in this sample; it is not a population rate.
4. A net gain over Never of at least 1 scene in Stage 2B and in Stage 2D (the gain must not come from one stage alone).
5. All folds and integrity checks complete, no scene missing, every fit converged.

If both candidates pass: more successful scenes, then fewer scenes lost, then fewer interventions, then `M_Q_full`. The criteria are not relaxed after the results are seen. A gain of one scene, or a gain in one stage only, is reported as it is and does not pass.

If no candidate passes:

```text
stage3b: COMPLETED_DEV_NO_GATE_PASS
current_configuration_next_rollout: NOT_AUTHORIZED
current_configuration_model_search: CLOSED_FOR_THIS_CYCLE
calibration: NOT_RUN
final_test: NOT_RUN
```

No further model, distance, duration or training-amount search is added in this cycle. This closes the development of the current configuration; it is not a statement that observation-based selection, or the research question, is impossible.

If a candidate passes, the only next step is a real run of that one fixed candidate against Never and a fixed timer on 24 new development scenes (at most 72 full episodes), after a sample and cost report and a separate approval. Nothing is rented automatically, and passing is not an approval of calibration or of the final Test.

## 5. Code

| File | Content |
|---|---|
| `scripts/25_stage3b_full_feature.py` (new) | direction lock, manifest, full-feature fits, Stage 3A reproduction check, evaluation, entry-criteria decision |
| `configs/stage3b_full_feature.json` (new) | the lock and the criteria in machine-readable form |
| `tests/test_stage3b_full_feature.py` (new) | synthetic checks |

`scripts/24_stage3a_selector.py` is imported unchanged (its hash must equal the one in the Stage 3A manifest). Runtime modules are not imported.

## 6. Wording corrections to earlier summaries (interpretation only; no number or verdict is changed)

- "The base policy is the bottleneck" is an interpretation. No experiment separated base-policy quality as the cause of the intervention or selection results. The 50/100 gate is an operating gate, not a boundary at which intervention research becomes possible.
- "Refresh was indistinguishable from the noise-only arm" does not mean an equivalence test was run. In that sample no clear additional gain of Refresh was observed.
- Stage 3A is the result of a linear score on 8 principal components with a net-gain rule. It is not a failure of a risk-constrained selector and not a statement about all observation models.
- The restore and reproducibility results are passes for the policy, environment, host and states that were checked. They are not a proof of global determinism.
- Stages 0–3A are exploratory examinations. What has not been done is confirmatory testing, risk calibration and the final Test.
- The qualification result `BELOW_ORIGINAL_GATE` stays as a historical record. It is not changed to a pass, and the formal gate is not changed automatically.
