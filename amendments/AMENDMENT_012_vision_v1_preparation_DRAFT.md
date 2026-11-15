# Protocol change record 012 (DRAFT) — Vision-V1: ImageNet-pretrained ResNet-18 visual encoder for the base policy

**Status: draft. Preparation only. No training, no policy inference, no simulator evaluation and no paid compute has been run or authorised under this record.** The proposed cap of US$3.00 is a proposal. The H1 approval is not inherited.

Previous version: v0.3 + AMENDMENT_001–011 (repository HEAD `3842f7d`)
Date of draft: 2026-11-15 (UTC).
Reason: the user's instruction `prompts/AGENT_PROMPT_VISION_V1_RESNET18_POLICY_UPGRADE_v1_0.md` opens a new base-policy development cycle. Stages 2–3 and H1 found no intervention or selector that improves on doing nothing, on a base policy that solves 37 of 100 held-out scenes. The pinned baseline's visual encoder is four unpretrained convolution layers with global max pooling; an unreplicated community report (ManiSkill issue #1325) describes better StackCube results with a pretrained ResNet-18.

## 1. Scope

This is a **new base-policy development cycle**. It does not reopen the Stage 3B selector search or H1. It was designed after all earlier results had been seen.

About `AGENTS.md`: it fixes the research plan at v0.3 and forbids changing the research direction without a request from the user. This cycle is such a request, recorded here. `AGENTS.md` also forbids renting GPUs without an explicit request; nothing was rented.

Not part of this work: recovery selector, R5, oracle correction, any new control primitive, LLMs, calibration, the final Test, a pretrained-versus-scratch control, more seeds or more demonstrations.

```text
legacy_base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100), historical, unchanged
legacy_stage3b: COMPLETED_DEV_NO_GATE_PASS
handoff_h1: EXECUTED_ONCE
vision_v1: PREPARATION_ONLY
new_training: NOT_RUN
new_policy_inference: NOT_RUN
paid_compute: NOT_AUTHORIZED
calibration: NOT_RUN
final_test: NOT_RUN
```

## 2. Question

Does replacing the weak visual encoder with an ImageNet-pretrained ResNet-18-based encoder improve task success for a Diffusion Policy trained on exactly the same validated 500 StackCube demonstrations?

Caveat fixed in advance: against the legacy encoder this changes architecture, capacity and pretraining together. If success improves, the study cannot attribute it to pretraining alone.

## 3. The one model change: the encoder

Legacy (pinned upstream `PlainConv`, confirmed in the source): four 3×3 convolutions 16 → 32 → 64 → 128 with max pooling, a 1×1 convolution, global max pooling, `Linear(128, 256)` + ReLU. Input: 6 channels (two RGB cameras concatenated), 128×128. Not pretrained. 147,408 parameters.

Vision-V1 (`hci.vision_v1.ResNet18TwoCamEncoder`, architecture id `dp_resnet18_imagenet1k_v1_shared_twocam_fc256_v1`):

| Item | Fixed value |
|---|---|
| Input | the same tensor the upstream agent passes to its encoder: float `[N, 6, 128, 128]` in `[0, 1]`; channels 0–2 base camera, 3–5 hand camera (sorted camera names) |
| Per camera | ImageNet mean / std normalisation; native 128×128. No resize, no crop, no augmentation |
| Backbone | one torchvision ResNet-18, shared by both cameras, classification layer removed, global average pooling: 512 features per camera |
| Initialisation | `ResNet18_Weights.IMAGENET1K_V1`, file `resnet18-f37072fd.pth`, SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` (official `download.pytorch.org` URL; torchvision BSD-3-Clause) |
| Head | concatenate the two cameras (1024) → `Linear(1024, 256)` → ReLU |
| Output | 256, the width the unchanged action U-Net expects (global condition 2 × (256 + 25) = 562) |
| Trained | all convolutions, all BatchNorm affine parameters and the head, jointly with the policy |
| Parameters | encoder 11,438,912 (legacy 147,408); action U-Net 6,103,300 in both; total 17,542,212 (legacy 6,250,708) |

Unchanged: task and environment, the 500 demonstrations, the observation whitelist and camera order, the proprioceptive state, the targets, the action U-Net (64/128/256, group norm 8), the DDPM scheduler with 100 steps, horizons 2/8/16, the loss, the upstream optimizer and learning-rate schedule, EMA, batch 256, 100,000 updates, the success definition.

**One recorded deviation from "fine-tune everything".** The BatchNorm layers of the backbone always use their ImageNet running statistics; the statistics are never updated. Reason: the upstream trainer's EMA averages parameters only. Running statistics updated during training would stay in the non-EMA model and never reach the exported EMA weights, so the deployed policy would not be the trained one. With fixed statistics the training-mode and evaluation-mode outputs are identical, and the exported file alone reproduces the policy. The BatchNorm weight and bias are trained.

How it is wired, without editing the upstream checkout: the upstream module looks up the name `PlainConv` when it builds its agent. For a Vision-V1 run that name is replaced in memory by a factory with the same signature, in the trainer process and, at deployment, only while a Vision-V1 agent is constructed. A policy whose metadata has no `architecture_id` is loaded through the unchanged legacy class. An unknown id, a wrong pretrained file, a different input size or a missing camera is an error, not a fallback.

## 4. Training (two runs, fixed)

- Data: the same file bytes `train_d500_clean_v1.h5` (SHA256 `93416a4a…`), same 500 trajectories in the same order, same manifest and data gate as Stage 1D. No re-conversion, addition, removal or augmentation.
- Seeds **1 and 2**. Exactly two runs. No further seed, no warm start, no restart with other settings.
- The legacy command line plus the encoder: 100,000 updates, batch 256, `eval-freq` 10,000, 5 unseeded upstream monitoring episodes (descriptive log only).
- **Candidate: the EMA weights of tag 80,000 for both seeds** (the same tag as the legacy 80k policy; tag *i* means *i* + 1 optimizer updates). The 100k schedule is completed and the 100k checkpoint archived; it cannot replace 80k after outcomes are seen.
- A run that fails technically is kept as `INCOMPLETE`. No retry with another seed, learning rate, budget or fewer updates.
- A throughput probe (full data, batch 256, 300 updates, run name `vision_v1_probe`, never exported) comes first. If the projected all-in cost of both trainings and all evaluation passes exceeds the approved cap, training does not start (`BUDGET_BLOCKED`). Seeds, scenes and architecture are not reduced to fit.

## 5. New evaluation seeds (frozen in `configs/vision_v1_splits.json`)

| Set | Seeds | Use |
|---|---|---|
| `vision_v1_select` | 50400–50499 | both fixed 80k candidates, 100 scenes each |
| `vision_v1_qualify` | 50500–50599 | **once**: the selected new policy and the frozen legacy policy on the same 100 scenes |

- Seed-level check done: no overlap with `splits.json` (audit, train, dev, calibration, test), `policy_eval_splits.json` (monitor, select, qualify A, qualify B), the 550 demonstration scene seeds, or the scenes of Stage 2A / 2B / 2D / H1.
- Placement check at the server, before any training: cube placements of the 200 seeds against stored placements (demonstrations, pilots) and, by reset, against the audit, dev and policy-evaluation sets. Calibration and Test are compared by seed number only and never reset. An identical placement fails the step. No seed is replaced silently; a revised manifest needs approval.
- `policy_qualify_B` (the legacy 37/100) is not reused and is not a contemporaneous baseline.

Selection on `vision_v1_select`: the training seed with more `success_once` wins; a tie goes to seed 1. If both candidates are below 50/100, the work stops as `NO_VIABLE_DEV_CANDIDATE` and the qualification set stays unused.

Qualification: one paired session by a dedicated driver. A ledger file is created exclusively **before any rollout**; a second session is refused; an interrupted session is not resumed. Both policies use the existing noise convention `seed_for(scene, step, 0, 'dp')`; the same scene seed gives the same initial state. No intervention, 200 steps, `success_once_absorbing`.

## 6. Outcome labels (fixed before any outcome; not a significance test)

| Label | Condition on `vision_v1_qualify` |
|---|---|
| `TARGET_MET` | selected new policy ≥ 60/100 **and** at least 10 more successful scenes than the legacy policy on the same 100 scenes |
| `PROMISING_BELOW_TARGET` | ≥ 50/100, but `TARGET_MET` not reached |
| `BELOW_GATE` | < 50/100 |
| others | `NO_VIABLE_DEV_CANDIDATE`, `INCOMPLETE`, `BUDGET_BLOCKED`, `TECHNICAL_BLOCKED` |

Even with `TARGET_MET`, no recovery experiment follows automatically.

## 7. Authorisation

Every execution command (`scenes`, `budget`, `eval-select`, `choose`, `qualify-paired`, and every phase of `scripts/vision_v1/run.sh`) refuses to run unless `runs/vision_v1_001/authorization.json` exists with `approved_by: "user"`, `paid_compute_authorized: true`, a positive `cap_usd` and the SHA256 of the plan. No code in the repository writes that file.

## 8. Code

| File | Content |
|---|---|
| `src/hci/vision_v1.py` (new) | encoder, in-memory swap, deployment adapter with strict loading, architecture dispatch, state-dict checks, selection and label rules |
| `scripts/27_vision_v1_train.py` (new) | the legacy training launcher plus the encoder |
| `scripts/28_vision_v1_export.py` (new) | export of the declared EMA checkpoint with architecture id and encoder specification |
| `scripts/29_vision_v1.py` (new) | plan, dry run, placement check, budget gate, selection evaluation, single-use paired qualification |
| `scripts/vision_v1/run.sh` (new) | server driver |
| `configs/vision_v1_splits.json` (new) | the two new seed sets |
| `tests/test_vision_v1.py` (new) | 16 CPU tests |

Not modified: `policies.py`, `experiment.py`, `observations.py`, `training.py`, `scripts/03`, `04`, `09`, the upstream checkout, any earlier weights, results, splits or amendments. The plan records that the legacy-path files have the hashes of the H1 run.

## 9. What cannot be concluded from this work, whatever the result

- That pretraining alone is responsible for a difference.
- That 60/100 is statistically established, or that the new policy generalises beyond these scene sets.
- Anything about recovery actions, selectors, calibration or the final Test.
