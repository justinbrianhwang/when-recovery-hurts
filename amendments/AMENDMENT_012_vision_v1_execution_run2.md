# Amendment 012 (execution, run 2) — Vision-V1 trained and qualified once

Dates: 2026-11-15 to 2026-11-16 (UTC). Work id `vision_v1_001`, second execution.
Plan SHA256 `5fdee368fa529c0096464332e1af5dc2ac0eca39ff102ad7bc0fa8605ed3678e`, unchanged.
Earlier records, unchanged: `AMENDMENT_012_vision_v1_preparation_DRAFT.md`, `AMENDMENT_012_vision_v1_execution_blocked.md`.

## Approval

The user approved a re-run of the same design with an all-in cap of US$8.00. A new authorisation file
(`authorization_run2_cap8.json`) was written for the same plan hash; the first one (cap US$5.00) is kept and was not reused.

## What was run

- Two trainings of the Diffusion Policy with the ImageNet-pretrained ResNet-18 encoder on the same 500 demonstrations:
  train seeds 1 and 2, 100,000 updates, batch 256. Both completed (about 5 h each on an RTX 4090, 18.3 GB VRAM).
- The fixed 80k EMA candidate of each run on `vision_v1_select` (100 scenes): seed 1 85/100, seed 2 83/100. Seed 1 selected by the
  predeclared rule.
- One paired qualification on `vision_v1_qualify` (100 scenes, ledger written before any rollout): new policy 79/100, frozen
  legacy policy 28/100; both 25, new only 54, legacy only 3, neither 18. Label `TARGET_MET`.

## Deviation from the scripted chain

The first start of the selection evaluation stopped before any episode because four lists of scene seeds used in earlier stages
(`results/stage2a|stage2b|stage2d|handoff_h1_001/scenes.json`) had not been uploaded; the preparation's upload list had missed
them. The unchanged committed files were uploaded, hash-checked, and the same step was started again. No code, plan, policy, seed
or setting was changed; the failed log is kept. The plan's list of files needed on the server should include these files.

## Status and limits

`vision_v1: EXECUTED_ONCE, TARGET_MET`. `vision_v1_select` is used and `vision_v1_qualify` is consumed.
This is a development qualification on one task. It does not separate pretraining from encoder size, it is not the final Test,
and the labels are not significance tests. The legacy policy, Stage 3B and H1 results are unchanged. No recovery experiment,
calibration or final Test was run.

Cost: US$6.19 for both executions (US$0.32 + US$5.88), credit difference. The instance is destroyed.
New frozen policies (local, not in Git): `artifacts/policies/stack_v1_s1_d500_clean_resnet18_i080000` (weights `a37be25603d6c7b7…`,
selected) and `…_s2_…` (weights `52eff6a94a075bb3…`). Records: `results/vision_v1_001/run2/`.
