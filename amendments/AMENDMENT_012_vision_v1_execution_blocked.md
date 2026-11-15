# Amendment 012 (execution) — Vision-V1 stopped by the budget gate

Date: 2026-11-15. Work id `vision_v1_001`. Plan SHA256 `5fdee368fa529c0096464332e1af5dc2ac0eca39ff102ad7bc0fa8605ed3678e`.
The preparation is described in `AMENDMENT_012_vision_v1_preparation_DRAFT.md`, which is unchanged.

## Approval

The user approved Phase B once with an all-in cap of US$5.00 for this plan hash, and forbade changing seeds, batch size,
update count or scene counts to fit the budget.

## What happened

- Server checks passed: tests (217 passed, 2 skipped), data and pretrained-file hashes, data gate, expert replay (14 of 14),
  placement check of the 200 new evaluation seeds (0 identical placements).
- Throughput probe (300 updates, batch 256, RTX 4090): 6.20 updates/s, peak VRAM 18.3 GB.
- Budget gate: projected all-in total US$6.44 (US$5.38 without margin and reserve) against the US$5.00 cap. Blocked.
- Training, export, selection and qualification were not run. `vision_v1_select` and `vision_v1_qualify` are unused; no
  qualification ledger exists.
- Spent US$0.30 (credit difference), including a first host abandoned for a slow upload. Both instances are destroyed.

## Status

`vision_v1: BUDGET_BLOCKED`. No claim about the ResNet-18 policy can be made. Nothing was changed in the plan, the code,
the legacy policy or earlier results. A second attempt needs a new approval with a higher cap (about US$7.00 for the same
design); the authorisation of this run is not reused.

Records: `results/vision_v1_001/` (budget gate, placement check, data gate, replay, probe memory log, server status, cost).
