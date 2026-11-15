# Protocol change record 011 (execution) — Handoff H1, single approved run

This record puts the draft `AMENDMENT_011_handoff_h1_preparation_DRAFT.md` into effect for **one** run. The draft is kept unchanged; its sections 2–5 and 8 (question, fixed conditions, arms, checks, code) apply as written.

Previous version: v0.3 + AMENDMENT_001–010 (repository HEAD `883a72a`)
Date: 2026-11-15 (UTC), before any rollout of this run and before a GPU was rented.

## Approval

- Given by the user in the working session on 2026-11-15 with the instruction to read and execute `AGENT_PROMPT_HANDOFF_H1_EXECUTE_v1_1.md`: the single H1 experiment, total cap **US$1.00** including install, transfer, storage, GPU run and backup.
- Not approved: account top-up, a higher budget, further conditions or runs, training, qualification, calibration, final Test.
- The approval is recorded in `runs/handoff_h1_001/authorization.json` and tied to the plan by hash.

| Item | Value |
|---|---|
| Run id | `handoff_h1_001` (new folder; the preparation folder `handoff_h1_prep_001` is not reused for outputs) |
| Plan | `runs/handoff_h1_001/scope_plan.json`, SHA256 `b381e24fa49023450a3adc43d9587cf5f38a7de578595aa532a9addea4d00cd9` |
| Scenes | 10060–10071 (12 train seeds with no recorded use), subject to the placement check on the server |
| Upper bound | 576 continuations, 12 trunks, 48 R5 control-only reference runs, 16 reruns, audit on seeds 100–104 |
| Scale rule | first two scenes as a technical check, then 12 → 6 → 2 from measured time and cost only |

## Changes since the preparation

- `scripts/26_handoff_h1.py`: `scripts/23_stage2d_pilot.py` was added to the list of hashed code files, because the `scenes` command loads it for the cube-placement check. No behaviour changed. All other new files have the hashes of the preparation manifest; the plan fixes the hashes actually used.
- A mock end-to-end pass of `techcheck`, `scale` and `finalize` was run locally before renting (not a result).

## Unchanged

Base policy and its hashes, R5, the 200-step budget, the noise rule, checkpoints, repeats, restore method, audit tolerances. Stage 3B stays closed (`COMPLETED_DEV_NO_GATE_PASS`, `CLOSED_FOR_THIS_CYCLE`). The qualification record stays `BELOW_ORIGINAL_GATE`.

## Stopping

One run and one report. A technical fault or a projected cost above the cap ends the run with a record and a backup. Whatever the outcome, no other history length, repeated padding, waiting step, model, policy or additional scenes are added. A result is a difference (or no difference) under a controlled manipulation of the resume input in this sample; it is not evidence of an out-of-distribution cause, and no novelty is claimed.
