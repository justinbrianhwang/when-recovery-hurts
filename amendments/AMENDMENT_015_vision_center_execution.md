# Amendment 015 (execution) — Vision-CENTER: stopped by the budget gate before stage B1

Date: 2026-11-17 (UTC). Work id `vision_center_001`. Plan SHA256
`72b750057e31da493983f123a3cafd19cceed979d443f3daacfe76595da47e75`. Approved cap US$3.00 (user approval naming the cap and the
plan hash; recorded in `authorization.json`).

## Status

**BUDGET_BLOCKED. Stage B1 (200 used scenes) and stage B2 (100 new scenes) were not started.** No rescue, harm or gate figure
exists. The new scenes 50800–50899 have no recorded use (no ledger); only their initial cube placements were reset in the
placement check, without the policy. No constant, rule, estimator or gate was changed, and nothing was re-run.

## What ran (one rented RTX 4090, 56 minutes, deleted after the hash-checked backup)

| Step | Exit | Result |
|---|---:|---|
| Local: plan, code, policy and seed hashes against the plan | 0 | no file missing, none changed |
| Server: install | 0 | 25 minutes (slow package download on this host) |
| Server: environment check, test suite | 0 | 321 passed, 5 skipped |
| Server: files against the plan | 0 | none missing, none changed |
| Server: placement check of the 100 new seeds | 0 | no placement identical to any of 1,747 on record |
| Server: technical audit, 3 used scenes, three arms | 0 | pass |
| Server: budget gate for stage B1 | 2 | projection US$7.89 against the cap US$3.00: refused |
| Stage B1, stage B2 | — | not started |

## Audit on the real simulator (3 already-used scenes; outcomes decide nothing)

- All three reached the trigger. Vision-CENTER abstained in two (`ALIGNED`): actions and final state identical to the
  uninterrupted run. It acted in one (50402): 3 control steps, ended `within_tolerance`, release offset 5.5 mm (oracle arm on
  the same scene: 5.6 mm).
- Offset error at the trigger against logged ground truth: 0.6, 0.5 and 2.8 mm. Target position error: 0.1–0.4 mm.
- No cube position was read outside `CENTER_ORACLE`.
- This is the first time the simulator-side code ran. Three scenes are a technical check, not a performance estimate.

## Why the gate refused

The gate projects every scene at its bound (13 readings per scene, all steps) with a 1.25 margin. Its inputs on this host:
0.036 s per environment step (oracle pilot host: 0.028) and **9.3 s per vision reading** (plan assumption: 1.8 s, measured on
the development machine only). US$0.45 was already spent when the gate ran (plan assumption: US$0.20).

After the gate, the unchanged perception function was timed on the three stored fixture frames on the same server: up to 24 s
per reading with the default thread count, 0.2–2.6 s with the thread count limited to 1 or 8. The container showed 256 CPUs on
a loaded shared host. The slow reading is an effect of the execution environment, not of the estimator. The gate was not
re-run with a thread limit: a refusal stops the stage, and a second attempt is a spending decision for the user.

## Cost

US$0.44 (Vast credit 10.6034 → 10.1655), of the US$3.00 cap. No instance is left.

## Deviations and judgment calls

- A timing diagnostic that was not in the plan was run for about one minute on the server before deletion (stored fixture
  frames only; no simulator, no scene).
- The plan's cost estimate (US$0.90–1.40, worst case US$2.51) was wrong for this host because the reading time had never been
  measured on a server.

## Records

`results/vision_center_001/run1_budget_blocked/`: authorization, placement check, audit (records, traces, frames), budget
gate, server logs and status, per-file hashes of the server archive, timing diagnostic, cost.
