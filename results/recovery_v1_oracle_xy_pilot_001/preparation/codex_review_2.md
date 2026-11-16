The three reported fixes are present. `bounded_xy_move` stops before sending a command whose target exceeds the 3 cm cap, and `validate_scene` checks every recorded CENTER_ORACLE command. `cmd_run` recalculates the cost using its own price and spend before writing the seed ledger. Both `bound()` and `dry-run` refuse a plan with `pass=false` or recorded problems.

**Remaining defects**

1. **Budget retry can be blocked before the pilot starts** — [31_recovery_v1_oracle_xy.py:272](scripts/31_recovery_v1_oracle_xy.py:272). `cmd_run` exclusively creates `budget_gate_at_start.json` *before* checking the new projection and minimum price/spend. If that check refuses the run, no ledger or rollout exists, but a corrected `run` command fails on the existing JSON file.

2. **The audit timing omits tracking work** — [31_recovery_v1_oracle_xy.py:235](scripts/31_recovery_v1_oracle_xy.py:235) and [31_recovery_v1_oracle_xy.py:244](scripts/31_recovery_v1_oracle_xy.py:244). The audit replays each triggered scene to the trigger and runs four tracking probes, but `seconds_per_env_step` divides only `collect_scene` wall time by its counted steps. This is a narrower timing measure than the full audit; it can understate the host time used before the pilot budget gate. The pilot’s own two-scene gate provides a later check.

I found no additional concrete indexing, ground-truth input, handoff, restore, serialization, argument, or shell phase-order defect in the inspected paths. In particular, `rows[k - 1]` is used only after `k > 0`; `trunk_rows[t - 1]` denotes the trigger state; restore clears the tracking probe’s intervention; and `write_json` uses exclusive creation.

**Could not check:** behavior of the pinned ManiSkill controller objects on the real server, GPU execution, or whether the actual policy reaches the trigger on the audit scenes. This was a source-only, read-only pass; no simulator run or tests were executed.

_Note: link targets in this copy were shortened to repository-relative paths; the text is otherwise as returned by the reviewer._
