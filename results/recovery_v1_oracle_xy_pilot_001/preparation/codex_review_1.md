## Findings

1. **High — the 3 cm target cap can be exceeded.** [recovery_v1_oracle.py:159](src/hci/recovery_v1_oracle.py:159) projects `goal` onto the cap, then limits the step from the *current* TCP position. If the measured TCP has drifted outside the cap, that limited step can still command a target outside it. For example, at 4 cm from the trigger position, a 5 mm step toward the 3 cm boundary targets 3.5 cm. Neither the controller nor `validate_scene` rejects this case.

2. **Medium — the spending gate does not bind the pilot’s cost inputs.** [31_recovery_v1_oracle_xy.py:264](scripts/31_recovery_v1_oracle_xy.py:264) accepts a prior `budget_gate.json` by its pass flag and plan hash. The `run` command takes fresh `--usd-per-hour` and `--spent-usd` values without comparing them with the values used by `budget`. A budget check made with lower figures can therefore authorize a pilot whose stated cost exceeds the approved cap.

3. **Medium — an invalid plan is not rejected at execution.** [31_recovery_v1_oracle_xy.py:161](scripts/31_recovery_v1_oracle_xy.py:161) checks file hashes but never checks `plan['pass']`. A plan written with recorded problems could still pass the server commands if it has a matching authorization file and later gate files. The plan’s preparation failure is therefore not an execution gate.

## Checks I could not perform

I did not run the new code, the real policy, or the pinned simulator on the server. I could not verify live controller attributes, physical tracking and drift, or real server serialization and packaging behavior.

_Note: link targets in this copy were shortened to repository-relative paths; the text is otherwise as returned by the reviewer._
