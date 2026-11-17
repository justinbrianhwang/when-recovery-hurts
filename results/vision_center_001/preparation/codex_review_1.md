This is a **read-only review**. I did not run tests, the driver, or a simulator.

### Concrete defects

- [vision_center.py:366](src/hci/vision_center.py:366) checks only the queue length, step, intervention flag, and inference calls after an abstention. If `vision_center()` changes a queued action, policy history, or a counter while preserving queue length, this check passes. The later trunk comparison may catch a changed outcome or trajectory, but it does not establish the stated requirement that the runtime is untouched at abstention.

- [vision_center.py:373](src/hci/vision_center.py:373) uses `rows[k - 1]` for the state after control. `k` is `r.t - t`; `rows` contains one probe row per executed step, including steps from the subsequent `r.complete()` only after this expression. The index is correct when control executes `k` steps, including an episode ending during control. I found no row indexing defect there.

- [32_vision_center.py:265](scripts/32_vision_center.py:265) requires `scenes.json` to exist but does not check its `pass` value or plan hash. Running `shadow` directly after a placement check that wrote `{"pass": false}` still starts the 200-scene shadow pass. The shell wrapper stops on the failed `scenes` command, but the Python command does not enforce that prerequisite.

- [32_vision_center.py:243](scripts/32_vision_center.py:243) projects only the current stage after five shadow scenes. If shadow takes longer than the audit estimate, this gate can pass while the remaining shadow work **plus the planned pilot** exceeds the approved cap. The initial shadow projection includes both stages; this later projection drops the pilot.

- [32_vision_center.py:290](scripts/32_vision_center.py:290) writes `budget_gate_pilot_at_start.json` before writing the ledger. If ledger creation then fails—for example, because `artifacts/ledger` does not exist when `run` is called directly—the pilot is refused, but the budget file remains. The driver also does not check whether [the ledger already exists](scripts/32_vision_center.py:291) before writing it; a direct retry can overwrite the prior ledger before the existing `pilot` directory blocks the run at line 287 only when that directory exists.

- [run.sh:40](scripts/vision_center/run.sh:40) lets `pack` run without `PILOT_DONE`. With a partial or failed pilot, it can produce an archive that looks like a completed handoff.

I did **not** find a path where simulator object poses enter `VISION_CENTER`’s decision or control, an offset sign reversal in `reading()`, or a command/step off-by-one in `vision_center()` from the code inspected. The stated sign is `target − held cube` at [vision_center.py:235](src/hci/vision_center.py:235).

I could not check whether the fixed hand-camera transform and colour thresholds match actual ManiSkill frames, whether the geometric fits work on simulator images, or whether abstaining reproduces an uninterrupted simulator episode. Those require execution, which you prohibited.

_Note: absolute local paths in this copy were shortened to repository-relative paths; the text is otherwise unchanged._
