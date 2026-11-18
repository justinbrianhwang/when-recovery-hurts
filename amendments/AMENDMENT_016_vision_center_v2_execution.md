# Amendment 016 (execution) — Temporal Vision-CENTER v2, Phase B

Date: 2026-11-18 (UTC). Work id `vision_center_v2_001`. Plan SHA256 `1b4ea3bb65bd4cfdffc13a610fee010d568a5837e632701d4b0d265e6ba66e86`,
approved by the user by hash with a cap of US$5.00. Stage B1 and stage B2 were each run once. The preparation draft of this
amendment is kept as written.

## Approval and decisions fixed before the run

Base camera allowed (it is part of the policy observation); v1 unchanged as the baseline; the memory, uncertainty, abstain
and collision rules frozen as coded; the B1 and B2 criteria of the plan kept; simulator ground truth only for offline labels
and for CENTER_ORACLE.

## Pre-execution review

A read-only review (Codex) before any server was created returned NO-GO with three findings: the budget gate trusts the
spend and price it is given; the observation at the trigger step enters the memory although the plan text says "before the
trigger"; the memory is not updated during the move. They were reported without starting. The user replied to proceed
("오케이 우선 싹 다 해보자"), which was read as: run the unchanged code and plan and record the findings as limitations.
No code, constant or criterion was changed.

## What was run

One unusable host (driver 535, CUDA did not initialise) was stopped by the environment check before any scene and deleted.
On the second host (France, RTX 4090, Ryzen 7 7800X3D, driver 595.99.02, thread limit 1): install, tests (346 passed,
5 skipped), plan and code hashes, placement check of the 200 new scenes (no identical placement against 1,947 on record),
audit of 3 used scenes, budget gate (US$2.39 projected), stage B1, budget gate (US$1.70), stage B2, pack.

## Results

- Stage B1, 300 used scenes (development data): C0 253, v1 265, v2 275. v2 against C0 rescue 22, harm 0; v1 rescue 12,
  harm 0. Unnecessary acts 0, target pushed 0, memory errors 2 (limit 2), acts using memory 15. `V2_SHADOW_PASS`.
- Stage B2, 200 new scenes 50900–51099: C0 165, v1 173, v2 184, CENTER_ORACLE 190. v2 against C0 rescue 19, harm 0, net +19;
  v1 rescue 8, harm 0. v2 recovered 19 of 25 oracle rescues (76 %), v1 8. Memory errors 1 (limit 1). `V2_GATE_PASS`.
- v2 moves in stage B2: 34 (current frame 23, hand-camera memory 5, base-camera memory 6); all 34 scenes succeeded; largest
  target displacement during a v2 move 0.06 mm.
- Missed oracle rescues (6): held cube too low 2, offset inside the error bound 2, held-cube fit rejected 1, target never
  seen 1.
- The server result equals the local recomputation from the downloaded records.

## Limits

- An engineering gate, not a significance test; one task, one policy, one noise repeat, simulator images.
- The memory-error criterion was met with no margin in both stages (errors 5.0–5.2 mm, inside the bound carried).
- Scene 51009: the target was pushed 38.8 mm unseen in the 12 steps before the trigger; the memory kept the old position
  with a 10 mm bound and the hand-image check did not contradict it. v2 abstained only because another rule applied.
- Rollouts differ across hosts (uninterrupted outcome in 16 of 300 used scenes against run2, equal totals); inside the
  host C0 and every abstaining arm reproduced the uninterrupted run in every scene.

## Deviations

- Two test assertions written for the state before execution were updated (the pilot ledger now exists; the execution
  records name used scenes).
- The first instance was unusable and replaced by another host; nothing had run on it.

## Cost

US$0.97 of US$5.00 (Vast credit difference). No instance left.

## Not done

No training, no calibration, no final Test, no second pass, no change after an outcome, no further experiment.

## Records

`results/vision_center_v2_001/execution/` (scene records, traces, audit, gates, ledger, server logs, cost, post hoc
analysis). RGB frames (3.5 GB) are kept outside Git.
