# Reviewer risk matrix — strongest objections and what the current data can answer

| # | Objection | Severity | Answer available now | Not answerable without new experiments |
|---|---|---|---|---|
| 1 | "Low harm only because V2 rarely acts" | high | Stated first in §1; acting-subset bounds (≤10 %) in §7; certificate framed as system property | A certificate on acting scenes would need far more acting scenes |
| 2 | "Abstention rules not shown to matter" | high | Conceded (§11): rules-off had no extra harm in development; rules kept as precaution | Rule-level causal effect needs a dedicated ablation with confirmatory power |
| 3 | "One task, simulation only, hand-designed controller" | high | Framed as a protocol case study; limitations explicit | Other tasks / real robot |
| 4 | "No external baseline (learned recovery, retry)" | medium–high | Oracle and matched-hold pilot as references; limitation stated | Running baselines |
| 5 | "Discordance may be noise after re-planning" | medium | Noise keyed to scene and step; matched-hold pilot floor 1/1 per 100 scenes; efficacy is a system comparison | Matched-delay arm on confirmatory scenes |
| 6 | "Development host differs; mechanism not on certified configuration" | medium | Stated in §5, §8, §12; hand-back/decision statistics computed on calibration/test records too (Table S4, S3) | Re-collecting C1 on the calibration host |
| 7 | "Smoke criterion deviation decided by an AI agent" | medium | Disclosed factually in §12 and S9 with timing; runner reproduced same-host collector | — (author wording) |
| 8 | "S6 FAIL undermines the physical analysis" | medium | Outcomes/tests do not use S4–S6; partial explanation; residual stated | Logging controller internals / sub-steps on C1 host |
| 9 | "Memory claim confounded" | medium | Stated: H_mem tests memory + confirmation-count change | Clean ablation |
| 10 | "i.i.d. scenes assumed" | low–medium | Assumption stated; placement check | — |
| 11 | "Move under-tracks command (~36 %)" | medium | Measured and reported; feasibility gate proposed as future work | Controller re-tuning (would change the frozen candidate) |
| 12 | "Trigger can fire on the wrong object" | low–medium | 51152 reported; V2 abstained | Frequency estimate on confirmatory data (not logged as GT for all) |
| 13 | "Novelty" | medium | No first-claims; combination + calibration/test separation + code-level mechanism audit; RESCUE acknowledged | — |
