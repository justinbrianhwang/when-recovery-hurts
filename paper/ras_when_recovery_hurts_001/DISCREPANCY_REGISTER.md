# Discrepancy register

Differences found while writing the paper, and how each was resolved. Nothing in `results/` or `runs/` was changed.

| # | Where | Discrepancy | Resolution |
|---|---|---|---|
| D1 | Recount vs server/result files | None. All counts identical; all p-values equal within 1e-12 (implementations A and B). B3 equals the server `analysis.json`. | — |
| D2 | Interval method | Earlier reports quoted Wilson 95% intervals (e.g. Final Test harm 0–1.1%). The paper uses Clopper–Pearson (exact), because Wilson 1927 could not be verified as a reference in this session and CP matches the exact tests. | Manuscript uses CP; values differ slightly from the Wilson values in `results/*REPORT.md`. Both are descriptive. |
| D3 | Check 4 histogram | `CHECK4_SUMMARY.json` reports V2 first-step cosines 0.63–0.88; Fig. 6c also shows one branch near 0.16. | That branch is an oracle move (scene 51101, seed 1), not V2; the caption says "all 715 acted branches of all arms". |
| D4 | Failure diagnostic counts | README lists release labels 5+5+3 (=13) among 21 failures, and separately "the 14 release failures" (post hoc reading). | Supplementary S2 says "the 14 failures that released the cube over the target" (post hoc reading) and does not claim a label count. |
| D5 | B3 "triggered" vs C1 | B3 analysis counts were taken from the raw records; the server analysis only reports H_arch. | Checked equal for H_arch; other B3 counts come from the recount. |
| D6 | Acted counts of the oracle | The oracle enters control at every trigger (acted 195/191), even if the first step finds it already aligned. | Table 2 caption states this. |
| D7 | "Fewer than one in five" | First draft said V2 acts in fewer than one in five triggered scenes; actual 41/195 = 21%. | Corrected to "about one in five". |
| D8 | ACT rule | First draft wrote the ACT condition as offset ≥ max(15 mm, u). Code: offset ≥ 15 mm (ALIGNED) and offset − u ≥ 10 mm (ALIGNED_WITHIN_UNCERTAINTY). | Corrected in §4 and Table S3. |
| D9 | Trigger rule | First draft described the trigger with the hold-memory width (0.075 m). Code (`recovery_v1_oracle.trigger_rule`): open command after ≥ 3 close commands and finger opening 0.02–0.06 m. | Corrected in §4. |
| D10 | Final Test host check wording | First draft said the host check reproduced development scenes "on the test host" as dev records. It reproduces the reference recorded on the calibration host. | Corrected in §5. |
