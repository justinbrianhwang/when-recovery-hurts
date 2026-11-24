# Claim–evidence matrix

Every sentence in the manuscript that states a result maps to an evidence row (`EVIDENCE_LEDGER.csv`) and to a figure or
table. "Strength" says what kind of statement the evidence supports. Numbers were checked automatically against
`data/recount.json` (`figures/source_code/check_numbers.py`, 31 checks, 0 failures).

| # | Claim (manuscript section) | Evidence | Figure/Table | Strength | Must not be read as |
|---|---|---|---|---|---|
| 1 | Recovery can turn baseline successes into failures; rescue and harm must be counted separately (Intro, §3) | E01–E02, E09–E10, E15 | F1, T2, T4 | definitional + observed (1 harm in dev seed 1, 1 in test seed 2) | a claim that harm is frequent |
| 2 | V2 improves on C0 for the policy it was not developed on (§6) | E03 | T3 | pre-registered development test | a confirmatory result (that is claim 5) |
| 3 | Confirmed multi-frame estimate improves on single-frame (§6) | E04 | F4, T3 | pre-registered development test | proof that memory alone causes the gain (confound: m=1 also changes confirmation count) |
| 4 | Conditional harm ≤ 5% for both policies, δ = 0.05 (§7) | E06–E07 | F3a, T4 | confirmatory certificate on the calibration distribution, 595.99.02 host | physical safety; other tasks; other hosts |
| 5 | V2 increases success on held-out scenes (§7) | E09–E10 | F3b,c, T4 | confirmatory, run once, frozen manifest | an effect size estimate with guaranteed precision (bootstrap CI is post hoc) |
| 6 | The single dev harm (51157) involved under-tracking and the time budget (§8) | E15, E16 | F5, F6, S1, S3 | one-case descriptive mechanism | a causal decomposition |
| 7 | Direction of the move is right after step 1; Check 4 failures are a first-step lag (§8) | E16 | F6 | pre-declared check FAIL kept + post hoc reading | a PASS of Check 4 |
| 8 | Missed oracle rescues come from abstentions (§8) | raw C1 decisions (computed in this paper) | Table S1/S2 | descriptive (1 and 5 scenes) | evidence that abstention rules are optimal |
| 9 | S6 fails; partly explained by sub-step timing; residual unexplained (§9) | E17–E19 | F7, T5 | pre-declared FAIL + post hoc validation | validated contact separations |
| 10 | S4/S5 statuses are insensitive to mesh vs hull (§9) | E20 | F8, T5 | post hoc exploratory | continuous-time collision freedom |
| 11 | Outcome statistics do not depend on S4–S6 (§9) | design: Y from the simulator success flag | — | by construction | that the simulator is physically exact |
| 12 | V2 transfers unchanged to a PlainConv policy (§10) | E13–E14 | F9, T6 | one pre-declared development test | certified harm for PlainConv; transfer to other tasks |
| 13 | Host difference and smoke deviation (§12) | E22–E23 | — | documented limitation | invalidation of the calibration (its pairs come from one host and one session) |
