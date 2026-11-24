# Reviewer critique and fixes

An independent reviewer agent read the uncompiled manuscript, the tables, `recount.json` and the result reports
without the author's conclusions. Its claim check found **no numeric mismatch**. Below: each concern, and what was
changed (text and analysis of existing data only — no new simulation).

## Major

| # | Concern | Response / change |
|---|---|---|
| M1 | The certificate is diluted by structural zeros: harm can only occur where V2 moves (48/348, 36/337 baseline successes), so 5% is easy to meet; restricted to acted successes the CP upper bounds are 9.5% / 8.0% (cal), 7.6% / 9.9% (test). | **Accepted.** §7 now reports harm among acted baseline successes with CP bounds, states that the certificate is about the deployed system (policy + gated controller), and justifies ε = 5% (≈ 17 lost successes of 348, more than half the dev rescue gain; fixed at project start). Discussion adds that most of the margin comes from non-acting scenes. Numbers recomputed from raw records. |
| M2 | "Abstention is where harm is controlled" is contradicted: rules-off had the same harms (1, 0) and more rescues. | **Accepted.** Heading changed to "Abstention as a precaution"; text says the development data do not show that the rules reduce harm. |
| M3 | Executed move ≠ commanded move (≈ 40% tracking); x^cmd undefined; offsets above what one move can correct. | **Accepted.** x^cmd defined (x_t + Δx_t); §4 states the ≈ 2 of 5 mm tracking and that the cap rarely binds; §8 reports acted scenes with estimated offset > 40 mm: dev 3 (1 rescue, 1 harm = 51157), cal 0, test 8 (6 rescues, 1 harm — the only test harm). |
| M4 | Discordance may include policy-noise flips after resumption. | **Accepted.** §5 "Policy sampling noise": noise keyed to scene and step; matched-hold pilot gave 1 rescue / 1 harm on 100 scenes as an approximate floor; efficacy is a system-level comparison. Added to Limitations. A matched-delay arm on test scenes would need new runs (not done). |
| M5 | No external baselines; task-specific controller. | **Partly accepted.** Intro now frames the controller as a case study of the protocol; Limitations adds "no external baseline". Physics audit kept in the main text because S6 FAIL is a declared check whose status the paper must report; a learned-recovery baseline needs new experiments (not done). |
| M6 | Mechanism evidence comes from the development host, not the certified configuration. | **Accepted.** Intro claim softened ("descriptive mechanism analysis"); §8 Limits states the host and that 51157's C0 outcome differs on the calibration host; §9 states B4 ran on the calibration host. |
| M7 | Process accountability (AI agent; deviation decision). | **Accepted.** §5 "Role of the AI agent" paragraph; deviation text (main §12, Supp. S9) states that the agent took the decision without prior approval and that the author reviewed it and takes responsibility. **Author must confirm this wording (A8, A9).** |

## Minor (selected)

| # | Concern | Change |
|---|---|---|
| 1 | Figure numbering follows citation order | F1 moved to the Introduction; control figure placed before the spatial figure; floats `[!htbp]` |
| 2 | Oracle "Acted" = triggers | Table 2 caption: actually moved in 93 and 102 |
| 3 | H0 written `>` vs `≥` | Supplement aligned to `h > ε` |
| 4 | h* vs ε | F3 caption: "ε = 5% (marked h*)" |
| 6 | Fig. 4a missing category | text adds "no target estimate in 8 and 10" |
| 7 | 1.7–4.7 cm sample | stated as post hoc reading of 100 diagnostic scenes, 14 failures |
| 8 | 3,874 branches unexplained | 400 trunks + 9 arms × 386 triggered scene-runs |
| 9 | 3.3% residual scope | Limitations: on 11 sub-step-logged scene-runs |
| 12 | Holm thresholds for H_gen/H_mem | Table 3 caption |
| 13 | i.i.d. seed assumption | stated in §3 |
| — | Overstatements (abstract "bounded", conclusion "shown to keep", "absence of harm", PlainConv "corrects") | all reworded |

Not changed: equation-reference style (all resolve correctly after compiling); placeholders (author items).
