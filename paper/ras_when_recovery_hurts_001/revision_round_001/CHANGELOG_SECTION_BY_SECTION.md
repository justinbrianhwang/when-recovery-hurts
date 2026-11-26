# Changelog by section — revision round 001

| Section | What changed | Why (issue IDs) |
|---|---|---|
| 00 Abstract | Hand-back described as queue cleared + re-planning; certificate stated as rejecting h ≥ 5 % with family-wise error δ; "much of the margin comes from abstention" added; trimmed to 244 words | R01, R14, R15, R18, R25 |
| 01 Introduction | New paragraphs: Δsucc=(R−H)/n with the (30,0) vs (60,30) example; detection vs intervention; why this case study (observation-only, frozen, once, abstain = C0); "strongest objection stated first" (abstention margin, rules-off); contributions and non-claims rewritten (no novelty claim for paired counterfactual evaluation, binomial bounds or risk control) | spec §3, R18 |
| 02 Related work | Sentinel/FIPER calibration described as detector false-alarm control; RESCUE described fairly (same-state counterfactual interventions; our additions are certificate, held-out test, motion/physics audit); memory-guidance study linked to our ablation caveat; table updated | R23 |
| 03 Problem formulation | Scene/seed/noise definition; i.i.d. as explicit assumption; H0: h ≥ ε; p-value written as Pr(Bin ≤ k); frequentist meaning of the guarantee; unit of analysis; ground truth never fed to the controller | R14–R16 |
| 04 Vision-CENTER V2 | Rewritten from code: observable trigger with 51152 wrong-object case; abstain invariance; one target memory + hold memory; ordered decision rules; queue cleared at ACT; held close command; closed-loop re-estimation and stop rules; hand-back re-planning with measured timing; executed-motion ratio; exact ablation definitions (single frame favoured; rules-off bundle) | R01–R07 |
| 05 Protocol | pd_ee_delta_pos and 5 sub-steps; seeds differ only in training seed; plan hash; test ran once; noise-floor arm described with the same hand-back; responsibility sentence removed | R21, R22 |
| 06 Development | Framed as motivation, not confirmation; memory-source counts (81/83 from memory); per-source ablations descriptive | R04, R07 |
| 07 Calibration/Test | H0 h ≥ 0.05; Holm thresholds per policy; "What carries the certificate" paragraph (acting-subset bounds post hoc); ε justification corrected (≈17 > 14 dev rescues); pooled counts descriptive; test harm counts descriptive | R14, R15, R18 |
| 08 Mechanism | Rescue described with re-planned release; spatial figure placed before control figure (print order = file order); 51157 passed Check 4; release 3 steps after hand-back; Check 4 definition corrected; association vs cause; selection stated | R01, R08, R09, R19 |
| 09 Physical record | Four quantities defined; S4/S5/S6 definitions from code; 51152 = target grasped at trigger; S5 two events named; hull changes hand ≤0.24 mm; data sources A/B/C kept apart in text and Fig. 7 caption | R10–R13 |
| 10 Second architecture | "two ResNet-18 policies differ only in training seed"; development-level framing | R22 |
| 11 Discussion | "What the confirmatory results establish"; abstention as precaution with bundle caveat; future-work self-contradiction removed (feasibility/value gate, untested); wrong-object trigger paragraph | R17, R18, R07 |
| 12 Limitations | Guarantee wording; deviation wording without responsibility claim; ablation confounds include rules-off bundle; same-architecture policies | R15, R21 |
| 13 Conclusion | "below 5 %" with test qualifiers; abstention caveat; hand-back and executed motion mentioned | R15, R18 |
| Supplementary | S0 role statement; S1 assumption, H0 ≥, guarantee meaning, CP formula, acting-subset bounds; S3 implementation call path, ordered rules, Table S4 hand-back; S4 51157 timing; S5 Check 4 definition and associations; S6 three data sources, S4/S5/S6 definitions, 51152, independent reconstruction corrected; S9 deviation wording | R01–R13, R21 |
| Figures | Fig. 2 redrawn to the code; Fig. 4 label ε; Fig. 5 layout; files renamed by print order | R01–R04, R19 |
| Tables | T5 definitions corrected; new TS3 (Table S4) | R10–R13 |
| references.bib | Holm (no DOI, JSTOR URL), SAPIEN pages, ManiSkill3 RSS 2025; ResNet/HG-DAgger/Recovery RL verified | R20 |
| main.tex / supplementary.tex | underscore line-breaking for `\path` | R24 |
