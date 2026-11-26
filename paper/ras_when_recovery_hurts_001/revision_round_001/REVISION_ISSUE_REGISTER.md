# Revision issue register — round 001 (2026-11-25)

Canonical source: `Paper/Latex` (= repo `paper/ras_when_recovery_hurts_001/latex_project/`). Pre-revision backup:
`Paper/manuscript_package/revision_round_001/backup_Latex_before_revision/` (47 files, identical to repo commit `6027f0e`).
No experiment, frozen file, Calibration or Final Test was changed or re-run.

| ID | Severity | Location (before) | Problem | Evidence (code / data) | Fix | Status |
|---|---|---|---|---|---|---|
| R01 | BLOCKER | §4 "Bounded move and hand-back", Fig. 2, abstract | "The held release command is then passed to the unchanged policy" — false. The queue is cleared at the start and end of the move; the pending open command is discarded; the policy re-plans | `vision_center_v2.py` L224 and L253 `r.policy.clear()`; `policies.py` `clear(): queue=[]`; runner asserts no inference during the move and an empty queue at hand-back (L313) | Text, Fig. 2 and abstract rewritten; hand-back statistics added (all acted branches re-infer at the first step; first open after 1–24 steps; 1 branch never opens) — Table S4 | FIXED |
| R02 | MAJOR | §4 "Trigger", Fig. 2 "while the cube is held" | Wording implied grasp ground truth. The rule is observable only: queued open command, ≥3 executed close commands, finger opening 0.02–0.06 m | `recovery_v1_oracle.trigger_rule` L81–83 | Rewritten; wrong-object trigger 51152 described (target grasped; all vision arms abstained NO_HOLD_ESTIMATE; only oracle moved) | FIXED |
| R03 | MAJOR | §4 "no vertical and no gripper command" | The action carries the last executed (close) gripper command | `recovery_v1_oracle._xy_action` L142–144 `a[-1] = grip` | "zero vertical command and the held close command" | FIXED |
| R04 | MAJOR | §4 memory paragraph, Fig. 2 "hand / base / hold memories" | There is ONE target memory (wrist fit preferred, else base estimate; precise source owns the position) plus a hold memory | `vision_center_v2.Memory._observe_target`, `update` L81–102 | Rewritten; Fig. 3 caption explains "memory hand/base" = source of stored position | FIXED |
| R05 | MAJOR | §4 decision | ACT rule correct, but order of rules and LOW_CARRY for remembered targets missing | `vision_center_v2.decide` L172–183 | Ordered rule list in §4 and Supp S3 | FIXED |
| R06 | MAJOR | §4 / §8 "tracking ~40 %" | Mixed per-step and path ratios | `handback_stats.json`: net XY / commanded path, dev 0.26–0.40, medians 0.36/0.37; no move ended at the cap | Exact ratio, range, medians; "observed property, not a bound" | FIXED |
| R07 | MAJOR | §4 "V2 rules off disables the abstention rules" | Only 3 V2-specific rules removed as one bundle; V1 rules remain | `vision_center_c1.py` L17, L36 `RULES_OFF_REMOVED`, `decide_rules_off` L130 | Exact definition in §4, §11, §12, Supp S3 | FIXED |
| R08 | MAJOR | §8 Check 4 "progress ≥0.2 mm along the command" | Length condition is on the executed step magnitude | `trajectory_audit.py` L22–23, L134–135 (`an >= PROGRESS_MIN_M`) | Definition corrected (§8, Supp S5) | FIXED |
| R09 | MAJOR | Fig. 5(old)/6 caption "because the arm is still moving" | Causal claim not established (no controller internals logged) | `CHECK4_DIAGNOSTIC_REPORT.md` | "associated with … cause not established"; 51157 passed Check 4 stated | FIXED |
| R10 | MAJOR | §9 S5 "reconstructed penetrations ≤1 mm" | S5 = non-zero-impulse pairs reconstructed ≤1 mm APART (one-sided); penetration reported without threshold | `spatial_links.py` L179–180, L202–207 | Definition rewritten (§9, T5, Supp S6) | FIXED |
| R11 | MAJOR | §9 S6 "each contact pair listed" | S6 = listed pairs with PhysX separation > 0, all phases | `spatial_links.py` L181, L209–213 | Rewritten | FIXED |
| R12 | MAJOR | Supp S6 "independent reconstruction within 1e-12 mm for all kinds except cubeB–table" | Hand–cube pairs differ by 0.22–0.24 mm | `S6_DIAGNOSTIC_SUMMARY.json` A2 by_kind (hand-cubeA 0.241, hand-cubeB 0.220) | Corrected; hull swap also changes hand distances ≤0.24 mm | FIXED |
| R13 | MAJOR | §9 S4 failure 51152 | Presented as a move collision; actually the target was grasped at the trigger (−2.16 mm, ~27.6 N per finger before the move) | `S45_FAILURE_DETAILS.json` | Explained in §9, §11, T5, Supp S6 | FIXED |
| R14 | MAJOR | §3 null hypothesis `h > ε` | p = F_Bin(n+,ε)(k) is the test of H0: h ≥ ε (boundary included) | — | H0 changed to h ≥ ε everywhere (main, captions, Supp S1); data/p-values unchanged | FIXED |
| R15 | MAJOR | §3/§7 meaning of the certificate | Read as "true risk ≤ 5 %"; must be frequentist false-certification ≤ δ under the sampling assumption | — | Explicit statement in §3, §7, §12, Supp S1 | FIXED |
| R16 | MAJOR | §3 "we treat distinct seeds as independent" | i.i.d. is an assumption, not proved by distinct seeds | — | Stated as an assumption | FIXED |
| R17 | MAJOR | §11 "A recovery that reads its own executed displacement" | Self-contradiction: V2 already re-reads TCP and re-estimates every step | `vision_center_v2.py` move loop | Replaced by untested feasibility/value gate | FIXED |
| R18 | MAJOR | §1 / §7 / §11 abstention | Certificate margin mostly from abstention; rules-off had no extra harm | recount; review numbers | "Strongest objection" paragraph in §1; §7 "What carries the certificate"; §11 | FIXED |
| R19 | MAJOR | Figure numbering vs files | F6 printed as Fig. 5, F4 as Fig. 3, F3 as Fig. 4 | build | Files renamed by print order (Fig01…Fig09, FigS1…S3); citation order fixed in §8 | FIXED |
| R20 | MAJOR | references | Holm DOI does not resolve; SAPIEN pages; ManiSkill3 published RSS 2025 | Crossref / RSS proceedings (agent check) | bib corrected; HG-DAgger, Recovery RL, ResNet verified | FIXED |
| R21 | MAJOR | Deviation wording | "author takes responsibility" was written by the agent | — | Removed; factual "not approved in advance; reviewed afterwards"; wording is an author item | FIXED (author to confirm) |
| R22 | MINOR | §5 two ResNet-18 seeds | Must not read as two architectures | — | "differ only in the training seed" (§5, §10, §12) | FIXED |
| R23 | MINOR | §2 Sentinel/FIPER | Their calibration controls detector false alarms | agent check (PMLR, arXiv) | Stated; RESCUE described fairly | FIXED |
| R24 | MINOR | Supp tables | Table S4 unreadable; long `\texttt` overfull 69–131 pt | build log | Compact table; `\path` with underscore breaks | FIXED |
| R25 | MINOR | page 1 | footnote/footer collision after abstract growth | render | abstract trimmed to 244 words | FIXED |
| R26 | OPEN | RAS Guide for Authors | All RAS-specific pages 403 | agent check | Marked UNVERIFIED; author must check in a browser | OPEN |
| R27 | OPEN | Author items (name, affiliation, e-mail, CRediT, funding, data link, AI wording, deviation wording) | Only the author can decide | — | Placeholders kept; `AUTHOR_APPROVAL_ITEMS.md` | OPEN |
| R28 | OPEN (science) | No external baseline; noise floor; single task | Needs new experiments | — | Stated as limitations; `FUTURE_WORK_ONLY.md` | OPEN (not fixable without experiments) |

Counts: BLOCKER 1 (fixed 1); MAJOR 20 (fixed 20); MINOR 4 (fixed 4); OPEN 3 (2 author/journal, 1 requires new experiments).
