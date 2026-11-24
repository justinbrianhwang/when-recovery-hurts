# Journal compliance — Robotics and Autonomous Systems (Elsevier)

Status of each requirement. "VERIFIED" = read on an Elsevier page during this session; "UNVERIFIED" = the RAS Guide
for Authors page returned HTTP 403, so the item follows general Elsevier practice and must be checked by the author
before submission.

| Item | What we did | Source status |
|---|---|---|
| LaTeX class | `elsarticle`, `preprint,12pt`, numbered references (`elsarticle-num`) | Elsevier LaTeX instructions: VERIFIED. RAS reference style (numbered vs author–year): UNVERIFIED |
| Flat source upload | `submission/RAS_source_flat.zip` has no subfolders; figures as PDF next to `main.tex` | VERIFIED (Elsevier: no subfolders) |
| Upload the `.bbl` | `build.py` copies `main.bbl` into the package after compiling | VERIFIED. Done (main.bbl in the zip) |
| `.bst` file | `build.py` copies `elsarticle-num.bst` from the TeX installation | VERIFIED. Done (from the Tectonic bundle) |
| Highlights | 5 bullets, each ≤ 85 characters (`manuscripts/highlights.txt`, max 75) | 3–5 bullets, ≤ 85 characters: general Elsevier rule, UNVERIFIED for RAS |
| Abstract | 234 words, no references | ≤ 250 words: general Elsevier practice, UNVERIFIED for RAS |
| Keywords | 7 | 1–7: UNVERIFIED for RAS |
| Generative-AI declaration | Section placed immediately before the references, with the Elsevier title | VERIFIED (Elsevier policy: placement and title) — wording is a DRAFT for the author |
| CRediT, competing interests, funding, data availability | Present as placeholders | Required by Elsevier: VERIFIED in general; texts must be completed by the author |
| Line numbers | `lineno` enabled for review | UNVERIFIED (common request) |
| Page / word limits | Main text ≈ 5,400 words excluding tables, figures and references; 9 figures, 6 tables | UNVERIFIED (403) |
| Graphical abstract | Not prepared (optional in many Elsevier journals) | UNVERIFIED |
| Supplementary material | Separate PDF (`RAS_supplementary.pdf`) | General Elsevier practice |
| Figures | Vector PDF, fonts embedded (Type 42), CVD-safe palette; camera images are the stored RGB frames | General Elsevier artwork rules |
| Cover letter | `editorial/cover_letter_DRAFT.md` | — |

Before submission the author must open the RAS Guide for Authors and confirm every UNVERIFIED row.
