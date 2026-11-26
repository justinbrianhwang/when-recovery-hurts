# Journal compliance — verified vs unverified (2026-11-25)

**All RAS-specific pages returned HTTP 403** in this session:
- `https://www.sciencedirect.com/journal/robotics-and-autonomous-systems/publish/guide-for-authors` → 403
- `https://www.elsevier.com/journals/robotics-and-autonomous-systems/0921-8890/guide-for-authors` → 301 → 403
- open-access options page → 403

So every RAS-specific rule below is **UNVERIFIED**; generic Elsevier rules are marked with their source.

| Item | Value used | Source | Status |
|---|---|---|---|
| LaTeX class | elsarticle (preprint, 12pt, elsarticle-num) | elsevier.com/researcher/author/policies-and-guidelines/latex-instructions | generic VERIFIED |
| Flat upload | separate flat zip (`flat_export/RAS_source_flat.zip`, 50 files, no subfolders); `Paper/Latex` keeps subfolders for Overleaf | same page ("no subfolders") | generic VERIFIED |
| .bbl / .bst / .bib | all included in the flat zip | same page (all accepted item types) | generic VERIFIED |
| Overleaf TeX version | page states TeX Live 2022 for Overleaf | same page | generic VERIFIED (not tested on Overleaf here; compiled with Tectonic 0.15.0) |
| Highlights | 5 bullets, ≤85 characters | elsevier.com/researcher/author/tools-and-resources/highlights | generic VERIFIED; RAS requirement UNVERIFIED |
| Abstract length | 244 words | — | RAS limit UNVERIFIED |
| Keywords | 7 | — | UNVERIFIED |
| Reference style | numbered (elsarticle-num) | — | UNVERIFIED for RAS |
| Graphical abstract | not prepared | — | UNVERIFIED |
| Page/word limit | main 35 pp (preprint 12pt, line numbers) | — | UNVERIFIED |
| Supplementary | separate PDF (12 pp); LaTeX sources must not be uploaded as supplementary items | LaTeX page | generic VERIFIED |
| Generative-AI declaration | separate section immediately before references, Elsevier title | elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals | generic VERIFIED; wording is an author item |
| Data statement | placeholder | Elsevier research-data page 404 | UNVERIFIED |
| APC | not used | only a third-party 2021 figure found | UNVERIFIED |

Action for the author: open the RAS Guide for Authors in a browser and confirm the UNVERIFIED rows.
