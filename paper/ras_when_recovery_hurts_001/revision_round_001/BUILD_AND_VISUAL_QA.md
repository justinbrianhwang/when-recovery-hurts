# Build and visual QA — round 001

Engine: Tectonic 0.15.0 (bundle-provided elsarticle, elsarticle-num.bst). Logs: `build_logs/`.

| Build | Result | Pages |
|---|---|---|
| `main.tex` from project root | exit 0 | 35 (was 29) |
| `supplementary/supplementary.tex` from root | exit 0 | 12 (was 8) |
| `supplementary.tex` from inside `supplementary/` | exit 0, text identical to root build | 12 |
| flat export (`flat_export/flat/`) main / supplementary | exit 0, text identical to canonical builds | 35 / 12 |

Log checks: no undefined references or citations, no missing figures, no errors.

Remaining warnings (accepted after visual check):
- page 1 overfull vbox 14 pt (elsarticle preprint footer with e-mail footnote) — no visible overlap after the abstract
  was trimmed (an earlier 28.6 pt version overlapped and was fixed);
- overfull hbox 1–5 pt in §2 (table paragraph), §9, Supp S1, S3, S6 — not visible at print size;
- underfull boxes in narrow table columns (Table 1, Table 5).

Fixed during QA: Supplementary overfull 69–131 pt from unbreakable `\texttt` file paths (→ `\path` with underscore
breaks); Table S4 shrunk to illegible size (→ compact numeric columns); a broken caption quote in Table S4; Fig. 2 text
overflowing its boxes; Fig. 5 3D tick labels touching panel c; page-1 footer overlap.

Visual inspection: every page of both PDFs rendered at 100 dpi and inspected in contact sheets (main 35, supplementary
12). Figures appear in citation order with numbers matching the file names (Fig01…Fig09, FigS1…FigS3); equations
(1)–(7) numbered and referenced correctly; references list complete (28 entries, numbered).
