# Final writing report — RAS manuscript "When Recovery Hurts"

Date: 2026-11-23. No new simulation, training, Final Test rerun or paid GPU. `results/` and `runs/` unchanged.

## Deliverables

| File | State |
|---|---|
| `submission/RAS_main.pdf` | compiled (Tectonic 0.15.0, elsarticle preprint 12pt), **29 pages**, line-numbered |
| `submission/RAS_supplementary.pdf` | compiled, **8 pages** |
| `submission/RAS_source_flat.zip` | 24 files, no subfolders, includes `main.bbl` and `elsarticle-num.bst` |
| `submission/FINAL_PDF_AND_SUPPLEMENT_MANIFEST.json` | SHA-256 of PDFs, zip and every packaged file |
| `manuscripts/` | main.tex, supplementary.tex, references.bib (28 entries), abstract.txt (234 words), highlights.txt (5, ≤ 75 chars) |
| Figures | 9 main (F1–F9) + 3 supplementary (S1–S3), vector PDF from original records; provenance hashed |
| Tables | main: 6 (Table 1 inline, T2–T6 generated); supplementary: 4 (2 generated, 2 inline) |
| Documents | START_HERE, README_BUILD, EVIDENCE_LEDGER, CLAIM_EVIDENCE_MATRIX, DISCREPANCY_REGISTER, RECOUNT_REPORT, NOVELTY_POSITIONING, RELATED_WORK_MATRIX, JOURNAL_COMPLIANCE, FIGURE_STORYBOARD, FIGURE_CLAIM_AUDIT, FIGURE_PROVENANCE.csv, ATLAS_REUSE_AUDIT, FIGURE_CAPTIONS, REVIEWER_CRITIQUE_AND_FIXES, SUBMISSION_CHECKLIST, SOURCE_AND_HASH_INVENTORY.json, editorial/ (cover letter, AI declaration, author items) |

## Verification

- Recount from hash-checked raw records, two implementations, 0 discrepancies (`RECOUNT_REPORT.md`).
- Manuscript numbers vs recount: 31 automated checks, 0 failures (`build_logs/check_numbers.json`).
- Compilation: no undefined references or citations; one 14 pt overfull vbox on page 1 (elsarticle preprint footer),
  visually checked — no overlap. Every page of both PDFs was rendered and inspected; layout issues found during
  inspection (footnote/footer overlap, floats pushed to the end, an arXiv note in a journal entry) were fixed.
- Independent reviewer agent: no numeric mismatch; 7 major and 15 minor points, handled as described in
  `REVIEWER_CRITIQUE_AND_FIXES.md` (text and existing-data analysis only).

## New numbers computed for the paper (from existing raw records; descriptive)

- Harm among baseline successes in which V2 moved: cal 1/48, 0/36 (CP 95% upper 9.5%, 8.0%); test 0/38, 1/46 (7.6%, 9.9%).
- Acted scenes with estimated offset > 40 mm: dev 3 (1 rescue, 1 harm), cal 0, test 8 (6 rescues, 1 harm).
- Oracle actually moved in 93 / 102 development scenes (enters control at 195 / 191).
- V2 decisions per stage (Table S3); missed oracle rescues in development: 1 and 5, all abstentions.
- Post hoc CP bounds and paired bootstrap CIs (labelled post hoc).

## Unresolved

- RAS Guide for Authors could not be read (HTTP 403): reference style, limits and graphical abstract unverified.
- 6 references with unverified DOI/pages (flagged in `references.bib`).
- S6 remains FAIL; the residual 1,874 finger-box proximity pairs are unexplained.
- No external baseline and no matched-delay arm on confirmatory scenes (would need new runs).

## Author items

See `editorial/author_items_to_confirm.md` (A1–A16): name, affiliation, e-mail, CRediT, competing interests, funding,
data link, AI-declaration wording, deviation wording, Final Test token disclosure, journal-guide checks, unverified DOIs.
