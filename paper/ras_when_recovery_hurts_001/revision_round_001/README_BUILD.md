# Build guide — revision round 001

## Source of truth and derivation direction
- **Canonical:** Drive `Paper/Latex/` = repo `paper/ras_when_recovery_hurts_001/latex_project/`. Edit only here.
- `make_latex_project.py` (old direction, single file → split) now **refuses to run** over `latex_project/`.
- Derived from the canonical project: `export_flat_from_latex.py` → flat Elsevier zip and PDFs.
- Figure/table generators (`figures/source_code/*.py`, `tables/source_code/make_tables.py`) write to `figures/` and
  `tables/`; the files are then copied into `latex_project/figures` (renamed by print order, see
  `FIGURE_CLAIM_AUDIT_REVISED.csv`) and `latex_project/tables`.

## Overleaf
1. Zip the contents of `Paper/Latex` and upload as a new project.
2. Main document `main.tex` → main manuscript.
3. Switch the main document to `supplementary/supplementary.tex` → Supplementary (paths resolve both from the root
   and from `supplementary/`).

## Local (Tectonic 0.15.0 used here)
```
tectonic main.tex
tectonic supplementary/supplementary.tex
```

## Flat Elsevier package
```
python -P paper/ras_when_recovery_hurts_001/export_flat_from_latex.py --out <new folder> --tectonic <path to tectonic>
```
Produces `flat/` (no subfolders), `RAS_source_flat.zip` with `main.bbl` and `elsarticle-num.bst`, compiled PDFs and
`FLAT_PACKAGE_MANIFEST.json`. The flat PDFs were checked to be text-identical to the canonical build.

## Number check
```
python -P paper/ras_when_recovery_hurts_001/figures/source_code/check_numbers_latex.py
```
