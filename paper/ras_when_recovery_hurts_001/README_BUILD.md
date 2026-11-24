# How to rebuild the paper

All commands from the repository root. Nothing writes into `results/` or `runs/`.

```bash
# 1. Recount every headline number from the raw records (about 40 min; needs the Git-ignored runs/ archives)
python -P paper/ras_when_recovery_hurts_001/figures/source_code/recount_all.py --out paper/ras_when_recovery_hurts_001/data/recount.json
# 2. Figures (vector PDF) and tables (LaTeX)
python -P paper/ras_when_recovery_hurts_001/figures/source_code/make_figures.py
python -P paper/ras_when_recovery_hurts_001/figures/source_code/make_supp_figures.py
python -P paper/ras_when_recovery_hurts_001/tables/source_code/make_tables.py
# 3. Check the manuscript numbers against the recount (exit 1 on any mismatch)
python -P paper/ras_when_recovery_hurts_001/figures/source_code/check_numbers.py
# 4. Flat Elsevier package, compile, zip, manifest (exit 2 if no TeX engine is installed)
python -P paper/ras_when_recovery_hurts_001/build.py --render-dir <scratch folder for page PNGs>
# 5. Provenance CSV, captions, hash inventory
python -P paper/ras_when_recovery_hurts_001/figures/source_code/make_inventory.py
```

Requirements: Python 3.11+ with numpy, scipy, matplotlib (3.11.1 used), pymupdf; the project package `src/hci` for
implementation A of the recount; a TeX engine (Tectonic, or TeX Live/MiKTeX with `elsarticle`) for step 4.

Outputs: `submission/RAS_main.pdf`, `submission/RAS_supplementary.pdf`, `submission/RAS_source_flat.zip` (flat: no
subfolders, includes `main.bbl` and `elsarticle-num.bst` after compiling), `submission/FINAL_PDF_AND_SUPPLEMENT_MANIFEST.json`.
`submission/RAS_source_flat/` is a derived folder and is regenerated on every build.
