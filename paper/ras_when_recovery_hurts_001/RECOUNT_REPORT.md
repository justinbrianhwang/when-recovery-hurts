# Recount report

**Script:** `figures/source_code/recount_all.py` → `data/recount.json`. Read-only on the raw records.

## Procedure

1. For every scene record of every stage, the SHA-256 of its trajectory file was recomputed and compared with the
   `trace_sha256` stored in the record. Any mismatch, or any `*.error.json` record in a pass folder, aborts the run.
2. Every quantity was computed twice:
   - **A** — project code: `hci.v2_confirm.certificate`, `hci.v2_confirm.efficacy`, `hci.vision_center_c1.exact_one_sided`.
   - **B** — hand-written count of the paired cells from the raw records; `scipy.stats.binom` for the p-values; a
     hand-written Holm step-down.
3. Assertions: identical counts (n₊, harm, rescues), p-values equal within 1e-12, identical Holm decisions. B3 was also
   checked against the server `analysis.json`.
4. Post hoc, descriptive, labelled `_POSTHOC`: one-sided 95% Clopper–Pearson upper bounds of conditional harm; paired
   bootstrap (20,000 resamples of scenes, percentile, seed 20261123) of the success difference.

## Formulas

- Certificate: p = Σ_{j≤k} C(n₊, j) 0.05^j 0.95^{n₊−j}; Holm over the two policies at δ = 0.05.
- Efficacy: p = Σ_{j≥b} C(b+c, j) 2^{−(b+c)}; Holm over the two policies at α = 0.05.
- H_mem: intersection–union p = max(p_seed1, p_seed2).

## Result (all assertions passed; 0 discrepancies)

| Quantity | Seed 1 | Seed 2 |
|---|---|---|
| C1 C0 → V2 (rescue/harm) | 171 → 184 (14/1) | 164 → 180 (16/0) |
| C1 single frame | 175 (4/0) | 173 (9/0) |
| C1 oracle | 185 (15/1) | 184 (21/1) |
| H_gen | — | 16 vs 0, p = 1.526e-5, rejected |
| H_mem | 10 vs 1, p = 0.0059 | 7 vs 0, p = 0.0078; IU p = 0.0078, rejected |
| Calibration n₊ / harm / p | 348 / 1 / 3.418e-7 | 337 / 0 / 3.111e-8 |
| Calibration CP upper (post hoc) | 1.356% | 0.885% |
| Calibration success (descriptive) | 348 → 371 | 337 → 366 |
| Final Test C0 → V2 | 350 → 379 | 347 → 373 |
| Final Test rescue / harm / p | 29 / 0 / 1.863e-9 | 27 / 1 / 1.080e-7 |
| Final Test bootstrap 95% CI (post hoc) | [+4.75, +10.00] pp | [+4.00, +9.00] pp |
| B3 PlainConv C0 → V2 | 52 → 87 (37/2), p = 1.421e-9 | — |

The manuscripts are checked against `recount.json` by `figures/source_code/check_numbers.py` (31 checks, 0 failures;
log `build_logs/check_numbers.json`).
