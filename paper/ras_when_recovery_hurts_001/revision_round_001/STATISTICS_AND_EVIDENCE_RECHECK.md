# Statistics and evidence recheck — round 001

All values below were re-read from `data/recount.json` (two implementations, built from hash-checked raw records)
and `data/handback_stats.json` (new, raw records). `figures/source_code/check_numbers_latex.py` checks 36 manuscript
literals against these files: **36/36 pass** (`build_logs/check_numbers_latex.json`). No number in the manuscript
changed in this revision except where new descriptive quantities were added.

| Quantity | Expected (spec) | Recount | Match |
|---|---|---|---|
| C1 C0 | 171 / 164 | 171 / 164 | ✓ |
| C1 V2 (R/H) | 184 (14/1) / 180 (16/0) | same | ✓ |
| H_gen | 16 vs 0, p ≈ 1.5e-5 | 1.526e-5 | ✓ |
| H_mem | 10 vs 1, 7 vs 0, IU p ≈ 0.0078 | 0.0059, 0.0078 → 0.0078 | ✓ |
| Calibration n+, k, p | 348/1/3.4e-7; 337/0/3.1e-8 | 3.418e-7; 3.111e-8 | ✓ |
| Calibration success | 348→371; 337→366 | same | ✓ |
| Test | 350→379 (29/0, 1.862645e-9); 347→373 (27/1, 1.080334e-7) | 1.8626e-9; 1.0803e-7 | ✓ |
| Pooled 2×2 | 696/56/1/47 | same | ✓ |
| B3 | 52→87, 37/2, p ≈ 1.4e-9 | 1.4206e-9 | ✓ |
| S6 | 54,267 / 580,539 | same (S6_ERRORS_BY_GROUP) | ✓ |
| B4 | 5,181 → 1,874 of 56,558 | same | ✓ |
| S4 | 1 FAIL of 715 (oracle 51152 s2) | same | ✓ |
| S5 | 11 of 3,874; 2 events | same | ✓ |
| Check 4 | V2 9/41, 5/39; 110 first-step | same | ✓ |

`RECOUNT_DISCREPANCIES.md`: none found.

## Test assumptions as now written
- Unit: scene. Independence across scenes is an **assumption** about the seed-to-scene map (stated in §3, Supp S1).
- Certificate: H0 h ≥ ε; p = F_Bin(n+,ε)(k); Holm over 2 policies; guarantee = P(certify | h ≥ ε) ≤ δ (frequentist),
  conditional on the calibration host configuration.
- Efficacy: exact one-sided conditional binomial (McNemar exact) on discordant scenes; Holm over 2 policies.
- Post hoc (labelled): one-sided CP upper bounds (all successes; acting subset 1/48, 0/36, 0/38, 1/46 → 9.5 %, 8.0 %,
  7.6 %, 9.9 %); paired bootstrap percentile intervals (20,000, seed 20261123).
- Pooled 2×2: descriptive only.

## New descriptive evidence added (raw records, no simulation)
- Hand-back: every acted V2 branch re-infers at the first step after the move; first open command 1–24 steps later
  (median 2–3); one calibration branch never opens (Table S4).
- Executed motion: net/commanded path dev 0.26–0.40 (medians 0.36, 0.37); no move ended at the cap.
- Large offsets (>40 mm estimated): dev 3 (1 rescue, 1 harm), cal 0, test 8 (6 rescues, 1 harm).
- 51152: target grasped at trigger; all vision arms abstained (NO_HOLD_ESTIMATE).
- Independent reconstruction: hand–cube 0.22–0.24 mm; hull swap hand ≤0.24 mm, arm links ≤0.03 mm, fingers 0.

## Evidence scope
- Re-verified from raw records here: C1, Calibration, Test, B3 counts; hand-back; decisions.
- Reported, not independently re-verified in this round: S4/S5/S6 tables, B4, static cases, Check 4 (read from the
  frozen post hoc result files; their own hashes are in `SHA256_MANIFEST.json` of those folders).
