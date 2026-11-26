# Related-work and bibliography verification — round 001 (2026-11-25)

Checked by a web-verification agent against Crossref, doi.org, PMLR, RSS proceedings and arXiv; no file was written
by the agent. Status per entry in `references.bib` (`verification` field).

| Key | Status | Change |
|---|---|---|
| holm1979 | CORRECTED | DOI 10.2307/4615733 does not resolve (doi.org and Crossref 404). DOI removed; JSTOR stable URL 4615733 added; Scand. J. Statist. 6(2):65–70 (1979) confirmed by secondary sources |
| he2016resnet | VERIFIED (Crossref) | none; DOI 10.1109/CVPR.2016.90, pp. 770–778 |
| xiang2020sapien | CORRECTED (Crossref) | pages 11094–11104 (was 11097–11107); title capitalisation "Part-Based" |
| kelly2019hgdagger | VERIFIED (Crossref) | DOI 10.1109/ICRA.2019.8793698, pp. 8077–8083 |
| thananjeyan2021recoveryrl | VERIFIED (Crossref) | RA-L 6(3):4915–4922, DOI 10.1109/LRA.2021.3070252; title "With" |
| tao2024maniskill3 | CORRECTED | published at RSS 2025 as "Demonstrating GPU Parallelized Robot Simulation and Rendering for Generalizable Embodied AI with ManiSkill3", DOI 10.15607/RSS.2025.XXI.021; arXiv 2410.00425 kept as note |
| agia2025unpacking | VERIFIED | PMLR 270:689–723; STAC threshold calibrated on successful validation rollouts (conformal argument); VLM progress check zero-shot |
| romer2025fiper | VERIFIED (summary level) | OOD and action-entropy scores calibrated with conformal prediction on successful rollouts |
| ahmedt2026rescue | VERIFIED (abstract) | restores simulator and policy state at alarm checkpoints and compares interventions from the same state; no statistical harm guarantee stated in the abstract |
| other 19 entries | previously VERIFIED (round 0) | unchanged |

Unresolved: none of the 28 entries carries an unresolvable DOI any more; the Holm entry has no DOI by design.

## Claims about related work in the manuscript (checked)
- Sentinel and FIPER: calibration controls detector false alarms — consistent with the checks above.
- RESCUE: same-state counterfactual intervention evaluation — described as such; our differences are scope (one
  intervention end to end) and the calibration/test separation with a harm certificate and a motion/physics audit.
- No "first" claims remain in the manuscript.
