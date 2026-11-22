# B1 Calibration preflight — written before the first calibration rollout

Date: 2026-11-22 (UTC). Host: Vast instance 62040825, RTX 4090, NVIDIA driver 595.99.02, Intel i7-10700.

## Frozen manifest

`frozen_calibration/manifest.json` (sha256 `01b70a065f82bcd80ccc031061e73bfdd3f86a302123387fbc578a4bbad3ef5e`), written on the
server by `scripts/40_v2_confirm.py freeze --stage calibration`. 28 files. Every file equals the committed local file,
except the host `pip_freeze.txt`, whose copy here equals the frozen hash. Upstream ManiSkill commit `a4a4f927…`.
Family: (VISION_CENTER_V2, train_seed_1), (VISION_CENTER_V2, train_seed_2). ε = δ = 0.05.

## Real-simulator smoke check of the runner (used C1 development scenes only; no calibration or test seed)

Scenes 51103, 51157, 51253 × both policies, arms C0 and V2, separate output and ledger folders.

1. **Against the original C1 records: NOT identical** (`smoke_compare.json`). The trunk traces already differ. Triggers and
   some outcomes differ too: 51157 C0 succeeded in C1 and failed here with both policies.
   - The pre-declared criterion in `CALIBRATION_TEST_DESIGN.md` §5 step 3 ("identical to the C1 records") is therefore not
     met.
2. **Against the unchanged C1 collector on the same host: identical** (`runner_vs_samehost.json`). The B4 job ran
   `collect_scene` with all nine C1 arms on this host, and the runner's trunk, C0 and V2 traces and outcomes for 51157 equal
   it for both policies.
3. B4 (11 used scene-runs, same host): the collector on this host also differs from the original C1 traces in 11 of 11
   scene-runs, with different outcomes in 6 of them. These scenes were selected as S6 extreme cases and are not a random
   sample. Logging ON vs OFF was identical in 11 of 11.

**Reading.** The runner reproduces the C1 collector exactly. The difference from the original C1 run is a host effect:
C1 ran on driver 550.120 (i7-13700K); this host runs 595.99.02. A different driver can change the rendered images, and
with them the policy's actions. Phase B (driver 595.99.02) was reproduced exactly on 595.99.02 (rgb_replay_001).

## Decision recorded before calibration

- Calibration validity rests on Y0 and Yg coming from one snapshot in one session on one host. That holds, and is enforced
  by the collector.
- **Host type for Calibration and the Final Test is fixed to RTX 4090, driver 595.99.02.** The certificate refers to
  this simulator and rendering configuration.
- The C1 development results (driver 550.120) stay as they are. Development, Calibration and Test therefore come from two
  host configurations, and the paper must state this. Outcome differences between hosts were 16/300 in Phase B and 6/11
  in the selected B4 scenes.
- Nothing about the candidate, policies, ε, δ, tests or rules changed.
