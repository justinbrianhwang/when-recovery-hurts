# Check 4 case notes (POST_HOC_EXPLORATORY)

The 14 V2 moves that failed Check 4 in `trajectory_audit_001/per_scene_checks.csv`, recomputed from the raw traces
(record and trace sha256 per row in `CHECK4_FAILURE_CASES.csv`). Every command is 5.0 mm; "before" = TCP xy step from
row t0−1 to t0 (the policy's own motion just before the trigger). Failing step: 1 in all 14 (cosine only).

| Policy | Scene | Steps | Step-1 cos | Step-1 TCP (mm) | Before (mm) | cos(cmd1, before) | Steps 2+ min cos | |Σcmd| / |ΣTCP| (mm) | True offset before → after (mm) | vs C0 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 1 | 51185 | 5 | 0.663 | 1.37 | 2.38 | −0.32 | 0.996 | 25.0 / 8.2 | 18.7 → 10.5 | both succeed |
| seed 1 | 51188 | 4 | 0.631 | 2.55 | 4.19 | −0.03 | 0.934 | 20.0 / 7.3 | 16.1 → 9.3 | both succeed |
| seed 1 | 51191 | 6 | 0.780 | 0.82 | 0.62 | −1.00 | 0.993 | 30.0 / 9.9 | 20.2 → 10.3 | both succeed |
| seed 1 | 51198 | 11 | 0.759 | 1.03 | 0.18 | −0.15 | 0.998 | 55.0 / 19.3 | 21.6 → 2.3 | rescue |
| seed 1 | 51199 | 12 | 0.879 | 0.80 | 0.33 | −0.84 | 0.993 | 60.0 / 22.0 | 31.3 → 9.5 | rescue |
| seed 1 | 51250 | 12 | 0.752 | 1.39 | 2.03 | −0.59 | 0.995 | 60.0 / 21.1 | 25.6 → 4.4 | rescue |
| seed 1 | 51264 | 10 | 0.823 | 1.11 | 0.59 | −0.84 | 0.993 | 50.0 / 17.4 | 20.6 → 4.0 | rescue |
| seed 1 | 51273 | 5 | 0.749 | 0.87 | 1.36 | −0.91 | 0.995 | 25.0 / 8.4 | 17.0 → 8.7 | both succeed |
| seed 1 | 51279 | 4 | 0.801 | 1.34 | 1.38 | −0.43 | 0.991 | 20.0 / 6.3 | 15.6 → 9.5 | both succeed |
| seed 2 | 51199 | 11 | 0.663 | 0.75 | 0.78 | −0.70 | 0.991 | 55.0 / 19.7 | 22.9 → 3.8 | rescue |
| seed 2 | 51240 | 8 | 0.858 | 0.47 | 1.53 | −0.23 | 0.999 | 40.0 / 13.2 | 22.9 → 9.7 | both succeed |
| seed 2 | 51279 | 6 | 0.868 | 1.18 | 0.93 | −0.56 | 0.997 | 30.0 / 10.1 | 18.6 → 8.5 | both succeed |
| seed 2 | 51285 | 7 | 0.846 | 1.16 | 0.94 | −0.90 | 0.995 | 35.0 / 12.0 | 19.1 → 7.3 | rescue |
| seed 2 | 51288 | 2 | 0.880 | 1.05 | 1.15 | +0.49 | 0.999 | 10.0 / 2.6 | 14.3 → 11.7 | both succeed |

Notes
- 13 of 14 first commands point against (cos < 0) the TCP motion just before the trigger; the exception (seed 2 51288,
  cos +0.49) still failed, so the motion before the trigger is not a sufficient explanation.
- Every move ended within tolerance except seed 1 51199 (12-step budget). The true offset after the move is ≤ 11.7 mm in all 14.
- The vectors of step 1 (command, actual, motion before) are drawn in `figures/check4_first_step_vectors.png`.

Harm case (not a Check 4 failure), seed 1 scene 51157, V2: true offset 54.6 mm at the trigger; 12 commands of 5 mm, each
followed at cosine 1.00 with 1.94–2.03 mm of TCP motion (z drift −0.09 mm per step, no robot–target impulse); the held cube
moved 24.3 mm net and was still 30.4 mm from the target at the 12-step budget; the policy released 3 steps later and the
episode ended at the horizon without success while C0 succeeded at step 190. This is an observation of one case; it
does not establish whether the per-step progress, the step budget or the horizon caused the harm.
