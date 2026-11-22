# Questions the stored records cannot answer, and the minimum experiments that could (NOT RUN; not approved)

Nothing below was executed. Each item needs a simulator, and some need paid GPU time; none is authorised by this task.
They are listed so that a later decision can be made; they are not a plan.

| Open question (from the diagnosis) | Why the records cannot answer it | Minimal design | Safety / validity limits |
|---|---|---|---|
| Q1 Is the contact list generated before the last sub-step's position update (time offset, H-S6-1)? | Only the end-of-step pose and the last sub-step's contacts are logged | Replay a few stored C1 trunks (same actions) on the local CPU backend with a probe that logs poses and `get_contacts()` after **every** sub-step (step loop wrapped, not changed); compare each sub-step's separations with geometry at that sub-step's start and end poses | A replay is not the original run (CPU replays diverged slightly before); logging must be shown not to change states (ON/OFF bitwise comparison as in C1 probe-check) |
| Q2 Why do slow finger–cube pairs (finger box #2) and camera–cube pairs disagree by 1–6 mm (H-S6-3/4)? | PCM manifold contents, refresh rule and the per-shape contact/rest offsets are not logged | Same replay, logging per shape `contact_offset`, `rest_offset`, and per point the PhysX feature indices if the API exposes them; a static two-box scene with a 30°-rolled box at known gaps, separations read from `get_contacts()` vs closed-form distances | Static scenes test the definition only, not the C1 dynamics; SAPIEN may not expose manifold internals (then UNKNOWN stays) |
| Q3 Is the Check 4 first-step deviation a joint-drive transient (H-C4-3)? | IK targets, drive targets, joint torques and sub-step states are not logged | Replay the 14 V2 failures and matched passes from the stored trunk with logging of the controller's `_target_qpos`, joint qpos/qvel per sub-step and the TCP per sub-step | Replay divergence; does not test other controllers; must not change the controller or its gains |
| Q4 Does a non-convex triangulation matter for other audits (S4 clearances with arm links)? | Can be answered offline | Re-run the existing S4/S5 distances with the convex hull of the logged vertices (reference path) and compare; offline, CPU | Changes no existing result; would be a new post-hoc analysis needing its own approval |
| Q5 Continuous-time collision freedom between logged rows | Sub-step states are not observed; observing sub-steps still does not give continuous-time guarantees | Sub-step logging gives 5 samples per control step, not a guarantee; a bound would need verified velocity and geometry bounds | Do not use an unverified bound (e.g. "distance > d0 + d1") as a guarantee |

Estimated scale if ever approved: Q1–Q3 a few dozen scene replays on CPU (no GPU needed for physics-only replays without
rendering); Q4 offline CPU only. Any of these would need a new plan and explicit approval; Calibration and Final Test
remain untouched.
