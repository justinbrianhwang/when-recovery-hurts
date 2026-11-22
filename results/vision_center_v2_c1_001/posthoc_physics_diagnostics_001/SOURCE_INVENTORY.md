# Source inventory (every input read by the post-hoc physics diagnostics)

All inputs were read-only. The full per-file sha256 lists and the immutability checks are in `analysis_manifest.json`.

| Input | Path (repository-relative or Drive development-run folder) | sha256 / check | Coverage |
|---|---|---|---|
| Frozen C1 plan | `results/vision_center_v2_c1_001/preparation_v3/plan.json` | `6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7` (checked) | plan of the run |
| Raw archive of the run | `runs/vision_center_v2_c1_001_server/vision_center_v2_c1_out.tgz` (4,203,350,820 bytes; not in Git) | `c3960c8db0b0dcfa7a9da56cf602056c57c691bcbe06bea22151e359782b60f2` (checked) | everything the server wrote |
| Extracted run folder | `runs/vision_center_v2_c1_001_server/extracted/runs/vision_center_v2_c1_001/` | every file against the per-file sha256 list written on the server at packaging (`vision_center_v2_c1_out.sha256`) | 2 policies × 200 scenes (51100–51299): 400 scene records, 400 traces (`pass_<policy>/traces/trace_<scene>.json.gz`, ~60 MB per policy), 400 frame files |
| Scene records | `.../pass_<policy>/scenes/<scene>.json` | each record's sha256 is written into every derived row (`record_sha256`) | trigger, cells (9 arms), control steps, collision geometry (initial_state) |
| Traces | `.../pass_<policy>/traces/trace_<scene>.json.gz` | checked against the `trace_sha256` stored in its record at every load (`common.traces`) | per arm and the trunk: TCP, cube poses and velocities, link poses, joint qpos/qvel, the PhysX contact list of the last sub-step with points (position, normal, impulse, separation) |
| S6 pair rows | `DRIVE:development_run_001/mechanism_c1_001/tables/s6_pair_rows.csv.gz` | `30c6bb322a3d33388b667a48f02ea4ab86a1bf7922d662df1f6319eb7a2f98a7` = C1 analysis manifest and Drive manifest | 580,539 rows (293,614 + 286,925) |
| S6 official summary | `DRIVE:.../mechanism_c1_001/tables/s6_validity_summary.json` | Drive manifest | FAIL in both policies |
| Mechanism report / manifest | `DRIVE:.../mechanism_c1_001/C1_MECHANISM_REPORT.md`, `analysis_manifest.json` | Drive manifest | context |
| Trajectory audit checks | `DRIVE:.../trajectory_audit_001/per_scene_checks.csv` | `5765e911458a383ce34cd5bacfd51713e28bd4e0f20306b89a8f18eb85ecadde` = Drive manifest | every check of every arm of every triggered scene; decides which moves failed Check 4 |
| Trajectory audit summary | `DRIVE:.../trajectory_audit_001/audit_summary.json` | Drive manifest | case lists |
| C1 execution report | `DRIVE:.../C1_DEVELOPMENT_EXECUTION_REPORT.md` | Drive manifest | context |
| Code read (unchanged) | `src/hci/spatial_probe.py`, `spatial_links.py`, `spatial_geometry.py`, `trajectory_audit.py`, `recovery_v1_oracle.py`, `vision_center_v2.py`, `simulation.py`, `configs/stackcube.yaml`, `scripts/35_c1_trajectory_audit.py`, `mechanism_analysis/s6_validity.py` | sha256 and `git diff HEAD` in the manifest | definitions of S6 and Check 4 |
| Upstream source (read-only, public) | ManiSkill `a4a4f9272ad64b1564035874b605ceb687b63ed8`: `utils/structs/types.py`, `agents/robots/panda/panda.py`, `agents/controllers/pd_ee_pose.py`, `pd_joint_pos.py`, `base_controller.py`, `envs/sapien_env.py` | pinned commit | SceneConfig defaults (contact_offset 0.02, rest_offset 0, PCM, TGS, 15 position iterations); 100 Hz sim / 20 Hz control; pd_ee_delta_pos semantics |

Notes
- The Drive folder is the durable copy of the C1 analysis outputs (`results/vision_center_v2_c1_001/development_run_001/`
  in the project Drive); a scratch copy with identical hashes was used during the C1 analysis.
- Two policies share the same 200 scenes; nothing here treats them as 400 independent scenes (scene-level bootstrap keeps
  both policies of a scene together; models hold out a scene for all arms and both policies).
- Missing data: none. Not logged by the run (stated, not worked around): sub-step states and contacts of the first four
  sub-steps, PhysX contact/rest offsets, PCM manifold state, controller IK and drive targets.
