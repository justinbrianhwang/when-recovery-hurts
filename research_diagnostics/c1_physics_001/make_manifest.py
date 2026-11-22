"""Writes analysis_manifest.json and SHA256_MANIFEST.json for the post-hoc physics diagnostics.

Immutability checks (end of the work):
  - every file of the extracted C1 run against the per-file sha256 list written ON THE SERVER at packaging time
    (runs/vision_center_v2_c1_001_server/vision_center_v2_c1_out.sha256) -> the raw data are unchanged since the run;
  - the raw archive sha256;
  - the protected code / plan / config files against the committed HEAD (git diff) and their sha256;
  - the Drive copies of the analysis inputs against the Drive SHA256_MANIFEST.json of the development run.
"""
from __future__ import annotations

import json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

CODE = ['common.py', 'geometry_reference.py', 'diagnose_s6_offline.py', 's6_normal_ray_check.py', 's6_case_figures.py', 'diagnose_check4_offline.py',
        'run_geometry_validation.py', 'test_geometry_known_truth.py', 'test_diagnostics_loaders.py', 'make_manifest.py']
COMMANDS = [
    'python -P research_diagnostics/c1_physics_001/diagnose_s6_offline.py --workers 3',
    'python -P research_diagnostics/c1_physics_001/s6_normal_ray_check.py   (second run; the first run without the support-plane column is logged as logs/s6_normal_ray_check_try1_without_support_plane.log, its outputs were replaced)',
    'python -P research_diagnostics/c1_physics_001/s6_case_figures.py',
    'python -P research_diagnostics/c1_physics_001/diagnose_check4_offline.py',
    'python -P research_diagnostics/c1_physics_001/run_geometry_validation.py',
    'python -P -m pytest -v -p no:cacheprovider research_diagnostics/c1_physics_001/test_geometry_known_truth.py research_diagnostics/c1_physics_001/test_diagnostics_loaders.py',
    'python -P -m pytest -q -p no:cacheprovider tests/test_spatial_links.py tests/test_spatial_audit.py tests/test_trajectory_audit.py tests/test_spatial_probe.py',
    'python -P research_diagnostics/c1_physics_001/make_manifest.py']


def main():
    out = C.OUT; ext = C.RUN.parents[1]                                       # .../extracted
    server = {}
    for line in (C.REPO / 'runs' / 'vision_center_v2_c1_001_server' / 'vision_center_v2_c1_out.sha256').read_text(encoding='utf-8').splitlines():
        if line.strip(): h, p = line.split(None, 1); server[p.strip()] = h
    run_files = {p.relative_to(ext).as_posix(): p for p in sorted(C.RUN.rglob('*')) if p.is_file()}
    mism = [k for k, p in run_files.items() if server.get(k) != C.sha256(p)]
    run_check = {'files_in_run_folder': len(run_files), 'listed_on_server': sum(1 for k in run_files if k in server), 'sha256_mismatch_or_unlisted': mism,
                 'unchanged_since_packaging': not mism}
    arch = C.sha256(C.ARCHIVE)
    diff = subprocess.run(['git', 'diff', '--name-only', 'HEAD', '--', *C.PROTECTED_CODE], cwd=C.REPO, capture_output=True, text=True)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=C.REPO, capture_output=True, text=True).stdout.strip()
    prot = {'git_head': head, 'changed_vs_HEAD': [x for x in diff.stdout.split() if x], 'sha256': C.protected_hashes()}
    prot['unchanged'] = diff.returncode == 0 and not prot['changed_vs_HEAD']
    plan_ok = C.sha256(C.PLAN) == C.FROZEN_PLAN_SHA256
    dman = json.loads((C.DRIVE_RUN / 'SHA256_MANIFEST.json').read_text(encoding='utf-8'))['files']
    inputs = {}
    for rel_ in ('mechanism_c1_001/tables/s6_pair_rows.csv.gz', 'mechanism_c1_001/tables/s6_validity_summary.json', 'mechanism_c1_001/analysis_manifest.json',
                 'mechanism_c1_001/C1_MECHANISM_REPORT.md', 'trajectory_audit_001/per_scene_checks.csv', 'trajectory_audit_001/audit_summary.json', 'C1_DEVELOPMENT_EXECUTION_REPORT.md'):
        h = C.sha256(C.DRIVE_RUN / rel_); inputs['DRIVE:development_run_001/' + rel_] = {'sha256': h, 'equals_drive_manifest': dman.get(rel_) == h}
    code = {f'research_diagnostics/c1_physics_001/{f}': C.sha256(HERE / f) for f in CODE}
    import numpy, scipy, pandas, matplotlib
    outputs = {p.relative_to(out).as_posix(): C.sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('analysis_manifest.json', 'SHA256_MANIFEST.json')}
    status = 'COMPLETE' if (run_check['unchanged_since_packaging'] and arch == C.ARCHIVE_SHA256 and prot['unchanged'] and plan_ok and all(v['equals_drive_manifest'] for v in inputs.values())) else 'INCOMPLETE'
    man = {'what': 'C1 post-hoc physics diagnostics (S6 decomposition, Check 4 decomposition, independent geometry validation); stored records only; no simulation, rollout, inference, training, GPU or paid compute',
           'label': C.LABEL, 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'status': status,
           'frozen_plan': {'path': C.rel(C.PLAN), 'sha256_expected': C.FROZEN_PLAN_SHA256, 'sha256_now': C.sha256(C.PLAN), 'equal': plan_ok},
           'raw_archive': {'path': C.rel(C.ARCHIVE), 'sha256_expected': C.ARCHIVE_SHA256, 'sha256_now': arch, 'equal': arch == C.ARCHIVE_SHA256},
           'run_folder_vs_server_sha256_list': run_check, 'protected_files': prot, 'inputs_from_drive': inputs, 'code_sha256': code,
           'environment': {'python': platform.python_version(), 'platform': platform.platform(terse=True), 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'pandas': pandas.__version__, 'matplotlib': matplotlib.__version__},
           'upstream_source_read': 'ManiSkill a4a4f9272ad64b1564035874b605ceb687b63ed8 (raw.githubusercontent.com, read-only): utils/structs/types.py, agents/robots/panda/panda.py, agents/controllers/pd_ee_pose.py, pd_joint_pos.py, base_controller.py, envs/sapien_env.py, envs/tasks/tabletop/stack_cube.py, envs/scene.py',
           'commands': COMMANDS,
           'verification_scope': {'S6': 'official FAIL reproduced exactly; decomposition on all rows; raw recomputation on a rule-selected subset of 8,394 rows (400 scene-runs)',
                                  'Check4': 'all 715 moved branches of all arms; the 14 V2 failures reproduced three ways',
                                  'geometry': '19 closed-form cases + transform tests + cross-path checks; boxes verified, meshes cross-checked only'},
           'not_done': ['sub-step or controller-internal observation (needs a simulator; see FUTURE_DIAGNOSTIC_OPTIONS.md)', 'closed-form truth for convex-mesh penetration',
                        'EPA (NOT_IMPLEMENTED; Qhull Minkowski facets used instead)', 'S4/S5 re-analysis with hull geometry'],
           'results_unchanged': {'S6_official': 'FAIL (both policies)', 'Check4_V2': '9/41 and 5/39 FAIL', 'V2_rescue_harm_vs_C0': '14/1 (train_seed_1), 16/0 (train_seed_2)'},
           'outputs': outputs}
    C.jdump(out / 'analysis_manifest.json', man)
    allf = {p.relative_to(out).as_posix(): C.sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256_MANIFEST.json'}
    C.jdump(out / 'SHA256_MANIFEST.json', {'what': 'sha256 of every file of posthoc_physics_diagnostics_001 (except this file)', 'files': allf, 'code': code})
    print(json.dumps({'status': status, 'plan_equal': plan_ok, 'archive_equal': arch == C.ARCHIVE_SHA256, 'run_folder': {k: v for k, v in run_check.items() if k != 'sha256_mismatch_or_unlisted'},
                      'mismatch_n': len(mism), 'protected_unchanged': prot['unchanged'], 'inputs_equal': all(v['equals_drive_manifest'] for v in inputs.values()), 'outputs': len(outputs)}, indent=1))


if __name__ == '__main__':
    main()
