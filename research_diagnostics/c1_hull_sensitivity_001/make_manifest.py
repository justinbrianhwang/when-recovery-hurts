"""analysis_manifest.json and SHA256_MANIFEST.json of the S4/S5 hull sensitivity, with the immutability checks:
the C1 run folder against the per-file sha256 list written on the server, the raw archive, the frozen plan, the
protected code against git HEAD, and every scene record / trace used (sha256 checked at load against the record).
The manifest log is written outside the hashed folder."""
from __future__ import annotations

import json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hull_s45 as H  # noqa: E402
C = H.C
CODE = ['hull_s45.py', 'run_all.py', 'aggregate.py', 'check_equivalence.py', 'failure_details.py', 'test_hull_s45.py', 'make_manifest.py']
COMMANDS = [
    'python -P research_diagnostics/c1_hull_sensitivity_001/run_all.py --work runs/c1_hull_sensitivity_001/work --workers 4   (exit 0, 400/400, 493 s)',
    'python -P research_diagnostics/c1_hull_sensitivity_001/aggregate.py --work runs/c1_hull_sensitivity_001/work --out results/vision_center_v2_c1_001/posthoc_hull_sensitivity_001   (exit 0)',
    'python -P research_diagnostics/c1_hull_sensitivity_001/check_equivalence.py --out .../LEAN_VS_FULL_EQUIVALENCE.json   (exit 0, all identical)',
    'python -P research_diagnostics/c1_hull_sensitivity_001/failure_details.py --branches .../S45_BRANCH_RESULTS.csv --out .../S45_FAILURE_DETAILS.json   (exit 0)',
    'python -P -m pytest -v -p no:cacheprovider research_diagnostics/c1_hull_sensitivity_001/test_hull_s45.py   (exit 0, 4 passed)',
    'python -P research_diagnostics/c1_hull_sensitivity_001/make_manifest.py']
SUPERSEDED = ['try 1 (runs/c1_hull_sensitivity_001/superseded_try1/, not used): the mesh statistic logged_triangles_not_hull_faces counted a different diagonal of a planar hull face as off-hull '
              '(a new test failed); fixed to "not on any hull facet plane", then the whole run, aggregation and equivalence check were repeated in a new work folder. S4/S5 values of try 1 were identical.',
              'failure_details try 1: the phase label of trunk rows used the branch trigger (None for the trunk); fixed and re-run; numbers unchanged.']


def main():
    out = H.OUT; ext = C.RUN.parents[1]
    server = {}
    for line in (C.REPO / 'runs' / 'vision_center_v2_c1_001_server' / 'vision_center_v2_c1_out.sha256').read_text(encoding='utf-8').splitlines():
        if line.strip(): h, p = line.split(None, 1); server[p.strip()] = h
    run_files = {p.relative_to(ext).as_posix(): p for p in sorted(C.RUN.rglob('*')) if p.is_file()}
    mism = [k for k, p in run_files.items() if server.get(k) != C.sha256(p)]
    arch = C.sha256(C.ARCHIVE)
    diff = subprocess.run(['git', 'diff', '--name-only', 'HEAD', '--', *C.PROTECTED_CODE], cwd=C.REPO, capture_output=True, text=True)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=C.REPO, capture_output=True, text=True).stdout.strip()
    prot_changed = [x for x in diff.stdout.split() if x]; plan_ok = C.sha256(C.PLAN) == C.FROZEN_PLAN_SHA256
    summ = json.loads((out / 'S45_HULL_SENSITIVITY_SUMMARY.json').read_text(encoding='utf-8')); eq = json.loads((out / 'LEAN_VS_FULL_EQUIVALENCE.json').read_text(encoding='utf-8'))
    import numpy, scipy, matplotlib
    status = 'COMPLETE' if (not mism and arch == C.ARCHIVE_SHA256 and diff.returncode == 0 and not prot_changed and plan_ok and summ['complete'] and eq['all_identical']) else 'INCOMPLETE'
    outputs = {p.relative_to(out).as_posix(): C.sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('analysis_manifest.json', 'SHA256_MANIFEST.json')}
    man = {'what': 'C1 S4/S5 sensitivity to the convex-mesh triangulation: logged PhysX meshes vs the convex hull of the same vertices; stored records only; no simulation, rollout, training, GPU or paid compute',
           'label': C.LABEL, 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'status': status,
           'frozen_plan': {'path': C.rel(C.PLAN), 'sha256': C.sha256(C.PLAN), 'equal': plan_ok},
           'raw_archive': {'path': C.rel(C.ARCHIVE), 'sha256_now': arch, 'equal': arch == C.ARCHIVE_SHA256},
           'run_folder_vs_server_sha256_list': {'files': len(run_files), 'mismatch_or_unlisted': mism, 'unchanged_since_packaging': not mism},
           'protected_files': {'git_head': head, 'changed_vs_HEAD': prot_changed, 'unchanged': diff.returncode == 0 and not prot_changed, 'sha256': C.protected_hashes()},
           'inputs': 'every scene record and trace of both policies (400 scene-runs); each trace checked against the sha256 in its record at load; record and trace sha256 per branch in S45_BRANCH_RESULTS via the work files (runs/c1_hull_sensitivity_001/work, not in Git)',
           'recorded_case_inputs': 'DRIVE:development_run_001/figures/case_figures_001/*/*/plotted.json (sha256 per file in RECORDED_CASES_REPRODUCTION.csv)',
           'code_sha256': {f'research_diagnostics/c1_hull_sensitivity_001/{f}': C.sha256(HERE / f) for f in CODE},
           'code_reused_unchanged': {p: C.sha256(C.REPO / p) for p in ('src/hci/spatial_links.py', 'src/hci/spatial_geometry.py', 'research_diagnostics/c1_physics_001/common.py')},
           'environment': {'python': platform.python_version(), 'platform': platform.platform(terse=True), 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
           'commands': COMMANDS, 'superseded': SUPERSEDED,
           'results_unchanged': {'S6_official': 'FAIL (both policies)', 'Check4_V2': '9/41 and 5/39 FAIL', 'V2_rescue_harm_vs_C0': '14/1 (train_seed_1), 16/0 (train_seed_2)', 'plan': C.FROZEN_PLAN_SHA256},
           'outputs': outputs}
    C.jdump(out / 'analysis_manifest.json', man)
    allf = {p.relative_to(out).as_posix(): C.sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256_MANIFEST.json'}
    C.jdump(out / 'SHA256_MANIFEST.json', {'what': 'sha256 of every file of posthoc_hull_sensitivity_001 (except this file)', 'files': allf})
    print(json.dumps({'status': status, 'run_folder_mismatch': len(mism), 'archive_equal': arch == C.ARCHIVE_SHA256, 'plan_equal': plan_ok, 'protected_unchanged': not prot_changed, 'outputs': len(outputs)}))


if __name__ == '__main__':
    main()
