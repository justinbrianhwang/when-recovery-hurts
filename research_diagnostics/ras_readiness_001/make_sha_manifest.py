"""SHA256_MANIFEST.json of results/ras_submission_readiness_001 (every file except itself) plus the new code files and the
immutability checks of the protected C1 inputs (plan, archive, protected code vs git HEAD).

usage: python -P research_diagnostics/ras_readiness_001/make_sha_manifest.py
"""
import json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research_diagnostics' / 'c1_physics_001'))
import common as C  # noqa: E402

OUT = REPO / 'results' / 'ras_submission_readiness_001'
CODE = ['src/hci/v2_confirm.py', 'scripts/40_v2_confirm.py', 'scripts/41_gen_arch_plainconv.py', 'tests/test_v2_confirm.py', 'tests/test_gen_arch_plainconv.py',
        'research_diagnostics/s6_static_validation_001/static_two_box.py', 'research_diagnostics/ras_readiness_001/split_usage_audit.py', 'research_diagnostics/ras_readiness_001/power_tables.py',
        'research_diagnostics/ras_readiness_001/make_freeze_draft.py', 'research_diagnostics/ras_readiness_001/make_sha_manifest.py', 'paper_tools/make_figure_atlas.py']


def main():
    diff = subprocess.run(['git', 'diff', '--name-only', 'HEAD', '--', *C.PROTECTED_CODE], cwd=REPO, capture_output=True, text=True)
    checks = {'frozen_c1_plan_equal': C.sha256(C.PLAN) == C.FROZEN_PLAN_SHA256, 'c1_archive_equal': C.sha256(C.ARCHIVE) == C.ARCHIVE_SHA256,
              'protected_code_changed_vs_HEAD': [x for x in diff.stdout.split() if x], 'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()}
    files = {p.relative_to(OUT).as_posix(): C.sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'SHA256_MANIFEST.json'}
    man = {'what': 'sha256 of every file of results/ras_submission_readiness_001 (except this file) and of the new code', 'files': files, 'code': {c: C.sha256(REPO / c) for c in CODE}, 'immutability': checks}
    (OUT / 'SHA256_MANIFEST.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    print(json.dumps({'files': len(files), **{k: v for k, v in checks.items() if k != 'git_head'}}))


if __name__ == '__main__':
    main()
