"""FREEZE_MANIFEST_DRAFT.json: what a calibration freeze would contain today, with sha256. NOT a freeze: no amendment
approval exists, no host pip freeze exists, no calibration was run. scripts/40_v2_confirm.py freeze writes the real one.

usage: python -P research_diagnostics/ras_readiness_001/make_freeze_draft.py --out <json>
"""
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from hci import v2_confirm as V  # noqa: E402


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a = a.parse_args(); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    spec = importlib.util.spec_from_file_location('x40', REPO / 'scripts' / '40_v2_confirm.py'); X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
    R = 'results/ras_submission_readiness_001/'
    files = [*X.CODE, R + 'AMENDMENT_017_FINAL_PROPOSAL.md', R + 'calibration_power_table.json', *(f'{X.POLICY_PATHS[p]}/{f}' for p in V.POLICY_IDS for f in ('weights.pt', 'policy_meta.json'))]
    import os; os.chdir(REPO)
    man = V.manifest(files, {'status': 'DRAFT — NOT A FREEZE', 'stage': 'calibration', 'candidate': V.CANDIDATE, 'arms': list(V.ARMS), 'family': V.family(V.POLICY_IDS),
                             'epsilon': V.EPSILON, 'delta': V.DELTA, 'alpha_efficacy': V.ALPHA_EFFICACY, 'splits': {k: list(v) for k, v in V.SPLITS.items()},
                             'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                             'missing_for_a_real_freeze': ['user approval record of Amendment 017 (approval_amendment_017.json naming the final proposal by sha256)', 'full pip freeze of the calibration host',
                                                           'upstream commit of the installed third_party/ManiSkill (expected a4a4f9272ad64b1564035874b605ceb687b63ed8)', 'runner smoke on the real simulator (used development scenes)',
                                                           'commit, push and Drive copy of the real manifest before the first calibration rollout'],
                             'previous_draft': 'Drive results/vision_center_v2_001/V2_FREEZE_MANIFEST_DRAFT.json (Phase C0, sha256 6a2d90c7...)'})
    out.write_text(json.dumps(man, indent=1), encoding='utf-8'); print(len(man['files_sha256']), 'files')


if __name__ == '__main__':
    main()
