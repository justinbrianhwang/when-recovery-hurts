"""Pick one checkpoint among pre-declared candidates from their results on the SAME development seed set.

Rule (fixed before any result existed): highest success_once; ties go to the later training iteration.
The chosen candidate is then evaluated once on an untouched qualification set; this script never reads one.
"""
from pathlib import Path
import argparse, json
from hci.io import read_json, write_json


def choose(results):
    """results: list of dicts with iteration, success_once, n, seeds, set."""
    if not results: raise ValueError('No candidates')
    first = results[0]
    for r in results:
        if r['set'] != 'policy_select': raise ValueError(f"Selection must use policy_select, got {r['set']}")
        if r['seeds'] != first['seeds'] or r['n'] != len(first['seeds']): raise ValueError('Candidates were not evaluated on the same complete seed set')
    if len({r['iteration'] for r in results}) != len(results): raise ValueError('Duplicate candidate iteration')
    return max(results, key=lambda r: (r['success_once'], r['iteration']))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--candidate', action='append', required=True, metavar='ITERATION=RUN_DIR'); p.add_argument('--out', required=True)
    a = p.parse_args(); results = []
    for item in a.candidate:
        it, folder = item.split('=', 1); v = read_json(Path(folder) / 'policy_eval.json')
        results.append({'iteration': int(it), 'run': folder, 'policy': v['policy'], 'policy_sha256': v['policy_sha256'], 'set': v['set'], 'seeds': v['seeds'],
                        'n': v['n'], 'success_once': v['success_once'], 'wilson95': v['wilson95'], 'success_at_end': v.get('success_at_end')})
    best = choose(results)
    write_json(a.out, {'rule': 'highest success_once on policy_select; ties go to the later iteration (declared before evaluation)',
                       'candidates': [{k: v for k, v in r.items() if k != 'seeds'} for r in results], 'selected_iteration': best['iteration'],
                       'selected_policy': best['policy'], 'selected_policy_sha256': best['policy_sha256'],
                       'note': 'Selection uses development seeds; the selected policy must be qualified once on a separate set.'})
    print(json.dumps({'selected_iteration': best['iteration'], 'selected_policy': best['policy'],
                      'counts': {r['iteration']: f"{r['success_once']}/{r['n']}" for r in results}}))


if __name__ == '__main__':
    main()
