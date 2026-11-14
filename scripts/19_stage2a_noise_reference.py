"""Stage 2A, added AFTER the pilot results were seen: how often does Continue itself change with the future noise?

Reads the saved scene records only; nothing is rerun. For every snapshot the four Continue results (repeats 0..3)
are compared with each other in ordered pairs (r, r'), r != r'. This is a reference for reading the Rescue/Harm
counts: Refresh and Retreat-Retry also draw other noise than the Continue of their block.

Descriptive and post hoc. Pairs of one snapshot and snapshots of one scene are not independent; no interval is given.
"""
from pathlib import Path
import argparse, json
from hci.io import read_json, write_json


def reference(records):
    total = {'fail_to_success': 0, 'continue_fail_pairs': 0, 'success_to_fail': 0, 'continue_success_pairs': 0}
    by_t = {}; scenes = {'fail_to_success': set(), 'success_to_fail': set()}
    for rec in records:
        ys = {}
        for b in rec['blocks']: ys.setdefault(b['checkpoint'], {})[b['repeat']] = b['y_continue']
        for t, d in ys.items():
            row = by_t.setdefault(t, {k: 0 for k in total})
            for a in d:
                for b in d:
                    if a == b: continue
                    if d[a] == 0:
                        for x in (total, row): x['continue_fail_pairs'] += 1; x['fail_to_success'] += d[b]
                        if d[b]: scenes['fail_to_success'].add(rec['scene'])
                    else:
                        for x in (total, row): x['continue_success_pairs'] += 1; x['success_to_fail'] += 1 - d[b]
                        if not d[b]: scenes['success_to_fail'].add(rec['scene'])

    def rate(x):
        return {**x, 'fail_to_success_rate': x['fail_to_success'] / x['continue_fail_pairs'] if x['continue_fail_pairs'] else 'NA',
                'success_to_fail_rate': x['success_to_fail'] / x['continue_success_pairs'] if x['continue_success_pairs'] else 'NA'}
    return {'all_checkpoints': rate(total), **{f't{t}': rate(v) for t, v in sorted(by_t.items())},
            'scenes_with_fail_to_success': len(scenes['fail_to_success']), 'scenes_with_success_to_fail': len(scenes['success_to_fail']),
            'unit': 'ordered pair of two different repeats of Continue at the same snapshot (12 pairs per snapshot)',
            'scope': 'Post hoc reference computed after the results were seen. Not a planned comparison, not a test. '
                     'Pairs within a snapshot and snapshots within a scene are correlated.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--scenes', required=True); p.add_argument('--out', required=True); a = p.parse_args()
    out = reference([read_json(f) for f in sorted(Path(a.scenes).glob('[0-9]*.json')) if not f.name.endswith('.error.json')])
    write_json(a.out, out); print(json.dumps(out, indent=2))
