"""Seed-level usage audit of the Calibration (30000-30399) and Final Test (90000-90399) splits. Read-only.

Scans every JSON / JSON.GZ / CSV / JSONL file under runs/, results/, artifacts/ (including ledgers), frozen*/ and the
Drive project folder (if given) for (a) integer values under seed-like keys ('scene', 'seed', 'episode', 'episode_seed',
'scenes', 'seeds' lists) and (b) file names whose stem is such a seed; also lists frozen folders, TEST_STARTED markers
and ledgers. Nothing is reset, rendered or rolled out. A configuration file that DECLARES the splits (configs/splits.json)
is reported separately as a declaration, not a use.

usage: python -P research_diagnostics/ras_readiness_001/split_usage_audit.py --out <json> [--drive <folder>]
"""
from __future__ import annotations

import argparse, csv, gzip, io, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAL = set(range(30000, 30400)); TEST = set(range(90000, 90400))
KEYS = {'scene', 'seed', 'episode', 'episode_seed', 'scene_seed', 'scenes', 'seeds', 'episodes'}


def walk(x, hits, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in KEYS: grab(v, hits, f'{path}/{k}')
            walk(v, hits, f'{path}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x[:100000]): walk(v, hits, path)


def grab(v, hits, path):
    vals = v if isinstance(v, list) else [v]
    for z in vals:
        if isinstance(z, (int, float)) and not isinstance(z, bool) and float(z).is_integer():
            z = int(z)
            if z in CAL: hits.append(('calibration', z, path))
            if z in TEST: hits.append(('test', z, path))


def load(p):
    if p.suffix == '.gz':
        with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as f: return [json.load(f)]
    t = p.read_text(encoding='utf-8', errors='replace')
    if p.suffix == '.jsonl': return [json.loads(l) for l in t.splitlines() if l.strip()]
    if p.suffix == '.csv':
        rows = list(csv.DictReader(io.StringIO(t))); return [{k: (int(v) if v and re.fullmatch(r'-?\d+', v) else v) for k, v in r.items()} for r in rows]
    return [json.loads(t)]


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a.add_argument('--drive'); a = a.parse_args(); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    roots = [REPO / 'runs', REPO / 'results', REPO / 'artifacts', *REPO.glob('frozen*')] + ([Path(a.drive)] if a.drive else [])
    scanned = 0; failed = []; uses = []; names = []; declarations = []
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob('*'):
            if not p.is_file(): continue
            stem = p.name.split('.')[0]; stem_n = re.sub(r'^(trace_|frames_)', '', stem)
            if stem_n.isdigit() and (int(stem_n) in CAL or int(stem_n) in TEST): names.append(str(p))
            if p.suffix not in ('.json', '.gz', '.jsonl', '.csv') or (p.suffix == '.gz' and not p.name.endswith('.json.gz')): continue
            if p.stat().st_size > 400_000_000: continue
            try:
                for doc in load(p):
                    hits = []; walk(doc, hits)
                    for split, z, path in hits: uses.append({'file': str(p.relative_to(root.parent) if root.parent in p.parents else p.name), 'split': split, 'value': z, 'key_path': path})
                scanned += 1
            except Exception as e:
                failed.append({'file': p.name, 'error': type(e).__name__})
    sp = json.loads((REPO / 'configs' / 'splits.json').read_text(encoding='utf-8'))['splits']
    declarations.append({'file': 'configs/splits.json', 'calibration': [min(sp['calibration']), max(sp['calibration']), len(sp['calibration'])], 'test': [min(sp['test']), max(sp['test']), len(sp['test'])]})
    ledgers = sorted(p.name for p in (REPO / 'artifacts' / 'ledger').glob('*')) if (REPO / 'artifacts' / 'ledger').exists() else []
    markers = [str(p) for r in roots if r.exists() for p in r.rglob('TEST_STARTED*')]
    frozen = [str(p) for p in REPO.glob('frozen*')]
    by_file = {}
    for u in uses: by_file.setdefault(u['file'], set()).add(u['split'])
    res = {'scope': 'seed-level usage of 30000-30399 and 90000-90399; no environment reset', 'files_scanned': scanned, 'files_unreadable': failed[:50], 'n_unreadable': len(failed),
           'uses_under_seed_keys': uses[:200], 'n_uses': len(uses), 'files_with_uses': {k: sorted(v) for k, v in by_file.items()}, 'files_named_like_a_split_seed': names,
           'declarations': declarations, 'ledgers': ledgers, 'test_started_markers': markers, 'frozen_folders': frozen,
           'verdict': 'UNUSED' if not uses and not names and not markers and not frozen else 'REVIEW'}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('files_scanned', 'n_unreadable', 'n_uses', 'verdict', 'ledgers', 'frozen_folders', 'test_started_markers')}, indent=1)); print(json.dumps(res['files_with_uses'], indent=1)[:3000])


if __name__ == '__main__':
    main()
