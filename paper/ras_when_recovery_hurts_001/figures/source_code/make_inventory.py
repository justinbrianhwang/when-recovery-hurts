"""Derived documents: FIGURE_PROVENANCE.csv, FIGURE_CAPTIONS.md and SOURCE_AND_HASH_INVENTORY.json.

usage: python -P paper/ras_when_recovery_hurts_001/figures/source_code/make_inventory.py
"""
from __future__ import annotations

import csv, hashlib, json, re
from pathlib import Path

P = Path(__file__).resolve().parents[2]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows = []
    for prov in ('FIGURE_PROVENANCE.json', 'FIGURE_PROVENANCE_SUPP.json'):
        d = json.loads((P / 'figures' / prov).read_text(encoding='utf-8'))
        for name, o in d['outputs'].items():
            assert sha(P / 'figures' / name) == o['sha256'], f'{name} changed after its provenance was written'
            rows.append({'figure': name, 'sha256': o['sha256'], 'generator': d['generator'], 'generator_sha256': d['generator_sha256'],
                         'inputs': len(d['inputs_sha256']), 'inputs_list': ' | '.join(sorted({re.sub(r'/(scenes|traces|frames)/[^/]+$', r'/\1/*', k) for k in d['inputs_sha256']}))})
    with open(P / 'FIGURE_PROVENANCE.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    caps = []
    for doc in ('main', 'supplementary'):
        s = (P / 'manuscripts' / f'{doc}.tex').read_text(encoding='utf-8')
        for env in re.finditer(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', s, flags=re.S):
            body = env.group(0); m = re.search(r'\\caption\{', body)
            if not m: continue
            i, depth = m.end(), 1
            while depth:
                depth += {'{': 1, '}': -1}.get(body[i], 0); i += 1
            cap = body[m.end():i - 1]; lab = re.search(r'\\label\{([^}]+)\}', body); inc = re.search(r'\\(?:includegraphics|input)(?:\[[^\]]*\])?\{([^}]+)\}', body)
            caps.append(f"### {doc} — {env.group(1)} `{lab.group(1) if lab else '?'}` ({inc.group(1).replace(chr(92) + 'tabdir ', '') if inc else 'inline'})\n\n{cap.strip()}\n")
    (P / 'FIGURE_CAPTIONS.md').write_text('# Figure and table captions (extracted from the LaTeX sources)\n\n' + '\n'.join(caps), encoding='utf-8')
    inv = {}
    for sub in ('manuscripts', 'figures', 'tables', 'data', 'submission', 'editorial'):
        for f in sorted((P / sub).rglob('*')):
            if f.is_file() and 'RAS_source_flat' not in f.parts: inv[str(f.relative_to(P)).replace(chr(92), '/')] = sha(f)
    for f in sorted(P.glob('*')):
        if f.is_file() and f.name != 'SOURCE_AND_HASH_INVENTORY.json': inv[f.name] = sha(f)
    upstream = json.loads((P / 'figures/FIGURE_PROVENANCE.json').read_text(encoding='utf-8'))['inputs_sha256']
    upstream.update(json.loads((P / 'figures/FIGURE_PROVENANCE_SUPP.json').read_text(encoding='utf-8'))['inputs_sha256'])
    upstream.update(json.loads((P / 'tables/TABLE_PROVENANCE.json').read_text(encoding='utf-8'))['inputs_sha256'])
    (P / 'SOURCE_AND_HASH_INVENTORY.json').write_text(json.dumps({'paper_files_sha256': inv, 'upstream_inputs_sha256': dict(sorted(upstream.items())),
                                                                 'frozen_artifacts': {'C1_plan': '6b3f04fe...', 'amendment_017': 'b8fd6324...', 'calibration_manifest': '01b70a065f82bcd80ccc031061e73bfdd3f86a302123387fbc578a4bbad3ef5e',
                                                                                      'final_test_manifest': '94c5a591554efa99f821dd2d7ea1099619afec790ee92f995e95ee8a92161c5f'}}, indent=1), encoding='utf-8')
    print(len(rows), 'figures;', len(caps), 'captions;', len(inv), 'paper files;', len(upstream), 'upstream inputs')


if __name__ == '__main__':
    main()
