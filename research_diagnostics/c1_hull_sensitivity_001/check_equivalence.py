"""The lean S4/S5 path of hull_s45 against the full, unchanged hci.spatial_links path (branch -> spatial_items), on whole
real branches, for both geometries. Every S4/S5 field must be exactly equal (no tolerance).

usage: python -P research_diagnostics/c1_hull_sensitivity_001/check_equivalence.py --out <json>
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hull_s45 as H  # noqa: E402
C = H.C; L = H.L

# Fixed before looking at any result: the two harm scenes and the representative V2 rescue of seed 1 (C1 case index),
# with an acted and a non-acted branch each, plus the full trunk of one scene.
BRANCHES = [('train_seed_1', 51157, 'C0'), ('train_seed_1', 51157, 'VISION_CENTER_V2'), ('train_seed_2', 51253, 'C0'), ('train_seed_2', 51253, 'CENTER_ORACLE'),
            ('train_seed_1', 51103, 'VISION_CENTER_V2'), ('train_seed_1', 51103, 'trunk')]


def lean(rec, tr, arm, g):
    rows, src, t0, k, c = H.branch_rows(rec, tr, arm); acted = bool(c and c.get('acted') and k > 0); per = []
    for r in rows:
        W = L.world(r, g); m = {'t': int(r['t']), 's5': H.s5_pairs(r, W)}
        if acted and t0 is not None and t0 < r['t'] <= t0 + k: m['s4'] = H.s4_row(W)
        per.append(m)
    return H.items_from(per, t0, k, acted)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); a = p.parse_args(); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    res = []; ok_all = True
    for pol, sc, arm in BRANCHES:
        rec, _ = C.record(pol, sc); tr, _ = C.traces(pol, sc); g0 = rec['initial_state']['collision_geometry']
        for gname, g in (('LOGGED', g0), ('HULL', H.hull_geometry(g0))):
            t = time.time(); b = L.branch(rec, tr, arm, geometry=g); c = next((x for x in rec['cells'] if x['arm'] == arm), None)
            full = L.spatial_items(b, bool(c and c.get('acted') and b['control_steps'] > 0)); tf = time.time() - t
            t = time.time(); ln = lean(rec, tr, arm, g); tl = time.time() - t
            same = {k: json.dumps(full[k], sort_keys=True, default=float) == json.dumps(ln[k], sort_keys=True, default=float) for k in (H.S4_KEY, H.S5_KEY)}
            ok_all &= all(same.values())
            res.append({'policy': pol, 'scene': sc, 'arm': arm, 'geometry': gname, 'rows': len(b['rows']), 'identical': same, 'full_seconds': round(tf, 1), 'lean_seconds': round(tl, 1),
                        'S4_full': full[H.S4_KEY], 'S5_full_status': full[H.S5_KEY]['status'], 'S5_full_n_pairs': full[H.S5_KEY].get('n_pairs')})
            print(pol, sc, arm, gname, same, round(tf, 1), round(tl, 1), flush=True)
    C.jdump(out, {'what': 'lean S4/S5 (hull_s45) vs full hci.spatial_links.branch + spatial_items, exact JSON equality of every field', 'all_identical': ok_all, 'branches': res})
    if not ok_all: raise SystemExit(1)


if __name__ == '__main__':
    main()
