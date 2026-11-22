"""A2 definition check (POST_HOC_EXPLORATORY): is a PhysX point separation the minimum distance between the two shapes,
or a distance measured ALONG the contact normal from the contact point to the other shape?

For every PhysX point of a selected pair row: the point lies (within numerical noise) on one shape; we compute
  euclid  = signed distance from the point to the other shape (reference path)
  ray     = distance from the point along +-normal to the surface of the other shape (convex ray entry, own code)
and compare both with the point's own separation, with the shapes posed at row t and at f = 0.8 of the control step.
Rows: the slow (relative speed < 0.1 m/s) over-tolerance finger rows of the stratified subset, the other over-tolerance
rows and a within-tolerance sample (fixed seed). Reads S6_RECOMPUTED_SUBSET.csv.gz and the raw traces; writes
S6_NORMAL_RAY_CHECK.csv and S6_NORMAL_RAY_SUMMARY.json.
"""
from __future__ import annotations

import json, sys
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import diagnose_s6_offline as D  # noqa: E402
import geometry_reference as REF  # noqa: E402


def ray_entry(x, u, V):
    """Smallest t >= 0 with x + t u inside conv(V) (the ray from outside entering the shape), or None."""
    H = ConvexHull(V); n = H.equations[:, :3]; d = H.equations[:, 3]; nu = n @ u; s = n @ x + d
    t0, t1 = 0., np.inf
    for a, b in zip(nu, s):                 # constraint: b + t a <= 0
        if abs(a) < 1e-15:
            if b > 0: return None
        elif a > 0: t1 = min(t1, -b / a)
        else: t0 = max(t0, -b / a)
    return float(t0) if t0 <= t1 else None


def check(item):
    policy, scene, rows = item; rec, _ = C.record(policy, scene); tr, _ = C.traces(policy, scene); geo = rec['initial_state']['collision_geometry']; out = []
    idx = {arm: {int(r['t']): i for i, r in enumerate(rs)} for arm, rs in tr.items()}
    for r in rows:
        arm = r['arm']; i = idx[arm][int(r['t'])]; r1 = tr[arm][i]; r0 = tr[arm][i - 1]
        a, b = r['shape_pair'].split('|'); (na, sa), (nb, sb) = a.split('#'), b.split('#'); sa, sb = int(sa), int(sb)
        moving = lambda n: n in r1['link_pose'] or n.startswith('cube')
        for tag, f in (('t', 1.0), ('f0.8', 0.8)):
            pa = D.interp(D.body_pose(r0, geo, na), D.body_pose(r1, geo, na), f) if moving(na) else D.body_pose(r1, geo, na)
            pb = D.interp(D.body_pose(r0, geo, nb), D.body_pose(r1, geo, nb), f) if moving(nb) else D.body_pose(r1, geo, nb)
            da, db = geo[na]['shapes'][sa], geo[nb]['shapes'][sb]
            VA = REF.pose_vertices(D.local_vertices(da), da['local_pose'], pa); VB = REF.pose_vertices(D.local_vertices(db), db['local_pose'], pb)
            for c in r1['contacts']:
                if {(c['a'], c['shape_a']), (c['b'], c['shape_b'])} != {(na, sa), (nb, sb)}: continue
                for p in c['points']:
                    x = np.asarray(p[0:3], float); nrm = np.asarray(p[3:6], float); sep = float(p[9])
                    ea, eb = D.point_signed_distance(x, VA), D.point_signed_distance(x, VB)
                    on_a = abs(ea) <= abs(eb); other = VB if on_a else VA; euclid = eb if on_a else ea
                    rays = [t for t in (ray_entry(x, nrm, other), ray_entry(x, -nrm, other)) if t is not None]
                    u = nrm if (other.mean(0) - x) @ nrm > 0 else -nrm          # normal oriented from the point towards the other shape
                    plane = float(((other - x) @ u).min())                        # distance along u to the other shape's supporting plane
                    out.append({'policy': policy, 'scene': scene, 'arm': arm, 't': int(r['t']), 'shape_pair': r['shape_pair'], 'kind': r['kind'], 'group': r['group'], 'pose': tag,
                                'point_sep_m': sep, 'point_on': na if on_a else nb, 'dist_to_own_shape_m': min(abs(ea), abs(eb)), 'euclid_to_other_m': euclid,
                                'ray_along_normal_to_other_m': min(rays) if rays else None, 'support_plane_along_normal_m': plane, 'impulse_norm': float(np.linalg.norm(p[6:9]))})
    return out


def main():
    out = C.OUT; d = pd.read_csv(out / 'S6_RECOMPUTED_SUBSET.csv.gz'); s = d[d.selection == 'stratified_sample']
    rng = np.random.default_rng(D.SEED)
    slow_f = s[s.kind.str.startswith('fingers') & s.over_tol & (s.speed_rel_mps < 0.1)].assign(group='finger_slow_over_tol')
    other = s[s.over_tol & ~s.index.isin(slow_f.index)]; other = other.iloc[rng.choice(len(other), min(400, len(other)), replace=False)].assign(group='other_over_tol')
    within = s[~s.over_tol]; within = within.iloc[rng.choice(len(within), min(400, len(within)), replace=False)].assign(group='within_tol')
    sel = pd.concat([slow_f, other, within]); items = defaultdict(list)
    for r in sel.to_dict('records'): items[(r['policy'], int(r['scene']))].append(r)
    res = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for rr in ex.map(check, [(p, s_, rows) for (p, s_), rows in sorted(items.items())], chunksize=2): res += rr
    r = pd.DataFrame(res); r.to_csv(out / 'S6_NORMAL_RAY_CHECK.csv', index=False)
    r['err_euclid'] = (r.point_sep_m - r.euclid_to_other_m).abs(); r['err_ray'] = (r.point_sep_m - r.ray_along_normal_to_other_m).abs(); r['err_plane'] = (r.point_sep_m - r.support_plane_along_normal_m).abs()
    summ = {'label': C.LABEL, 'points': len(r), 'rows': len(sel), 'groups': {}}
    for (g, pose), gg in r.groupby(['group', 'pose']):
        summ['groups'][f'{g} @ {pose}'] = {'points': len(gg), 'point_on_own_shape_median_mm': float(gg.dist_to_own_shape_m.median() * 1e3),
                                          'median_abs_sep_minus_euclid_mm': float(gg.err_euclid.median() * 1e3), 'share_euclid_within_0.05mm': float((gg.err_euclid < 5e-5).mean()),
                                          'median_abs_sep_minus_ray_mm': float(gg.err_ray.median() * 1e3), 'share_ray_within_0.05mm': float((gg.err_ray < 5e-5).mean()),
                                          'points_without_ray_hit': int(gg.ray_along_normal_to_other_m.isna().sum()),
                                          'median_abs_sep_minus_support_plane_mm': float(gg.err_plane.median() * 1e3), 'share_support_plane_within_0.05mm': float((gg.err_plane < 5e-5).mean()),
                                          'share_best_of_euclid_or_plane_within_0.05mm': float(((gg.err_plane < 5e-5) | (gg.err_euclid < 5e-5)).mean())}
    # per row: does the row's minimum point separation equal the minimum over its points of the support-plane distance?
    rows = r.groupby(['group', 'pose', 'policy', 'scene', 'arm', 't', 'shape_pair']).agg(sep=('point_sep_m', 'min'), plane=('support_plane_along_normal_m', 'min'), euclid=('euclid_to_other_m', 'min')).reset_index()
    summ['rows_min_over_points'] = {f'{g} @ {p}': {'rows': len(x), 'median_abs_minsep_minus_minplane_mm': float((x.sep - x.plane).abs().median() * 1e3), 'share_within_1mm_plane': float(((x.sep - x.plane).abs() <= 1e-3).mean()),
                                                   'median_abs_minsep_minus_mineuclid_mm': float((x.sep - x.euclid).abs().median() * 1e3), 'share_within_1mm_euclid': float(((x.sep - x.euclid).abs() <= 1e-3).mean())}
                                    for (g, p), x in rows.groupby(['group', 'pose'])}
    summ['reading'] = ('If a point separation equals the ray distance along its normal but not the Euclidean distance, PhysX reports a separation along the contact normal at a '
                       'manifold point, which is not the minimum shape distance S6 compares it with. Association on stored states; PhysX internals not observed.')
    C.jdump(out / 'S6_NORMAL_RAY_SUMMARY.json', summ); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
