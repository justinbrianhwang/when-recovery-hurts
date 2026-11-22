"""A: decomposition of the C1 S6 measurement-validity FAIL (POST_HOC_EXPLORATORY; the official S6 stays FAIL).

Inputs (read-only): the S6 pair rows written by the C1 mechanism analysis (mechanism_c1_001/tables/s6_pair_rows.csv.gz,
one row per (policy, scene, arm, logged step t, listed shape pair) that PhysX reports separated), and the raw C1 traces
(each checked against the sha256 stored in its scene record).

A1  official S6 reproduced from the pair rows with the unchanged definition (|physx_sep - recon| > 1 mm), then grouped by
    impulse flag, pair kind, shape pair, phase, PhysX separation, relative speed, normal component of the relative velocity,
    angular speed, static pairs, and whether the same shape pair was already a separated listed pair at t-1.
    Rows are not independent (same scene / arm / step); row statistics are descriptive. Intervals resample scenes, keeping
    both policies' rows of a scene together.
A2  for a rule-selected subset, recomputed from the raw trace rows t-1 and t:
    - row mapping: the existing reconstruction at t (must equal the stored recon) and an independent one
      (geometry_reference: own posing, GJK / Qhull);
    - all PhysX points of the pair (count, min / max separation, impulse), and the signed distance of each point position
      to both shapes (where PhysX puts its points);
    - time-shift sensitivity, EXPLANATORY_ONLY: the same reconstruction with every moving body posed at a fraction f of the
      control step between rows t-1 (f = 0) and t (f = 1), linear position and slerp (constant-velocity ASSUMPTION), and a
      one-sub-step back-step with the LOGGED cube velocities (cubeA_v / cubeA_w ...) for the cubes and the finite-difference
      motion of the robot links. These are not the sub-step states (not logged) and never change S6.
"""
from __future__ import annotations

import argparse, gzip, json, math, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import geometry_reference as REF  # noqa: E402
from hci import spatial_geometry as G  # noqa: E402
from hci import spatial_links as SL  # noqa: E402

TOL = 0.001                                   # the official S6 tolerance, unchanged
FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SAMPLE_PER_STRATUM = 250                      # per (policy, kind, over_tol) stratum; all rows if fewer
SEED = 20261122


# ---------------------------------------------------------------- A1 helpers
def load_rows(path):
    d = pd.read_csv(path)
    d['abs_diff_m'] = d.diff_m.abs(); d['over_tol'] = d.abs_diff_m > TOL
    d['impulse_group'] = np.where(d.nonzero_impulse_recorded.isna(), 'missing', np.where(d.nonzero_impulse_recorded.astype(str) == 'True', 'nonzero_impulse', 'zero_impulse'))
    d['shape_pair'] = d.a + '#' + d.shape_a.astype(str) + '|' + d.b + '#' + d.shape_b.astype(str)
    d['sep_bin'] = pd.cut(d.physx_sep_m, [0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.0401], labels=['(0,1]mm', '(1,5]mm', '(5,10]mm', '(10,20]mm', '(20,30]mm', '(30,40]mm']).astype(str)
    d['speed_bin'] = pd.cut(d.speed_rel_mps, [-1, 0.001, 0.01, 0.05, 0.1, 1e9], labels=['static <1 mm/s', '1-10 mm/s', '10-50 mm/s', '50-100 mm/s', '>=100 mm/s']).astype(str)
    d['vn_dt_mm'] = d.vrel_dot_n_mps * d.dt_sub_s * 1e3
    d['vn_bin'] = pd.cut(d.vn_dt_mm, [-1e9, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1e9], labels=['<-0.5', '-0.5..-0.1', '-0.1..-0.01', '|.|<0.01', '0.01..0.1', '0.1..0.5', '>0.5']).astype(str)
    d['ang_bin'] = pd.cut(d.ang_speed_a_rps, [-1, 0.1, 1, 1e9], labels=['<0.1 rad/s', '0.1-1 rad/s', '>=1 rad/s']).astype(str)
    key = list(zip(d.policy, d.scene, d.arm, d.shape_pair, d.t))
    present = set(key)
    d['listed_separated_at_t_minus_1'] = [(p, s, a, sp, t - 1) in present for p, s, a, sp, t in key]
    return d


def stats(g):
    a = g.abs_diff_m.to_numpy(); s = g.diff_m.to_numpy(); i = int(np.argmax(a))
    w = g.iloc[i]
    return {'n': len(g), 'over_tol': int((a > TOL).sum()), 'over_tol_fraction': float((a > TOL).mean()),
            'signed_median_mm': float(np.median(s) * 1e3), 'abs_median_mm': float(np.median(a) * 1e3), 'abs_p90_mm': float(np.percentile(a, 90) * 1e3),
            'abs_p99_mm': float(np.percentile(a, 99) * 1e3), 'abs_max_mm': float(a.max() * 1e3),
            'worst_policy': w.policy, 'worst_scene': int(w.scene), 'worst_arm': w.arm, 'worst_t': int(w.t), 'worst_shape_pair': w.shape_pair, 'worst_diff_mm': float(w.diff_m * 1e3)}


GROUPS = ['impulse_group', 'kind', 'shape_pair', 'phase', 'sep_bin', 'speed_bin', 'vn_bin', 'ang_bin', 'listed_separated_at_t_minus_1']
CROSS = [('kind', 'impulse_group'), ('kind', 'sep_bin'), ('sep_bin', 'impulse_group'), ('speed_bin', 'sep_bin'), ('kind', 'speed_bin')]


def group_table(d):
    out = []
    for pol, dp in [('both (rows of both policies; descriptive)', d)] + [(p, d[d.policy == p]) for p in C.POLICIES]:
        out.append({'source_kind': C.SOURCE_KIND, 'label': C.LABEL, 'policy': pol, 'grouping': 'all', 'level': 'all', **stats(dp)})
        for g in GROUPS:
            for lv, gg in dp.groupby(g, sort=True): out.append({'source_kind': C.SOURCE_KIND, 'label': C.LABEL, 'policy': pol, 'grouping': g, 'level': str(lv), **stats(gg)})
        for g1, g2 in CROSS:
            for (l1, l2), gg in dp.groupby([g1, g2], sort=True): out.append({'source_kind': C.SOURCE_KIND, 'label': C.LABEL, 'policy': pol, 'grouping': f'{g1} x {g2}', 'level': f'{l1} | {l2}', **stats(gg)})
    return out


def scene_bootstrap(d, by, B=2000):
    """95% interval of the over-tolerance fraction per level, resampling scenes (both policies' rows of a scene kept together)."""
    rng = np.random.default_rng(SEED); scenes = np.array(sorted(d.scene.unique()))
    agg = d.groupby(['scene', by]).agg(n=('over_tol', 'size'), k=('over_tol', 'sum')).reset_index()
    out = {}
    for lv, g in agg.groupby(by):
        n = g.set_index('scene').n.reindex(scenes, fill_value=0).to_numpy(); k = g.set_index('scene').k.reindex(scenes, fill_value=0).to_numpy()
        idx = rng.integers(0, len(scenes), (B, len(scenes))); fr = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
        out[str(lv)] = {'fraction': float(k.sum() / max(n.sum(), 1)), 'scene_bootstrap95': [float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))], 'scenes_with_rows': int((n > 0).sum())}
    return out


# ---------------------------------------------------------------- A2: raw-trace recomputation
def body_pose(row, geo, body):
    if body in ('cubeA', 'cubeB'): return list(row[body]) + list(row[f'{body}_q'])
    if body in row['link_pose']: return list(row['link_pose'][body])
    return list(geo[body]['pose_at_this_row'])           # static bodies (table)


def shape_of(geo, body, idx, pose):
    return [s for s in SL.posed(geo, body, pose) if s.name == f'{body}#{idx}']


def interp(p0, p1, f): return list((1 - f) * np.asarray(p0[:3]) + f * np.asarray(p1[:3])) + list(G.slerp(p0[3:], p1[3:], f))


def backstep(row, geo, body, p0, p1, dt_sub):
    """One sub-step back from row t: the logged linear / angular velocity for the cubes; the control-step finite
    difference (f = 1 - 1/5) for robot links; static bodies unchanged."""
    if body in ('cubeA', 'cubeB'):
        v = np.asarray(row[f'{body}_v'], float); w = np.asarray(row[f'{body}_w'], float); p = np.asarray(row[body], float) - v * dt_sub
        ang = np.linalg.norm(w) * dt_sub; q1 = np.asarray(row[f'{body}_q'], float)
        if ang > 0:
            ax = w / np.linalg.norm(w); dq = np.array([math.cos(-ang / 2), *(math.sin(-ang / 2) * ax)])     # rotation by -w dt (world frame) applied on the left
            w0, x0, y0, z0 = dq; w1, x1, y1, z1 = q1
            q = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1, w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
        else: q = q1
        return list(p) + list(q)
    if body in row['link_pose']: return interp(p0, p1, 1 - 1 / 5.)
    return p1


def point_signed_distance(x, V):
    """Signed distance of a point to the convex hull of V (reference path): GJK outside, minus the facet distance inside."""
    from scipy.spatial import ConvexHull
    H = ConvexHull(V); s = H.equations[:, :3] @ x + H.equations[:, 3]
    if np.all(s <= 0): return float(s.max())
    d, *_ = REF.gjk_distance(np.asarray([x]), V); return d


def recompute(item):
    policy, scene, rows = item
    rec, rec_sha = C.record(policy, scene); tr, tr_sha = C.traces(policy, scene); geo = rec['initial_state']['collision_geometry']; out = []
    idx = {arm: {int(r['t']): i for i, r in enumerate(rs)} for arm, rs in tr.items()}
    for r in rows:
        arm = r['arm']; i = idx[arm][int(r['t'])]; r1 = tr[arm][i]; r0 = tr[arm][i - 1]
        a, sa, b, sb = r['a'], int(r['shape_a']), r['b'], int(r['shape_b']); dt_sub = float(r1['sim_timestep_s'])
        pa0, pa1, pb0, pb1 = body_pose(r0, geo, a), body_pose(r1, geo, a), body_pose(r0, geo, b), body_pose(r1, geo, b)
        A1, B1 = shape_of(geo, a, sa, pa1), shape_of(geo, b, sb, pb1)
        recon_t = SL.gap(A1, B1)
        ref_t, ref_method = REF.signed_distance_ref(A1[0].V, B1[0].V)        # vertices posed by the existing code; posing checked separately below
        # independent posing of the same shape (own rotation conversion)
        da, db = geo[a]['shapes'][sa], geo[b]['shapes'][sb]
        VA = REF.pose_vertices(local_vertices(da), da['local_pose'], pa1); VB = REF.pose_vertices(local_vertices(db), db['local_pose'], pb1)
        ref_indep, _ = REF.signed_distance_ref(VA, VB)
        # PhysX points of the pair (both orientations)
        pts = []; orient = None
        for c in r1['contacts']:
            if (c['a'], c['shape_a'], c['b'], c['shape_b']) == (a, sa, b, sb): pts += c['points']; orient = 'ab'
            elif (c['b'], c['shape_b'], c['a'], c['shape_a']) == (a, sa, b, sb): pts += c['points']; orient = 'ba'
        seps = [p[9] for p in pts]; imp = [float(np.linalg.norm(p[6:9])) for p in pts]
        pmin = min(pts, key=lambda p: p[9]) if pts else None
        dpa = dpb = None
        if pmin is not None:
            x = np.asarray(pmin[0:3], float); dpa = point_signed_distance(x, VA); dpb = point_signed_distance(x, VB)
        dpa8 = dpb8 = None
        if pmin is not None:                 # the same point against both shapes posed at f = 0.8 (start of the last sub-step under constant velocity)
            pa8 = interp(pa0, pa1, 0.8) if (a in r1['link_pose'] or a.startswith('cube')) else pa1; pb8 = interp(pb0, pb1, 0.8) if (b in r1['link_pose'] or b.startswith('cube')) else pb1
            dpa8 = point_signed_distance(x, REF.pose_vertices(local_vertices(da), da['local_pose'], pa8)); dpb8 = point_signed_distance(x, REF.pose_vertices(local_vertices(db), db['local_pose'], pb8))
        shifted = {}
        for f in FRACTIONS:
            shifted[f] = SL.gap(shape_of(geo, a, sa, interp(pa0, pa1, f) if a in r1['link_pose'] or a.startswith('cube') else pa1),
                                shape_of(geo, b, sb, interp(pb0, pb1, f) if b in r1['link_pose'] or b.startswith('cube') else pb1))
        bs = SL.gap(shape_of(geo, a, sa, backstep(r1, geo, a, pa0, pa1, dt_sub)), shape_of(geo, b, sb, backstep(r1, geo, b, pb0, pb1, dt_sub)))
        sep = float(r['physx_sep_m']); best_f = min(FRACTIONS, key=lambda f: abs(sep - shifted[f]))
        out.append({'source_kind': C.SOURCE_KIND, 'label': C.LABEL, 'selection': r['selection'], 'policy': policy, 'scene': scene, 'arm': arm, 't': int(r['t']),
                    'shape_pair': r['shape_pair'], 'kind': r['kind'], 'impulse_group': r['impulse_group'], 'over_tol': bool(r['over_tol']),
                    'record_sha256': rec_sha, 'trace_sha256': tr_sha,
                    'physx_sep_m': sep, 'stored_recon_m': float(r['recon_m']), 'recon_recomputed_m': recon_t, 'recon_mapping_abs_err_m': abs(recon_t - float(r['recon_m'])),
                    'reference_same_vertices_m': ref_t, 'reference_method': ref_method, 'reference_independent_pose_m': ref_indep,
                    'reference_minus_existing_m': ref_indep - recon_t, 'n_points': len(pts), 'contact_orientation_in_list': orient,
                    'point_sep_min_m': min(seps) if seps else None, 'point_sep_max_m': max(seps) if seps else None, 'point_sep_spread_m': (max(seps) - min(seps)) if seps else None,
                    'point_impulse_sum_Ns': sum(imp) if imp else None,
                    'minpoint_signed_dist_to_shape_a_m': dpa, 'minpoint_signed_dist_to_shape_b_m': dpb,
                    'minpoint_signed_dist_to_shape_a_f0.8_m': dpa8, 'minpoint_signed_dist_to_shape_b_f0.8_m': dpb8,
                    'minpoint_nearest_surface_t_m': None if dpa is None else min(abs(dpa), abs(dpb)), 'minpoint_nearest_surface_f0.8_m': None if dpa8 is None else min(abs(dpa8), abs(dpb8)),
                    'speed_rel_mps': r['speed_rel_mps'], 'vn_dt_mm': r['vn_dt_mm'],
                    **{f'diff_at_f{f:.1f}_m': sep - shifted[f] for f in FRACTIONS}, 'diff_backstep_logged_velocity_m': sep - bs,
                    'best_fraction': best_f, 'best_fraction_abs_diff_m': abs(sep - shifted[best_f])})
    return out


def local_vertices(d):
    if 'Box' in d['type']: return REF.box_vertices(d['half_size'])
    return np.asarray(d['vertices'], float) * np.asarray(d['scale'], float)


def select(d):
    """Rule-based subset: per (policy, kind, over_tol) a fixed-seed random sample of SAMPLE_PER_STRATUM rows (all if fewer),
    plus the extremes: per policy and impulse group the 10 largest |diff|, per policy and kind the 3 largest."""
    rng = np.random.default_rng(SEED); parts = []
    for (p, k, o), g in d.groupby(['policy', 'kind', 'over_tol']):
        g = g if len(g) <= SAMPLE_PER_STRATUM else g.iloc[np.sort(rng.choice(len(g), SAMPLE_PER_STRATUM, replace=False))]
        parts.append(g.assign(selection='stratified_sample'))
    for (p, ig), g in d.groupby(['policy', 'impulse_group']): parts.append(g.nlargest(10, 'abs_diff_m').assign(selection=f'extreme_{ig}'))
    for (p, k), g in d.groupby(['policy', 'kind']): parts.append(g.nlargest(3, 'abs_diff_m').assign(selection='extreme_kind'))
    s = pd.concat(parts)
    s = s.sort_values('selection').drop_duplicates(['policy', 'scene', 'arm', 't', 'shape_pair'], keep='last')     # an extreme keeps its extreme label
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(C.OUT)); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--smoke-scenes', type=int, default=0, help='smoke test only: recompute the subset of this many scenes'); a = ap.parse_args()
    out = Path(a.out)
    if (out / 'S6_DIAGNOSTIC_SUMMARY.json').exists(): raise SystemExit(f'refused: {out} already holds an S6 diagnosis (new run id needed)')
    (out / 'figures').mkdir(parents=True, exist_ok=True); t0 = time.time(); C.OUT = out
    src = C.MECH / 'tables' / 's6_pair_rows.csv.gz'; src_sha = C.sha256(src)
    man = json.loads((C.MECH / 'analysis_manifest.json').read_text(encoding='utf-8'))
    want = man['outputs'].get('tables/s6_pair_rows.csv.gz')
    if want is not None and want != src_sha: raise SystemExit(f'pair rows sha256 {src_sha} != analysis manifest {want}')
    d = load_rows(src)
    official = json.loads((C.MECH / 'tables' / 's6_validity_summary.json').read_text(encoding='utf-8'))
    repro = {p: {'pairs': int((d.policy == p).sum()), 'over_tolerance': int(d[d.policy == p].over_tol.sum()),
                 'expected_pairs': official[p]['official_S6']['pairs'], 'expected_over_tolerance': official[p]['official_S6']['over_tolerance'],
                 'status_official': official[p]['official_S6']['status']} for p in C.POLICIES}
    for p, v in repro.items():
        v['reproduced'] = v['pairs'] == v['expected_pairs'] and v['over_tolerance'] == v['expected_over_tolerance']
        if not v['reproduced']: raise SystemExit(f'S6 not reproduced for {p}: {v}')
    print('A1 reproduced', repro, flush=True)
    gt = group_table(d); C.write_csv(out / 'S6_ERRORS_BY_GROUP.csv', gt)
    boot = {by: scene_bootstrap(d, by) for by in ('kind', 'sep_bin', 'impulse_group', 'speed_bin')}
    # A2
    sel = select(d); items = defaultdict(list)
    for r in sel.to_dict('records'): items[(r['policy'], int(r['scene']))].append(r)
    work = [(p, s, rows) for (p, s), rows in sorted(items.items())]
    if a.smoke_scenes: work = work[:a.smoke_scenes]; sel = pd.DataFrame([r for _, _, rr in work for r in rr])
    print('A2 rows', len(sel), 'scenes', len(work), flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, rr in enumerate(ex.map(recompute, work, chunksize=2)):
            res += rr
            if i % 50 == 0: print('  scene', i, len(res), round(time.time() - t0), flush=True)
    rd = pd.DataFrame(res); rd.to_csv(out / 'S6_RECOMPUTED_SUBSET.csv.gz', index=False, compression='gzip')
    summary = summarise(d, rd, repro, boot, src, src_sha)
    ext = extremes(rd); C.write_csv(out / 'S6_EXTREME_CASES.csv', ext)
    summary['extreme_cases_rule'] = 'per policy: the largest |diff| with a nonzero impulse and with zero impulse; the largest among static pairs (relative speed < 1 mm/s); the largest among near-contact pairs (PhysX separation <= 1 mm); the largest cube-table pair'
    figures(d, rd, out / 'figures')
    summary['seconds'] = round(time.time() - t0, 1)
    C.jdump(out / 'S6_DIAGNOSTIC_SUMMARY.json', summary); print('done', summary['seconds'], flush=True)


def q(x, p): x = np.asarray(x, float); x = x[~np.isnan(x)]; return float(np.percentile(x, p)) if len(x) else None


def summarise(d, rd, repro, boot, src, src_sha):
    s = {'label': C.LABEL, 'source_kind': C.SOURCE_KIND, 'input': {'pair_rows': C.rel(src), 'sha256': src_sha}, 'tolerance_m': TOL,
         'original_S6': {p: {'status': 'FAIL', **v} for p, v in repro.items()}, 'original_S6_note': 'reproduced exactly from the stored pair rows; FAIL kept; nothing below changes it',
         'contact_offset_evidence': {'max_physx_separation_listed_m': float(d.physx_sep_m.max()), 'p99_9_m': float(d.physx_sep_m.quantile(0.999)),
                                     'reading': 'consistent with contact_offset 0.02 m per shape (sum 0.04 m) of the pinned ManiSkill SceneConfig; the run did not log the value'},
         'impulse_groups': {k: {'n': int(len(g)), 'over_tol': int(g.over_tol.sum()), 'fraction': float(g.over_tol.mean())} for k, g in d.groupby('impulse_group')},
         'scene_bootstrap_over_tol_fraction': boot}
    near = d[d.physx_sep_m <= TOL]; far = d[d.physx_sep_m > 0.01]
    s['near_contact_vs_far'] = {'physx_sep_le_1mm': {'n': len(near), 'over_tol': int(near.over_tol.sum())}, 'physx_sep_gt_10mm': {'n': len(far), 'over_tol': int(far.over_tol.sum())},
                                'share_of_over_tol_rows_with_physx_sep_gt_10mm': float(far.over_tol.sum() / d.over_tol.sum())}
    # A2 summaries (subset; stratified, so per stratum, never as a population estimate)
    a2 = {'rows': len(rd), 'scenes': int(rd[['policy', 'scene']].drop_duplicates().shape[0]),
          'row_mapping': {'max_abs_err_recomputed_vs_stored_m': float(rd.recon_mapping_abs_err_m.max()), 'rows_with_err_gt_1e-9_m': int((rd.recon_mapping_abs_err_m > 1e-9).sum())},
          'independent_reconstruction': {'max_abs_reference_minus_existing_m': float(rd.reference_minus_existing_m.abs().max()),
                                         'p99_abs_m': q(rd.reference_minus_existing_m.abs(), 99), 'methods': rd.reference_method.value_counts().to_dict(),
                                         'by_kind': {k: {'n': len(g), 'max_abs_mm': float(g.reference_minus_existing_m.abs().max() * 1e3), 'median_signed_mm': float(g.reference_minus_existing_m.median() * 1e3),
                                                         'sign_disagreements': int((np.sign(g.reference_independent_pose_m) != np.sign(g.recon_recomputed_m)).sum()),
                                                         'over_tol_rows_flipped_to_within_by_reference': int(((g.physx_sep_m - g.stored_recon_m).abs().gt(TOL) & (g.physx_sep_m - g.reference_independent_pose_m).abs().le(TOL)).sum())}
                                                     for k, g in rd.groupby('kind')},
                                         'note': 'existing = hci.spatial_geometry on the logged triangles; reference = convex hull of the logged vertices (GJK / Qhull). They differ only where the logged triangulation is not the convex hull (the PhysX-cooked hand mesh).'},
          'point_placement': {'nearest_surface_at_t_median_mm': q(rd['minpoint_nearest_surface_t_m'] * 1e3, 50), 'nearest_surface_at_t_p90_mm': q(rd['minpoint_nearest_surface_t_m'] * 1e3, 90),
                              'nearest_surface_at_f0.8_median_mm': q(rd['minpoint_nearest_surface_f0.8_m'] * 1e3, 50), 'nearest_surface_at_f0.8_p90_mm': q(rd['minpoint_nearest_surface_f0.8_m'] * 1e3, 90),
                              'over_tol_rows': {'at_t_median_mm': q(rd[rd.over_tol]['minpoint_nearest_surface_t_m'] * 1e3, 50), 'at_f0.8_median_mm': q(rd[rd.over_tol]['minpoint_nearest_surface_f0.8_m'] * 1e3, 50)}},
          'points': {'n_points_distribution': rd.n_points.value_counts().sort_index().to_dict(), 'rows_without_points': int((rd.n_points == 0).sum()),
                     'min_point_equals_stored_physx_sep_rows': int((np.abs(rd.point_sep_min_m - rd.physx_sep_m) < 1e-9).sum()),
                     'median_point_sep_spread_mm': q(rd.point_sep_spread_m * 1e3, 50)}}
    by = []
    for (k, o), g in rd[rd.selection == 'stratified_sample'].groupby(['kind', 'over_tol']):
        row = {'kind': k, 'over_tol_at_t': bool(o), 'n': len(g)}
        for f in FRACTIONS:
            col = f'diff_at_f{f:.1f}_m'; row[f'median_abs_diff_mm_f{f:.1f}'] = q(g[col].abs() * 1e3, 50); row[f'within_1mm_f{f:.1f}'] = float((g[col].abs() <= TOL).mean())
        row['median_abs_diff_mm_backstep'] = q(g.diff_backstep_logged_velocity_m.abs() * 1e3, 50); row['within_1mm_backstep'] = float((g.diff_backstep_logged_velocity_m.abs() <= TOL).mean())
        row['best_fraction_counts'] = g.best_fraction.value_counts().sort_index().to_dict(); row['within_1mm_at_best_fraction'] = float((g.best_fraction_abs_diff_m <= TOL).mean())
        row['minpoint_on_shape_a_median_abs_mm'] = q(g.minpoint_signed_dist_to_shape_a_m.abs() * 1e3, 50); row['minpoint_on_shape_b_median_abs_mm'] = q(g.minpoint_signed_dist_to_shape_b_m.abs() * 1e3, 50)
        by.append(row)
    a2['time_shift_by_kind_EXPLANATORY_ONLY'] = by
    C.write_csv(Path(C.OUT) / 'S6_TIME_SHIFT_BY_KIND.csv', [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in by])
    s['A2_raw_recomputation_subset'] = a2
    return s


def extremes(rd):
    out = []
    def add(g, why):
        if len(g): r = g.loc[g.physx_sep_m.sub(g.stored_recon_m).abs().idxmax()].to_dict(); r['why'] = why; out.append(r)
    for p in C.POLICIES:
        g = rd[rd.policy == p]
        add(g[g.impulse_group == 'nonzero_impulse'], 'largest |diff| with a recorded nonzero impulse')
        add(g[g.impulse_group == 'zero_impulse'], 'largest |diff| with zero impulse (proximity-only listed pair)')
        add(g[g.speed_rel_mps < 0.001], 'largest |diff| among static pairs (relative speed < 1 mm/s)')
        add(g[g.physx_sep_m <= TOL], 'largest |diff| among near-contact pairs (PhysX separation <= 1 mm)')
        add(g[g.kind.isin(['cubeA-table', 'cubeB-table'])], 'largest |diff| among cube-table pairs (logged cube velocities available)')
    return out


def figures(d, rd, fdir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110)
    for p, c in zip(C.POLICIES, ('#2a6fdb', '#d9822b')):
        g = d[d.policy == p].groupby('sep_bin').agg(n=('over_tol', 'size'), k=('over_tol', 'sum'))
        ax[0].bar(np.arange(len(g)) + (0.2 if p.endswith('2') else -0.2), g.k / g.n, width=0.4, color=c, label=p)
        ax[0].set_xticks(np.arange(len(g))); ax[0].set_xticklabels(g.index, fontsize=8)
    ax[0].set_ylabel('fraction of pair-rows with |diff| > 1 mm'); ax[0].set_xlabel('PhysX min point separation (bin)'); ax[0].legend(fontsize=8)
    ax[0].set_title('A1: S6 disagreement by PhysX separation (all 580,539 rows)', fontsize=9, loc='left')
    s = d.sample(min(len(d), 60000), random_state=SEED)
    ax[1].hexbin(s.vn_dt_mm.clip(-2, 2), (s.diff_m * 1e3).clip(-6, 6), gridsize=70, bins='log', cmap='Greys'); ax[1].axhline(1, color='r', lw=.7, ls='--'); ax[1].axhline(-1, color='r', lw=.7, ls='--')
    ax[1].set_xlabel('relative normal velocity x sub-step time, (v_a - v_b).n dt_sub [mm]'); ax[1].set_ylabel('PhysX sep - reconstruction [mm]')
    ax[1].set_title('A1: diff vs normal motion in one sub-step (60k-row sample; red: +-1 mm)', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(fdir / 's6_by_separation_and_motion.png'); plt.close(fig)
    g = rd[rd.selection == 'stratified_sample']
    kinds = [k for k in g.kind.value_counts().index if (g.kind == k).sum() >= 20]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110, sharey=True)
    for ax, o in zip(axs, (True, False)):
        for k in kinds:
            gg = g[(g.kind == k) & (g.over_tol == o)]
            if len(gg) < 10: continue
            ax.plot(FRACTIONS, [np.median(gg[f'diff_at_f{f:.1f}_m'].abs()) * 1e3 for f in FRACTIONS], marker='o', ms=3, label=f'{k} (n={len(gg)})')
        ax.axhline(1, color='r', lw=.7, ls='--'); ax.set_xlabel('pose fraction of the control step used for the reconstruction (0 = row t-1, 1 = row t)')
        ax.set_title(f'A2 EXPLANATORY_ONLY: median |diff| vs pose time, rows {"over" if o else "within"} 1 mm at t', fontsize=8.5, loc='left')
    axs[0].set_ylabel('median |PhysX sep - reconstruction| [mm]'); axs[0].legend(fontsize=6.5); axs[0].set_yscale('log')
    fig.tight_layout(); fig.savefig(fdir / 's6_time_shift_sensitivity.png'); plt.close(fig)


if __name__ == '__main__':
    main()
