"""C1 S4/S5 convex-hull sensitivity (POST_HOC_EXPLORATORY; stored records only; no simulation, no paid compute).

Question: does the triangulation error of the logged PhysX convex meshes (found by the post-hoc physics diagnostics:
the logged triangles of panda_link0-7 and panda_hand are not the convex hull of their own vertices; hand distances
change by up to 0.24 mm) change any S4 or S5 status of C1?

Two geometries, everything else identical (same rows, same poses, same contact lists, same code path hci.spatial_links):
  LOGGED  the collision geometry as stored in each record (the official reconstruction);
  HULL    the same record geometry with every ConvexMesh replaced by the convex hull of the same (scaled) vertices
          (scipy Qhull); boxes, the plane, poses and scales unchanged.

S4 and S5 are recomputed exactly as hci.spatial_links.spatial_items defines them, per branch as scripts/38 builds them
(the full trunk once; every arm from trigger - 10 to its end):
  S4  min over the logged poses of the move (trigger < t <= trigger + control_steps, acted arms) of the reconstructed
      distance robot (arm links, hand, camera, fingers) - target (cubeB); PASS if > 0.
  S5  every observed contact shape pair (a point with nonzero impulse) of every row of the branch is reconstructed no
      more than 1 mm apart; FAIL on a violation; NOT_OBSERVABLE if an observed pair involves an unreconstructed shape.
Only the distances S4 and S5 need are computed (the full row metrics take ~0.28 s per row); the equivalence of this lean
path with the full hci.spatial_links path is checked separately (check_equivalence.py) on whole branches.
"""
from __future__ import annotations

import copy, hashlib, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'c1_physics_001'))
import common as C  # noqa: E402  (paths, sha-checked loaders)
from hci import spatial_links as L  # noqa: E402
from hci import spatial_geometry as G  # noqa: E402

OUT = C.REPO / 'results' / 'vision_center_v2_c1_001' / 'posthoc_hull_sensitivity_001'
S4_KEY, S5_KEY = 'S4_robot_clear_of_target_at_logged_poses', 'S5_recorded_contacts_agree_with_geometry'
MESH_BODIES = tuple(f'panda_link{i}' for i in range(8)) + ('panda_hand',)


# ---------------------------------------------------------------- geometry variants
_HULL_CACHE = {}


def hull_of_shape(d):
    """A copy of one logged ConvexMesh shape dict whose vertices/triangles are the convex hull of the same scaled vertices.
    Vertices not on the hull are dropped; the scale is kept (the hull is computed on the scaled points and the kept
    vertices are the original unscaled ones, so _local_shape's scaling is unchanged)."""
    from scipy.spatial import ConvexHull
    key = hashlib.sha256(json.dumps([d['vertices'], d['triangles'], d['scale']]).encode()).hexdigest()
    if key not in _HULL_CACHE:
        V = np.asarray(d['vertices'], float); h = ConvexHull(V * np.asarray(d['scale'], float))
        keep = np.unique(h.simplices); remap = {int(o): i for i, o in enumerate(keep)}
        _HULL_CACHE[key] = (V[keep].tolist(), [[remap[int(x)] for x in s] for s in h.simplices])
    out = dict(d); out['vertices'], out['triangles'] = copy.deepcopy(_HULL_CACHE[key]); out['hull_of_logged'] = True; return out


def hull_geometry(geometry):
    """The record geometry with every ConvexMesh replaced by its hull; everything else is the same object content."""
    g = copy.deepcopy(geometry)
    for body, e in g.items():
        e['shapes'] = [hull_of_shape(d) if 'ConvexMesh' in d['type'] else d for d in e.get('shapes', [])]
    return g


def shape_stats(geometry):
    """Per logged mesh shape: vertex/triangle counts, volume of the logged triangulation vs its hull, how deep the logged
    surface dips inside the hull (max over triangle vertices, centroids and edge midpoints), and how many logged
    triangles are not hull faces. An a-priori picture of where the reconstruction can differ."""
    from scipy.spatial import ConvexHull
    rows = []
    for body, e in geometry.items():
        for i, d in enumerate(e.get('shapes', [])):
            if 'ConvexMesh' not in d['type']: continue
            V = np.asarray(d['vertices'], float) * np.asarray(d['scale'], float); T = np.asarray(d['triangles'], int); h = ConvexHull(V)
            P = V[T]; samples = np.concatenate([V, P.mean(1), (P[:, 0] + P[:, 1]) / 2, (P[:, 1] + P[:, 2]) / 2, (P[:, 2] + P[:, 0]) / 2])
            depth = -(samples @ h.equations[:, :3].T + h.equations[:, 3]).max(1)           # >= 0 inside the hull
            # a logged triangle lies on the hull surface iff its three vertices lie on one hull facet plane (a different
            # diagonal of the same planar face is still on the hull); tolerance 1e-9 m
            on = np.abs(np.einsum('tkj,fj->tkf', P, h.equations[:, :3]) + h.equations[:, 3]).max(1) <= 1e-9
            off = int((~on.any(1)).sum())
            logged_vol = G.Convex(V, T).volume()
            rows.append({'body': body, 'shape': i, 'vertices_logged': len(V), 'vertices_on_hull': len(h.vertices), 'triangles_logged': len(T), 'triangles_hull': len(h.simplices),
                         'logged_triangles_not_hull_faces': off, 'volume_logged_m3': logged_vol, 'volume_hull_m3': float(h.volume), 'volume_ratio_logged_to_hull': logged_vol / h.volume,
                         'max_logged_surface_depth_inside_hull_m': float(depth.max()), 'max_vertex_outside_hull_m': float(max(0., -depth[:len(V)].min()))})
    return rows


# ---------------------------------------------------------------- the rows of one branch (as hci.spatial_links.branch)
def branch_rows(rec, tr, arm):
    trig = rec['trigger']['step'] if rec.get('triggered') else None
    if arm == 'trunk': rows = list(tr['trunk']); trig = None
    else:
        rows = ([r for r in tr['trunk'] if r['t'] <= trig] + list(tr.get(arm) or [])) if trig is not None else list(tr['trunk'])
        lo = max(0, (trig or 0) - 10); rows = [r for r in rows if r['t'] >= lo]
    c = next((x for x in rec['cells'] if x['arm'] == arm), None) if trig is not None else None
    k = c['control_steps'] if (c and c.get('acted')) else 0
    src = ['trunk' if (arm == 'trunk' or (trig is not None and r['t'] <= trig)) else arm for r in rows]      # which trace a row comes from (dedupe key)
    return rows, src, trig, k, c


def arms_of(rec, tr):
    return ['trunk'] + [c['arm'] for c in rec['cells'] if c['arm'] in tr] if rec.get('triggered') else ['trunk']


# ---------------------------------------------------------------- the distances S4 / S5 need, per row and geometry
def s4_row(W):
    """d per robot group to cubeB and d_robot_cubeB, exactly as hci.spatial_links.row_metrics."""
    out = {}
    for g, names in L.GROUPS.items():
        S = [s for n in names if n in W for s in W[n]]; out[g] = L.gap(S, W['cubeB']) if S else None
    out['robot'] = min(v for v in out.values() if v is not None); return out


def s5_pairs(row, W):
    """Observed contact shape pairs of one row with their reconstructed distance (None if a shape is not reconstructed),
    in the order of hci.spatial_links.row_metrics."""
    out = []
    for (a, b), e in L.contact_pairs(row).items():
        for (na, ia, nb, ib, s, obs) in e['shape_pairs']:
            if not obs: continue
            A_ = [x for x in W.get(na, []) if x.name == f'{na}#{ia}']; B_ = [x for x in W.get(nb, []) if x.name == f'{nb}#{ib}']
            out.append({'a': na, 'shape_a': ia, 'b': nb, 'shape_b': ib, 'physx_min_separation_m': s, 'reconstructed_m': L.gap(A_, B_) if (A_ and B_) else None})
    return out


def items_from(rows_m, trig, k, acted):
    """S4 / S5 dicts with the fields and rules of hci.spatial_links.spatial_items, from per-row lean metrics."""
    st = lambda ok, **kw: {'status': 'PASS' if ok else 'FAIL', 'threshold': L.EXPLORATORY, **kw}
    mv = [m for m in rows_m if acted and trig is not None and trig < m['t'] <= trig + k]
    out = {}
    if mv:
        d = min(m['s4']['robot'] for m in mv); out[S4_KEY] = st(d > 0, min_distance_m=d)
    else: out[S4_KEY] = {'status': 'NOT_APPLICABLE', 'why': 'no move (abstained, C0 or zero control steps)', 'threshold': L.EXPLORATORY}
    pairs = [(m['t'], p) for m in rows_m for p in m['s5']]
    bad = [(t, p['a'], p['b'], p['reconstructed_m']) for t, p in pairs if p['reconstructed_m'] is not None and p['reconstructed_m'] > L.CONTACT_GEOM_TOL_M]
    pen = [-p['reconstructed_m'] for t, p in pairs if p['reconstructed_m'] is not None and p['reconstructed_m'] < 0]
    pen_info = {'max_penetration_m': max(pen) if pen else 0., 'median_penetration_m': float(np.median(pen)) if pen else 0., 'n_overlapping': len(pen), 'penetration': 'reported only, no threshold (one-sided rule)'}
    n5 = sum(1 for t, p in pairs if p['reconstructed_m'] is not None); sk = sum(1 for t, p in pairs if p['reconstructed_m'] is None)
    if bad: out[S5_KEY] = st(False, violations=bad[:20], n_violations=len(bad), n_pairs=n5, n_not_reconstructed=sk, **pen_info)
    elif sk: out[S5_KEY] = {'status': 'NOT_OBSERVABLE', 'why': f'{sk} observed contact shape pairs involve a shape the audit does not reconstruct (review a5)', 'n_pairs': n5, 'n_not_reconstructed': sk, 'n_violations': len(bad), 'threshold': L.EXPLORATORY, **pen_info}
    elif n5: out[S5_KEY] = st(True, violations=[], n_violations=0, n_pairs=n5, n_not_reconstructed=0, **pen_info)
    else: out[S5_KEY] = {'status': 'NOT_OBSERVABLE', 'why': 'no shape pair could be compared', 'n_pairs': 0, 'threshold': L.EXPLORATORY}
    return out


def scene_task(args):
    """Everything for one (policy, scene): per-branch S4/S5 under both geometries, per-move-row S4 distances, per observed
    pair distances. Rows shared by several branches are computed once."""
    policy, scene = args
    rec, rec_sha = C.record(policy, scene); tr, tr_sha = C.traces(policy, scene)
    geo = {'LOGGED': rec['initial_state']['collision_geometry']}; geo['HULL'] = hull_geometry(geo['LOGGED'])
    trig = rec['trigger']['step'] if rec.get('triggered') else None
    cache = {}; branches = []; move_rows = []; pair_rows = []

    def metrics(row, src, need_s4):
        key = (src, int(row['t']))
        if key not in cache: cache[key] = {}
        e = cache[key]
        for gname, g in geo.items():
            if gname not in e:
                W = L.world(row, g); e[gname] = {'W': W, 's5': s5_pairs(row, W)}
            if need_s4 and 's4' not in e[gname]: e[gname]['s4'] = s4_row(e[gname]['W'])
        return e

    for arm in arms_of(rec, tr):
        rows, src, t0, k, c = branch_rows(rec, tr, arm); acted = bool(c and c.get('acted') and k > 0)
        per = {g: [] for g in geo}
        for r, s in zip(rows, src):
            need = acted and t0 is not None and t0 < r['t'] <= t0 + k
            e = metrics(r, s, need)
            for g in geo: per[g].append({'t': int(r['t']), 's5': e[g]['s5'], **({'s4': e[g]['s4']} if need else {})})
        res = {g: items_from(per[g], t0, k, acted) for g in geo}
        branches.append({'policy': policy, 'scene': scene, 'arm': arm, 'acted': acted, 'control_steps': k, 'trigger': t0, 'rows': len(rows),
                         'y': None if c is None else c.get('y'), 'record_sha256': rec_sha, 'trace_sha256': tr_sha, 'items': res})
    for (s, t), e in sorted(cache.items(), key=lambda x: (x[0][0], x[0][1])):
        if 's4' in e['LOGGED']:
            a, b = e['LOGGED']['s4'], e['HULL']['s4']
            move_rows.append({'policy': policy, 'scene': scene, 'arm': s, 't': t, **{f'd_{g}_cubeB_logged_m': a[g] for g in (*L.GROUPS, 'robot')}, **{f'd_{g}_cubeB_hull_m': b[g] for g in (*L.GROUPS, 'robot')},
                              'nearest_group_logged': min((g for g in L.GROUPS if a[g] is not None), key=lambda g: a[g]), 'nearest_group_hull': min((g for g in L.GROUPS if b[g] is not None), key=lambda g: b[g])})
        for pa, pb in zip(e['LOGGED']['s5'], e['HULL']['s5']):
            assert (pa['a'], pa['shape_a'], pa['b'], pa['shape_b']) == (pb['a'], pb['shape_a'], pb['b'], pb['shape_b'])
            mesh = pa['a'] in MESH_BODIES or pa['b'] in MESH_BODIES
            pair_rows.append({'policy': policy, 'scene': scene, 'trace': s, 't': t, 'a': pa['a'], 'shape_a': pa['shape_a'], 'b': pa['b'], 'shape_b': pa['shape_b'], 'involves_mesh': mesh,
                              'physx_min_separation_m': pa['physx_min_separation_m'], 'reconstructed_logged_m': pa['reconstructed_m'], 'reconstructed_hull_m': pb['reconstructed_m']})
    stats = shape_stats(geo['LOGGED'])
    return {'policy': policy, 'scene': scene, 'branches': branches, 'move_rows': move_rows, 'pair_rows': pair_rows, 'mesh_stats': stats,
            'geometry_sha256': hashlib.sha256(json.dumps(geo['LOGGED'], sort_keys=True).encode()).hexdigest()}
