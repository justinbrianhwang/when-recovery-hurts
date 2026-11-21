"""Spatial audit of records written with the spatial logger (hci.spatial_probe, schema 'c1_spatial_probe_v1'): every robot
link, the cubes and the table as the simulator's own collision shapes at every logged step, the PhysX contact list, and
the whole-robot swept volume. Post-hoc, from stored records only; nothing here runs a simulator or reaches a controller.

Records without the spatial fields (Phase B, and any record of plan dd3af27e) are never filled in: every quantity that
needs them is NOT_OBSERVABLE. The earlier audit (hci.spatial_audit, E1-E10) is unchanged; the items here (S1-S6) are
separate, and their thresholds are EXPLORATORY unless named from the C1 spec.

Sources of truth, in this order: the logged collision shapes (PhysX's own, initial row) posed by the logged link / cube
poses; the logged PhysX contact list of the last sub-step. Interpolated sub-poses are an ASSUMPTION (swept volume only).
"""
import math
from pathlib import Path
import numpy as np
from . import spatial_geometry as G
from . import spatial_probe as SP

CONTACT_GEOM_TOL_M = 0.001          # the earlier audit's tolerance (hci.spatial_audit): a recorded contact must be within 1 mm of the reconstruction
SEPARATION_TOL_M = 0.001            # PhysX point separation vs the reconstructed distance of the same two shapes (EXPLORATORY)
ROBOT_VOXEL_M = 0.005               # whole-robot swept volume voxel (approximation)
GROUPS = {'arm_links': tuple(f'panda_link{i}' for i in range(8)), 'hand': ('panda_hand',), 'camera': ('camera_link',), 'fingers': ('panda_leftfinger', 'panda_rightfinger')}
TARGETS = ('cubeA', 'cubeB', 'table')
EXPLORATORY = 'EXPLORATORY (declared in hci.spatial_links before any C1 record exists; not in the C1 expectation spec)'


def has_spatial(rows): return bool(rows) and all(r.get('spatial_schema') == SP.SCHEMA for r in rows)


def group_of(link):
    for g, names in GROUPS.items():
        if link in names: return g
    return None


# ---------------------------------------------------------------- shapes
_LOCAL = {}                           # id(shape dict) -> (the dict itself, its local Convex); the dict is kept, so an id is never reused


def _local_shape(d):
    """A logged collision shape in its own frame (Convex), or None for a plane."""
    hit = _LOCAL.get(id(d))
    if hit is not None and hit[0] is d: return hit[1]
    t = d['type']
    if 'Box' in t: s = G.Convex(G._SIGNS * np.asarray(d['half_size'], float), G._BOX_TRI)
    elif 'ConvexMesh' in t: s = G.Convex(np.asarray(d['vertices'], float) * np.asarray(d['scale'], float), d['triangles'])
    elif 'Plane' in t: s = None
    else: raise ValueError(f'shape type {t} is not handled by the audit')
    _LOCAL[id(d)] = (d, s); return s


def posed(geometry, body, pose):
    """World Convex shapes of one body at a pose [x, y, z, qw, qx, qy, qz]."""
    Rb = G.quat_to_R(pose[3:]); pb = np.asarray(pose[:3], float); out = []
    for i, d in enumerate(geometry[body]['shapes']):
        loc = _local_shape(d)
        if loc is None: continue
        lp = d['local_pose']; out.append(loc.moved(Rb @ G.quat_to_R(lp[3:]), pb + Rb @ np.asarray(lp[:3], float), f'{body}#{i}'))
    return out


def world(row, geometry):
    """{body: [Convex]} for every robot link with shapes, both cubes and the table, at one logged row."""
    out = {n: posed(geometry, n, row['link_pose'][n]) for n in row['link_pose'] if geometry.get(n, {}).get('shapes')}
    out['cubeA'] = posed(geometry, 'cubeA', list(row['cubeA']) + list(row['cubeA_q'])); out['cubeB'] = posed(geometry, 'cubeB', list(row['cubeB']) + list(row['cubeB_q']))
    out['table'] = posed(geometry, 'table', geometry['table']['pose_at_this_row']); return out


def gap(X, Y):
    """Smallest exact signed distance between two lists of Convex (negative = penetration depth)."""
    return min(G.signed_distance(a, b) for a in X for b in Y)


# ---------------------------------------------------------------- contacts
def contact_pairs(row):
    """{(a, b) sorted by name: {'observed': any point with non-zero impulse, 'proximity_only': listed with zero impulse,
    'force_N': |sum impulse| / dt with the sign convention of body a, 'min_separation_m', 'points'}} from the logged list."""
    dt = row['sim_timestep_s']; out = {}
    for c in row['contacts']:
        a, b = c['a'], c['b']; sgn = 1.
        if b < a: a, b, sgn = b, a, -1.
        imp = np.sum([p[6:9] for p in c['points']], axis=0) * sgn if c['points'] else np.zeros(3)
        e = out.setdefault((a, b), {'impulse': np.zeros(3), 'observed': False, 'min_separation_m': math.inf, 'points': 0, 'shape_pairs': []})
        e['impulse'] = e['impulse'] + imp; e['observed'] |= any(any(v != 0 for v in p[6:9]) for p in c['points']); e['points'] += len(c['points'])
        e['min_separation_m'] = min([e['min_separation_m']] + [p[9] for p in c['points']]); e['shape_pairs'].append((c['a'], c['shape_a'], c['b'], c['shape_b'], min([p[9] for p in c['points']], default=math.inf), any(any(v != 0 for v in p[6:9]) for p in c['points'])))
    for e in out.values(): e['force_N'] = float(np.linalg.norm(e['impulse']) / dt); e['proximity_only'] = not e['observed']; e.pop('impulse')
    return out


def category(a, b):
    """'<robot group>-<cubeA|cubeB|table|other>', 'cubeA-cubeB', '<cube>-table', or None."""
    ga, gb = group_of(a), group_of(b)
    if ga and gb: return 'robot-self'
    if ga or gb:
        g, o = (ga, b) if ga else (gb, a); return f'{g}-{o if o in TARGETS else "other"}'
    s = {a, b}
    if s == {'cubeA', 'cubeB'}: return 'cubeA-cubeB'
    if 'table' in s and s & {'cubeA', 'cubeB'}: return f'{(s - {"table"}).pop()}-table'
    return 'other'


CATEGORIES = tuple(f'{g}-{t}' for g in GROUPS for t in (*TARGETS, 'other')) + ('cubeA-cubeB', 'cubeA-table', 'cubeB-table', 'robot-self', 'other')


# ---------------------------------------------------------------- per step
def row_metrics(row, geometry):
    """Exact clearances per robot group to each cube and the table; observed contact and proximity per category from the
    contact list; PhysX separation vs reconstruction for every listed shape pair; per-link world AABB."""
    W = world(row, geometry); out = {'t': int(row['t'])}
    for g, names in GROUPS.items():
        S = [s for n in names if n in W for s in W[n]]
        for tg in TARGETS: out[f'd_{g}_{tg}'] = gap(S, W[tg]) if S else None
    out['d_robot_cubeB'] = min(v for k, v in out.items() if k.startswith('d_') and k.endswith('_cubeB') and v is not None)
    out['robot_lowest_z'] = float(min(s.V[:, 2].min() for n in W if group_of(n) for s in W[n]))
    out['table_top_z'] = float(W['table'][0].V[:, 2].max())
    cp = contact_pairs(row); cats = {}
    for (a, b), e in cp.items():
        k = category(a, b); c = cats.setdefault(k, {'observed': False, 'proximity_only_pairs': 0, 'force_N': 0., 'min_separation_m': math.inf})
        c['observed'] |= e['observed']; c['proximity_only_pairs'] += int(e['proximity_only']); c['force_N'] = max(c['force_N'], e['force_N'] if e['observed'] else 0.); c['min_separation_m'] = min(c['min_separation_m'], e['min_separation_m'])
    for k in CATEGORIES:
        c = cats.get(k); out[f'observed_contact_{k}'] = bool(c and c['observed']); out[f'listed_{k}'] = bool(c); out[f'max_pair_force_{k}'] = c['force_N'] if c else 0.
    sep = []; skipped = []
    for (a, b), e in cp.items():
        for (na, ia, nb, ib, s, obs) in e['shape_pairs']:                # obs: this contact's own points carry an impulse
            A_ = [x for x in W.get(na, []) if x.name == f'{na}#{ia}']; B_ = [x for x in W.get(nb, []) if x.name == f'{nb}#{ib}']
            if not (A_ and B_): skipped.append({'a': na, 'shape_a': ia, 'b': nb, 'shape_b': ib, 'observed': obs, 'physx_min_separation_m': s}); continue      # e.g. the ground plane: not reconstructed, counted
            sep.append({'a': na, 'shape_a': ia, 'b': nb, 'shape_b': ib, 'physx_min_separation_m': s, 'reconstructed_m': gap(A_, B_), 'observed': obs})
    out['separation_vs_reconstruction'] = sep; out['contact_shape_pairs_not_reconstructed'] = skipped
    out['link_aabb'] = {n: [s.aabb()[0].tolist() + s.aabb()[1].tolist() for s in W[n]] for n in W if group_of(n)}
    return out


# ---------------------------------------------------------------- swept volume
def robot_shapes_along(rows, geometry, n=G.SIM_STEPS_PER_CONTROL):
    """Every robot shape at every logged row and at n-1 interpolated sub-poses between consecutive rows (per link: linear
    position, slerped orientation). The sub-poses are an ASSUMPTION, never an observation."""
    seq = []
    for r0, r1 in zip(rows, rows[1:]):
        for f in np.arange(0, n) / n:
            for name, p in r0['link_pose'].items():
                if not geometry.get(name, {}).get('shapes') or not group_of(name): continue
                q1 = r1['link_pose'][name]; pos = (1 - f) * np.asarray(p[:3]) + f * np.asarray(q1[:3]); q = G.slerp(p[3:], q1[3:], f)
                seq += posed(geometry, name, list(pos) + list(q))
    if rows:
        for name, p in rows[-1]['link_pose'].items():
            if geometry.get(name, {}).get('shapes') and group_of(name): seq += posed(geometry, name, p)
    return seq


def robot_swept_volume(rows, geometry, voxel=ROBOT_VOXEL_M):
    return G.swept_volume(robot_shapes_along(rows, geometry), voxel) if len(rows) else 0.


# ---------------------------------------------------------------- one branch
def branch(rec, tr, arm, geometry=None):
    """Rows of one arm from the trigger - 10 to its end, or (arm 'trunk') the whole trunk, once per scene (review a18): together
    every logged row is audited. Per-step metrics; None if not observable."""
    trig = rec['trigger']['step'] if rec.get('triggered') else None
    if arm == 'trunk':
        rows = list(tr['trunk']); trig = None
    else:
        rows = ([r for r in tr['trunk'] if r['t'] <= trig] + list(tr.get(arm) or [])) if trig is not None else list(tr['trunk'])
        lo = max(0, (trig or 0) - 10); rows = [r for r in rows if r['t'] >= lo]      # an arm branch: from trigger - 10; earlier rows are in the trunk entry
    if not has_spatial(rows): return None
    geometry = geometry or (rec.get('initial_state') or {}).get('collision_geometry')
    if not geometry: return None
    c = next((x for x in rec['cells'] if x['arm'] == arm), None) if trig is not None else None; k = c['control_steps'] if (c and c.get('acted')) else 0
    M = []
    for r in rows:                                                       # the trunk has no move: its phase is 'trunk', S1-S4 NOT_APPLICABLE
        m = row_metrics(r, geometry); m['phase'] = 'trunk' if arm == 'trunk' else ('before_trigger' if trig is None or r['t'] <= trig else ('control' if r['t'] <= trig + k else 'policy_after')); M.append(m)
    ctrl = [r for r in rows if trig is not None and trig <= r['t'] <= trig + k] if k else []
    return {'rows': rows, 'metrics': M, 'control_steps': k, 'trigger': trig, 'robot_swept_volume_control_m3': robot_swept_volume(ctrl, geometry) if len(ctrl) > 1 else None,
            'swept_note': 'logged link poses plus interpolated sub-poses (ASSUMPTION), 5 mm voxels: an approximation, not an observation'}


SPATIAL_ITEMS = (('S1_no_robot_contact_with_target_during_move', 'no observed contact (non-zero impulse) between any robot shape (arm links, hand, camera, fingers) and the target in the logged contact lists of the move'),
                 ('S2_no_robot_contact_with_table_during_move', 'no observed contact between any robot shape and the table during the move'),
                 ('S3_no_arm_hand_or_camera_contact_with_held_cube_during_move', 'no observed contact between the arm links, the hand or the camera and the held cube during the move (the fingers hold it)'),
                 ('S4_robot_clear_of_target_at_logged_poses', 'reconstructed distance robot - target > 0 at every logged pose of the move'),
                 ('S5_recorded_contacts_agree_with_geometry', f'every observed contact shape pair is reconstructed no more than {CONTACT_GEOM_TOL_M * 1e3:.0f} mm APART (one-sided, the rule of hci.spatial_audit). '
                  'Overlap is not a violation: PhysX lets bodies interpenetrate under load (about 2 mm under the grip in Phase B); the penetration depth is reported, without a threshold (review a19)'),
                 ('S6_physx_separation_agrees_with_geometry', f'PhysX point separation and the reconstructed distance of the same two shapes differ by at most {SEPARATION_TOL_M * 1e3:.0f} mm (every pair PhysX reports separated, all phases)'))


def spatial_items(b, acted):
    """S1-S6 for one branch (EXPLORATORY). b = branch(...) or None (record without the spatial fields)."""
    if b is None: return {k: {'status': 'NOT_OBSERVABLE', 'why': 'the record has no spatial fields (written without hci.spatial_probe)', 'threshold': EXPLORATORY} for k, _ in SPATIAL_ITEMS}
    M = b['metrics']; mv = [m for m in M if m['phase'] == 'control' and m['t'] > b['trigger']] if acted else []
    st = lambda ok, **kw: {'status': 'PASS' if ok else 'FAIL', 'threshold': EXPLORATORY, **kw}
    na = {'status': 'NOT_APPLICABLE', 'why': 'no move (abstained, C0 or zero control steps)', 'threshold': EXPLORATORY}
    out = {}
    if mv:
        rob = lambda tg: [k for k in CATEGORIES if k.endswith('-' + tg) and k.split('-')[0] in GROUPS]
        hit = lambda ks: [m['t'] for m in mv if any(m[f'observed_contact_{k}'] for k in ks)]
        out['S1_no_robot_contact_with_target_during_move'] = st(not hit(rob('cubeB')), steps=hit(rob('cubeB')))
        out['S2_no_robot_contact_with_table_during_move'] = st(not hit(rob('table')), steps=hit(rob('table')))
        k3 = [f'{g}-cubeA' for g in ('arm_links', 'hand', 'camera')]; out['S3_no_arm_hand_or_camera_contact_with_held_cube_during_move'] = st(not hit(k3), steps=hit(k3))
        d = min(m['d_robot_cubeB'] for m in mv); out['S4_robot_clear_of_target_at_logged_poses'] = st(d > 0, min_distance_m=d)
    else:
        for k in ('S1_no_robot_contact_with_target_during_move', 'S2_no_robot_contact_with_table_during_move', 'S3_no_arm_hand_or_camera_contact_with_held_cube_during_move', 'S4_robot_clear_of_target_at_logged_poses'): out[k] = dict(na)
    bad = [(m['t'], s['a'], s['b'], s['reconstructed_m']) for m in M for s in m['separation_vs_reconstruction'] if s['observed'] and s['reconstructed_m'] > CONTACT_GEOM_TOL_M]
    pen = [-s['reconstructed_m'] for m in M for s in m['separation_vs_reconstruction'] if s['observed'] and s['reconstructed_m'] < 0]
    pen_info = {'max_penetration_m': max(pen) if pen else 0., 'median_penetration_m': float(np.median(pen)) if pen else 0., 'n_overlapping': len(pen), 'penetration': 'reported only, no threshold (one-sided rule)'}
    n5 = sum(1 for m in M for s in m['separation_vs_reconstruction'] if s['observed']); none = lambda n: {'status': 'NOT_OBSERVABLE', 'why': 'no shape pair could be compared', 'n_pairs': n, 'threshold': EXPLORATORY}
    sk = sum(1 for m in M for x in m['contact_shape_pairs_not_reconstructed'] if x['observed'])
    if bad: out['S5_recorded_contacts_agree_with_geometry'] = st(False, violations=bad[:20], n_violations=len(bad), n_pairs=n5, n_not_reconstructed=sk, **pen_info)      # a definite violation is FAIL (review a6)
    elif sk: out['S5_recorded_contacts_agree_with_geometry'] = {'status': 'NOT_OBSERVABLE', 'why': f'{sk} observed contact shape pairs involve a shape the audit does not reconstruct (review a5)', 'n_pairs': n5, 'n_not_reconstructed': sk, 'n_violations': len(bad), 'threshold': EXPLORATORY, **pen_info}
    else: out['S5_recorded_contacts_agree_with_geometry'] = st(True, violations=[], n_violations=0, n_pairs=n5, n_not_reconstructed=0, **pen_info) if n5 else none(0)
    dif = [abs(s['physx_min_separation_m'] - s['reconstructed_m']) for m in M for s in m['separation_vs_reconstruction'] if s['physx_min_separation_m'] > 0]      # every pair PhysX reports separated, whatever the reconstruction says (review a6)
    sk6 = sum(1 for m in M for x in m['contact_shape_pairs_not_reconstructed'] if x['physx_min_separation_m'] > 0)
    worst = max(dif) if dif else None; summ = dict(n_pairs=len(dif), n_not_reconstructed=sk6, max_abs_difference_m=worst, median_abs_difference_m=float(np.median(dif)) if dif else None)
    if dif and worst > SEPARATION_TOL_M: out['S6_physx_separation_agrees_with_geometry'] = st(False, **summ)                       # a definite disagreement is FAIL
    elif sk6: out['S6_physx_separation_agrees_with_geometry'] = {'status': 'NOT_OBSERVABLE', 'why': f'{sk6} separated shape pairs involve a shape the audit does not reconstruct (review a7)', **summ, 'threshold': EXPLORATORY}
    else: out['S6_physx_separation_agrees_with_geometry'] = st(True, **summ) if dif else none(0)
    return out


# ---------------------------------------------------------------- geometry against the fixed upstream URDF / STL
def read_stl(path):
    """Binary or ASCII STL -> (vertices (n,3), triangles (m,3)), vertices merged exactly."""
    b = Path(path).read_bytes()
    if b[:5] == b'solid' and b'facet' in b[:400]:
        pts = [list(map(float, l.split()[1:4])) for l in b.decode('ascii', 'replace').splitlines() if l.strip().startswith('vertex')]; P = np.asarray(pts, float)
    else:
        n = int(np.frombuffer(b[80:84], np.uint32)[0]); rec = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])); P = rec['v'].reshape(-1, 3).astype(float)
    V, inv = np.unique(P, axis=0, return_inverse=True); return V, inv.reshape(-1, 3)


def urdf_collisions(urdf_path):
    """{link: [{'mesh': path | None, 'box': size | None, 'xyz', 'rpy'}]} from the <collision> elements of a URDF."""
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot(); out = {}
    for l in root.findall('link'):
        lst = []
        for c in l.findall('collision'):
            o = c.find('origin'); xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]; rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            g = c.find('geometry'); m = g.find('mesh'); bx = g.find('box')
            lst.append({'mesh': None if m is None else str(Path(urdf_path).parent / m.get('filename')), 'box': None if bx is None else [float(v) for v in bx.get('size').split()], 'xyz': xyz, 'rpy': rpy,
                        'scale': [1., 1., 1.] if m is None or m.get('scale') is None else [float(v) for v in m.get('scale').split()]})
        out[l.get('name')] = lst
    return out


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def geometry_against_urdf(geometry, urdf_path, vertex_tol_m=5e-4, volume_rel_tol=0.03):
    """Each logged robot collision shape against the URDF: boxes by size and origin; convex meshes by the convex hull of
    the STL in the link frame (every PhysX hull vertex within vertex_tol_m of that hull; volumes within volume_rel_tol).
    Tolerances EXPLORATORY: PhysX cooks its own hull (vertex limit), so vertex sets differ."""
    from scipy.spatial import ConvexHull
    U = urdf_collisions(urdf_path); rows = []
    for link, ent in geometry.items():
        if link not in U: continue
        shapes = ent['shapes']
        if len(shapes) != len(U[link]): rows.append({'link': link, 'pass': False, 'why': f'{len(shapes)} logged shapes, {len(U[link])} in the URDF'}); continue
        for i, (d, u) in enumerate(zip(shapes, U[link])):
            lp = d['local_pose']; Rl = G.quat_to_R(lp[3:]); Ru = rpy_to_R(*u['rpy'])
            pos_err = float(np.linalg.norm(np.asarray(lp[:3]) - np.asarray(u['xyz']))); rot_err = float(np.degrees(np.arccos(np.clip((np.trace(Rl.T @ Ru) - 1) / 2, -1, 1))))
            if u['box'] is not None:
                ok = 'Box' in d['type'] and np.allclose(np.asarray(d['half_size']) * 2, u['box'], atol=1e-6) and pos_err < 1e-6 and rot_err < 1e-3
                rows.append({'link': link, 'shape': i, 'kind': 'box', 'pass': bool(ok), 'size_logged': (np.asarray(d.get('half_size', [np.nan] * 3)) * 2).tolist(), 'size_urdf': u['box'], 'origin_error_m': pos_err, 'rotation_error_deg': rot_err})
            else:
                V, _ = read_stl(u['mesh']); V = (V * np.asarray(u['scale'])) @ Ru.T + np.asarray(u['xyz']); h = ConvexHull(V)
                P = (np.asarray(d['vertices']) * np.asarray(d['scale'])) @ Rl.T + np.asarray(lp[:3]); eq = h.equations
                outside = float(max(0., (P @ eq[:, :3].T + eq[:, 3]).max(1).max())); hp = ConvexHull(P)
                ok = 'ConvexMesh' in d['type'] and outside <= vertex_tol_m and abs(hp.volume / h.volume - 1) <= volume_rel_tol
                rows.append({'link': link, 'shape': i, 'kind': 'convex mesh', 'mesh': Path(u['mesh']).name, 'pass': bool(ok), 'physx_vertices': len(P), 'stl_hull_vertices': len(h.vertices),
                             'max_vertex_outside_stl_hull_m': outside, 'volume_ratio_physx_to_stl_hull': float(hp.volume / h.volume)})
    return {'rows': rows, 'pass': bool(rows) and all(r['pass'] for r in rows), 'urdf': Path(urdf_path).name, 'tolerances': {'vertex_m': vertex_tol_m, 'volume_rel': volume_rel_tol, 'note': EXPLORATORY}}


# ---------------------------------------------------------------- record sufficiency for the spatial-logger format
REQUIREMENTS_V2 = [
    ('robot joint state (9 joints) per step', 'robot_qpos, robot_qvel, robot_joint_names (initial row)'),
    ('pose of every robot link per step (read from the simulator, no forward kinematics)', 'link_pose'),
    ('TCP pose per step', 'tcp, tcp_q (unchanged) and link_pose[panda_hand_tcp]'),
    ('finger poses per step', 'link_pose[panda_leftfinger / panda_rightfinger], finger_qpos'),
    ('collision geometry of every body, with shape identifiers', 'collision_geometry (initial row): body -> shapes with index, type, local pose, size / PhysX hull'),
    ('simulation time per step', 'elapsed_steps, sim_time_s, sim_timestep_s, sim_substeps_per_control'),
    ('contacts hand - target, hand - held cube, target - table, gripper - table, arm links - target / held cube / table / other', 'contacts (every pair of the last sub-step, with points), f_hand_*, f_B_table, f_finger*_table, f_links_*'),
    ('contact position, normal, impulse and separation', 'contacts[].points as PhysX gives them'),
]


def requirements_v2(rec, tr):
    rows = [r for k in tr for r in tr[k]]; ini = rec.get('initial_state') or {}; ok = has_spatial(rows) and 'collision_geometry' in ini
    items = [{'item': it, 'status': 'AVAILABLE' if ok else 'NOT_OBSERVABLE', 'source': src} for it, src in REQUIREMENTS_V2]
    items.append({'item': 'contacts of the first four of the five PhysX sub-steps; contacts between logged steps', 'status': 'REQUIRES_SIMULATOR_INSTRUMENTATION',
                  'source': 'not logged: the step loop is not changed; continuous-time collision freedom is not established'})
    items.append({'item': 'RGB after the move (both cameras)', 'status': 'AVAILABLE' if rec.get('frames') else 'NOT_OBSERVABLE', 'source': 'C1 frames after_<ARM> at the listed steps'})
    return {'spatial_logger_record': ok, 'items': items}
