"""Spatial occupancy audit: geometry with known answers, contact vs proximity, swept volume, frame/trace sync, commanded vs
measured paths, missing data, and determinism. No simulator; C1-format records come from C1.collect_scene on the mock
runtime (as in tests/test_trajectory_audit.py)."""
import copy, gzip, importlib.util, json, math
from pathlib import Path
import numpy as np
import pytest
from hci import spatial_geometry as G
from hci import spatial_audit as SA

Rz = lambda a: G.rot_z(math.radians(a))
qz = lambda a: [math.cos(math.radians(a) / 2), 0., 0., math.sin(math.radians(a) / 2)]
qx = lambda a: [math.cos(math.radians(a) / 2), math.sin(math.radians(a) / 2), 0., 0.]


# ---------------------------------------------------------------- 1 bounding boxes: position and rotation
def test_quaternion_is_w_first_and_rotates_as_expected():
    assert np.allclose(G.quat_to_R([1, 0, 0, 0]), np.eye(3))
    assert np.allclose(G.quat_to_R(qz(90)) @ [1, 0, 0], [0, 1, 0]) and np.allclose(G.quat_to_R(qx(90)) @ [0, 1, 0], [0, 0, 1])
    R = G.quat_to_R([.3, -.5, .7, .1]); assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert math.isclose(G.rotation_angle_deg(qz(10), qz(40)), 30, abs_tol=1e-9)


def test_cube_box_has_the_logged_centre_size_and_rotation():
    c = G.cube([.1, -.2, .02], qz(30)); assert np.allclose(c.V.mean(0), [.1, -.2, .02]) and math.isclose(c.volume(), .04 ** 3, rel_tol=1e-12)
    corner = np.array([.1, -.2, .02]) + Rz(30) @ [.02, .02, .02]; assert np.min(np.linalg.norm(c.V - corner, axis=1)) < 1e-12
    assert np.isclose(c.V[:, 2].min(), 0) and np.isclose(c.V[:, 2].max(), .04)


# ---------------------------------------------------------------- 2 coordinate transforms
def test_finger_and_hand_frames_follow_the_urdf_from_the_logged_tcp():
    tcp = np.array([.1, .2, .3]); g = G.gripper(tcp, [1, 0, 0, 0], [.02, .015])
    hand_origin = tcp - [0, 0, G.HAND_TO_TCP_Z_M]
    # rubber tip centre: finger joint origin + joint value along +/-y + tip origin (URDF)
    assert np.allclose(g['left_rubber_tip'].V.mean(0), hand_origin + [0, .02 + 7.58e-3, G.FINGER_JOINT_Z_M + 45.25e-3])
    assert np.allclose(g['right_rubber_tip'].V.mean(0), hand_origin + [0, -.015 - 7.58e-3, G.FINGER_JOINT_Z_M + 45.25e-3])
    inner_left = g['left_rubber_tip'].V[:, 1].min() - tcp[1]; assert math.isclose(inner_left, .02 + 7.58e-3 - 7.6e-3, abs_tol=1e-12)   # inner pad face at the joint value
    # pads sit at the TCP height (0.0584 + 0.04525 = 0.10365 m from the hand origin; TCP at 0.1034)
    assert abs(g['left_rubber_tip'].V.mean(0)[2] - tcp[2]) < 3e-4
    # a TCP turned 180 deg about x (gripper pointing down) puts the hand above the TCP and keeps left on +y of the TCP frame
    gd = G.gripper(tcp, qx(180), [.02, .02]); assert gd['hand'].V[:, 2].min() > tcp[2] and np.isclose(gd['left_rubber_tip'].V.mean(0)[1], tcp[1] - (.02 + 7.58e-3))
    assert len([k for k in g if k.startswith('left_')]) == 4 and math.isclose(g['hand'].volume(), G.hand_hull_local().volume())


def test_hand_hull_matches_its_stored_upstream_mesh():
    import hashlib, struct
    d = json.loads(G.HULL_FILE.read_text()); stl = (G.ROOT / 'assets/upstream_geometry/panda_hand_collision.stl').read_bytes()
    assert hashlib.sha256(stl).hexdigest() == d['stl_sha256'] == d['upstream_file_sha256']['mani_skill/assets/robots/panda/franka_description/meshes/collision/hand.stl']
    n = struct.unpack('<I', stl[80:84])[0]; V = np.frombuffer(stl[84:84 + 50 * n], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))['v'].reshape(-1, 3).astype(float)
    assert np.allclose(np.unique(V, axis=0), np.unique(np.array(d['vertices']), axis=0))
    h = G.hand_hull_local(); P = h.V[h.T]; nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    assert (h.V @ nrm.T - np.einsum('ij,ij->i', nrm, P[:, 0])).max() < 1e-9                      # convex: every vertex inside every face plane


# ---------------------------------------------------------------- 3 XY footprints
def test_footprints_reflect_size_and_rotation_and_overlap_is_exact():
    sq = G.footprint(G.cube([0, 0, .02], [1, 0, 0, 0])); assert math.isclose(G.polygon_area(sq), .04 ** 2)
    di = G.footprint(G.cube([0, 0, .02], qz(45))); assert math.isclose(G.polygon_area(di), .04 ** 2) and math.isclose(np.abs(di).max(), .02 * math.sqrt(2))
    tilt = G.footprint(G.cube([0, 0, .03], qx(20))); assert G.polygon_area(tilt) > .04 ** 2 * 1.2           # a tilted cube covers more of the table
    b = G.footprint(G.cube([.01, 0, .02], [1, 0, 0, 0])); assert math.isclose(G.overlap_area(sq, b), .03 * .04, rel_tol=1e-9)
    far = G.footprint(G.cube([.05, 0, .02], [1, 0, 0, 0])); assert G.overlap_area(sq, far) == 0 and math.isclose(G.polygon_gap(sq, far), .01)
    # centre distance alone would say 28.3 mm apart (> 20 mm) while the rotated footprints overlap
    a45 = G.footprint(G.cube([0, 0, .02], qz(45))); c2 = G.footprint(G.cube([.02, .02, .02], qz(45)))
    assert G.overlap_area(a45, c2) > 0 and G.polygon_gap(a45, c2) == 0


# ---------------------------------------------------------------- 4 surface gaps
@pytest.mark.parametrize('pose,expect', [
    (([.05, 0, .02], [1, 0, 0, 0]), .01),                                 # face to face
    (([.05, .05, .02], [1, 0, 0, 0]), math.hypot(.01, .01)),              # edge to edge
    (([.05, .05, .07], [1, 0, 0, 0]), math.sqrt(3) * .01),                # corner to corner
    (([.03, 0, .02], [1, 0, 0, 0]), -.01),                                # 10 mm penetration
    (([0, 0, .02 + .04 + .02 * math.sqrt(2) - .02 + .003], qx(45)), .003),  # rotated edge above a face
])
def test_box_surface_gap_known_cases(pose, expect):
    a = G.cube([0, 0, .02], [1, 0, 0, 0]); b = G.cube(*pose); assert math.isclose(G.signed_distance(a, b), expect, abs_tol=1e-12)
    assert math.isclose(G.signed_distance(b, a), expect, abs_tol=1e-12)


def test_hull_to_box_gap_agrees_with_surface_sampling():
    rng = np.random.default_rng(3); h = G.gripper([0, 0, .2], qx(180), [.03, .03])['hand']
    u = np.linspace(-.02, .02, 81); A, B = np.meshgrid(u, u); A, B = A.ravel(), B.ravel(); H = np.full_like(A, .02)
    local = np.concatenate([np.stack(x, 1) for s in (1, -1) for x in ((s * H, A, B), (A, s * H, B), (A, B, s * H))])
    for _ in range(6):
        p = np.array([rng.uniform(-.05, .05), rng.uniform(-.12, .12), rng.uniform(.37, .40)]); q = rng.normal(size=4); q /= np.linalg.norm(q)
        c = G.cube(p, q); exact = G.signed_distance(h, c); assert exact > 0
        pts = p + local @ G.quat_to_R(q).T                                 # cube surface samples (0.5 mm grid)
        samp = G.point_triangle(pts, h.V[h.T[:, 0]], h.V[h.T[:, 1]], h.V[h.T[:, 2]]).min()
        assert exact <= samp + 1e-12 and samp - exact < 4e-4                # sampling never undercuts the exact gap and converges to it


def test_segment_and_triangle_distances_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(60):
        p = rng.normal(size=3); a, b, c = rng.normal(size=(3, 3)); u, v = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300)); m = u + v <= 1
        bf = np.linalg.norm(a + np.outer(u[m], b - a) + np.outer(v[m], c - a) - p, axis=1).min(); ex = G.point_triangle(p[None], a[None], b[None], c[None])[0, 0]
        assert ex <= bf + 1e-12 and bf - ex < 5e-3
        P1, Q1, P2, Q2 = rng.normal(size=(4, 3)); s = np.linspace(0, 1, 600)
        bf = np.linalg.norm((P1 + np.outer(s, Q1 - P1))[:, None] - (P2 + np.outer(s, Q2 - P2))[None], axis=-1).min(); ex = G.segment_segment(P1[None], Q1[None], P2[None], Q2[None])[0, 0]
        assert ex <= bf + 1e-12 and bf - ex < 5e-3


# ---------------------------------------------------------------- 5 observed contact vs proximity vs potential collision
def row(t, A, B, tcp=(0., 0., .3), qA=(1, 0, 0, 0), forces=True, **f):
    r = {'t': t, 'tcp': list(tcp), 'tcp_q': list(qx(180)), 'finger_qpos': [.03, .03], 'cubeA': list(A), 'cubeA_q': list(qA), 'cubeB': list(B), 'cubeB_q': [1, 0, 0, 0]}
    if forces: r.update({'f_A_B': 0., 'f_A_left': 0., 'f_A_right': 0., 'f_A_table': 0., 'f_B_left': 0., 'f_B_right': 0., **f})
    return r


def test_proximity_contact_and_unrecorded_pairs_are_kept_apart():
    touching_no_force = SA.row_metrics(row(1, (.04, 0, .02), (0, 0, .02)))
    assert abs(touching_no_force['d_A_B']) < 1e-12 and touching_no_force['proximity_A_B'] and touching_no_force['observed_contact_A_B'] is False
    with_force = SA.row_metrics(row(1, (.04, 0, .02), (0, 0, .02), f_A_B=3.))
    assert with_force['observed_contact_A_B'] is True and with_force['force_A_B'] == 3.
    near = SA.row_metrics(row(1, (.044, 0, .02), (0, 0, .02))); assert near['proximity_A_B'] and near['d_A_B'] > 0
    far = SA.row_metrics(row(1, (.1, 0, .02), (0, 0, .02))); assert not far['proximity_A_B'] and far['observed_contact_A_B'] is False
    for p in SA.UNRECORDED_PAIRS: assert with_force[f'observed_contact_{p}'] is None              # never 'no contact'
    no_force_fields = SA.row_metrics(row(1, (.04, 0, .02), (0, 0, .02), forces=False))
    assert all(no_force_fields[f'observed_contact_{p}'] is None for p in SA.RECORDED_PAIRS) and no_force_fields['proximity_A_B']


def test_gripper_footprint_is_the_union_of_its_shapes_not_their_hull():
    # fingers 60 mm apart around a small square between them: the hull of the gripper would cover it, the shapes do not
    g = G.gripper([0, 0, .1], qx(180), [.03, .03]); fl = [G.footprint(s) for k, s in g.items() if k != 'hand']
    probe_sq = np.array([[-.003, -.003], [.003, -.003], [.003, .003], [-.003, .003]])
    assert G.overlap_area(G.footprint_many(list(g.values())), probe_sq) > 0 and G.union_overlap_area(fl, probe_sq) == 0
    assert G.union_gap(fl, probe_sq) > .01
    sq = G.footprint(G.cube([0, 0, .02], [1, 0, 0, 0])); half = np.array([[-.02, -.02], [0, -.02], [0, .02], [-.02, .02]])
    assert math.isclose(G.union_overlap_area([half], sq), .02 * .04, rel_tol=.02)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., 'x', True])
def test_an_invalid_recorded_force_is_an_error_never_no_contact(bad):
    with pytest.raises(ValueError, match='not a finite non-negative number'): SA.row_metrics(row(1, (.04, 0, .02), (0, 0, .02), f_A_B=bad))


def test_every_stored_distance_is_exact_even_far_apart():
    m = SA.row_metrics(row(1, (.3, 0, .02), (0, 0, .02))); assert all(m[f'exact_{p}'] for p in SA.DIST_PAIRS)
    assert math.isclose(m['d_A_B'], .26, abs_tol=1e-12)                       # 300 mm centres - 40 mm: exact, not an AABB bound


def test_penetration_depth_and_axis():
    a = G.cube([0, 0, .02], [1, 0, 0, 0]); b = G.cube([.035, 0, .02], [1, 0, 0, 0]); d, n = G.sat_overlap(a, b, return_axis=True)
    assert math.isclose(d, .005, abs_tol=1e-12) and np.allclose(np.abs(n), [1, 0, 0]) and n[0] > 0   # oriented from A to B
    b2 = G.box(b.V.mean(0) + n * (d + 1e-4), np.eye(3), (.02,) * 3); assert G.signed_distance(a, b2) > 0


def test_a_pass_through_between_logged_steps_is_a_potential_collision_not_an_observed_one():
    r0 = row(1, (-.06, 0, .02), (0, 0, .02)); r1 = row(2, (.06, 0, .02), (0, 0, .02))       # both logged poses 20 mm clear; the straight path crosses the target
    m0, m1 = SA.row_metrics(r0), SA.row_metrics(r1); assert m0['d_A_B'] > .019 and m1['d_A_B'] > .019
    sc = SA.screen_interval(r0, r1, m0, m1); assert sc['A_B'] is not None and sc['A_B'] < 0
    slow = row(2, (-.059, 0, .02), (0, 0, .02)); ms = SA.row_metrics(slow); assert SA.screen_interval(r0, slow, m0, ms)['A_B'] is None   # motion too small to close the gap: not screened


# ---------------------------------------------------------------- 6 swept volume
@pytest.mark.parametrize('dist', [0., .02, .05])
def test_swept_volume_of_a_translating_box_matches_the_closed_form(dist):
    seq = [G.box([x, 0, .1], np.eye(3), (.02, .015, .01)) for x in np.linspace(0, dist, 41)]
    exact = .04 * .03 * .02 + dist * .03 * .02; v = G.swept_volume(seq, .001); assert abs(v - exact) / exact < .06


def test_interpolated_poses_end_at_the_logged_pose_and_are_labelled_assumed():
    P = G.interpolate_poses([0, 0, 0], [1, 0, 0, 0], [.01, 0, 0], qz(20)); assert len(P) == G.SIM_STEPS_PER_CONTROL
    assert np.allclose(P[-1][0], [.01, 0, 0]) and math.isclose(G.rotation_angle_deg(P[-1][1], qz(20)), 0, abs_tol=1e-6)
    assert 'ASSUMPTION' in G.interpolate_poses.__doc__ and SA.swept_occupancy(*_fake_noact())['status'] == 'NOT_APPLICABLE'


def _fake_noact():
    rec = {'scene': 1, 'triggered': True, 'trigger': {'step': 1}, 'cells': [{'arm': 'C0', 'acted': False, 'control_steps': 0}]}; return rec, {'trunk': []}, 'C0'


# ---------------------------------------------------------------- 7-10 on C1-format records (mock runtime)


@pytest.fixture(scope='module')
def c1():
    pytest.importorskip('torch')
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    from test_trajectory_audit import probe2
    from hci.simulation import Runtime
    from hci import vision_center_c1 as C1

    def probe3(rt, z):                                                    # probe2 + the fields the real probe also logs
        r = probe2(rt, z); return {**r, 'cubeA_q': [1., 0., 0., 0.], 'cubeB_q': [1., 0., 0., 0.], 'tcp_q': list(qx(180)), 'f_A_left': 0., 'f_A_right': 0., 'f_A_table': 0.}
    r = Runtime(VisionEnv2(), BiasPolicy((.028, .012)), CFG); rec, tr, fr = C1.collect_scene(r, 7, CFG, probe=probe3, arms=C1.BASE_ARMS, policy_id='train_seed_1')
    return json.loads(json.dumps(rec)), json.loads(json.dumps(tr)), {k: np.stack(v) for k, v in fr.items() if v}


def test_frames_map_to_the_steps_the_record_names(c1):
    from hci import spatial_figures as F
    from hci import trajectory_audit as T
    rec, tr, fr = c1; t0 = rec['trigger']['step']
    img, why = F.frame_at(rec, fr, 'C0', t0); assert np.array_equal(img, fr['trunk'][t0]) and 'shared prefix' in why
    for arm in (a for a in rec['arms'] if a in fr):                       # control frame j = observation of step trigger + j: frame 0 is the trigger frame
        assert np.array_equal(fr[arm][0], fr['trunk'][t0]) or np.array_equal(fr[arm][0], fr['trunk'][t0][:, fr['trunk'].shape[2] - fr[arm].shape[2]:])
    for c in rec['cells']:
        fa = c['frames_after_trigger']
        if fa.get('key'):
            q = fa['steps'][2]; img, why = F.frame_at(rec, fr, c['arm'], q); assert np.array_equal(img, fr[fa['key']][2]) and why == 'recorded RGB'
            gap = next((q for q in range(t0 + 1, c['steps'] + 1) if q not in fa['steps']), None)
            if gap is not None and not (c['arm'] in fr and gap - t0 < len(fr[c['arm']])): assert F.frame_at(rec, fr, c['arm'], gap)[0] is None
        else:
            assert fa['same_as'] == 'C0'; q = t0 + 1; assert np.array_equal(F.frame_at(rec, fr, c['arm'], q)[0], F.frame_at(rec, fr, 'C0', q)[0])
    assert F.frame_at(rec, None, 'C0', 3) == (None, 'NO RGB STORED for this scene')


def test_commanded_targets_are_per_step_and_kept_apart_from_measured_motion(c1):
    from hci import spatial_figures as F
    rec, tr, fr = c1; acted = [c for c in rec['cells'] if c.get('acted') and c['control_steps'] > 0]; assert acted
    c = acted[0]; arm = c['arm']; t0 = rec['trigger']['step']; R = F.rows_of(rec, tr, arm); cm = F.commanded(rec, tr, arm)
    assert len(cm) == c['control_steps']
    for j, (q, before, target) in enumerate(cm):
        assert q == t0 + j + 1 and np.allclose(before, R[t0 + j]['tcp'][:2]) and np.allclose(np.subtract(target, before), c['control']['steps'][j]['commanded_xy_m'])
    act = SA.actual_behaviour(rec, tr, arm); end = t0 + c['control_steps']
    assert np.allclose(act['measured_tcp_displacement_m'], np.subtract(R[end]['tcp'], R[t0]['tcp'])) and np.allclose(act['commanded_step_xy_m'], [x['commanded_xy_m'] for x in c['control']['steps']])
    assert not np.allclose(act['commanded_step_norm_sum_m'], act['measured_tcp_xy_displacement_norm_m'])         # commands are targets, not a path


def test_missing_or_absent_data_is_not_observable_or_not_applicable(c1):
    rec, tr, fr = c1; rq = SA.requirements_check(rec, tr)
    assert rq['fields_present_in_this_record']['arm_qpos'] is False and rq['fields_present_in_this_record']['commanded_xy_m'] is True
    assert {i['status'] for i in rq['items']} <= {'AVAILABLE', 'DERIVABLE_FROM_RECORDED_DATA', 'MISSING', 'REQUIRES_SIMULATOR_INSTRUMENTATION'}
    assert any(i['status'] == 'MISSING' and 'arm link' in i['item'] for i in rq['items'])
    for c in rec['cells']:
        e = SA.expected_vs_actual(rec, tr, c['arm'], SA.branch_metrics(rec, tr, c['arm']))
        assert set(e['expected']) == {x[0] for x in SA.EXPECTED} and all(v['status'] in SA.STATUS for v in e['expected'].values())
        if c['arm'] == 'C0' or not c.get('acted'): assert all(v['status'] == 'NOT_APPLICABLE' for k, v in e['expected'].items() if k != 'E9_abstain_equals_C0')
        if c['arm'] != 'C0' and not c.get('acted'):
            assert e['expected']['E9_abstain_equals_C0']['status'] == 'PASS' and e['expected']['E9_abstain_equals_C0']['fields_compared'] >= 15
            for field, change in (('finger_qpos', lambda v: [v[0] + 1e-3, v[1]]), ('action', lambda v: v[:-1] + [-v[-1] or 1.]), ('f_B_left', lambda v: 2.)):
                t2 = copy.deepcopy(tr); r_ = t2[c['arm']][-1]; r_[field] = change(r_[field])    # one field at one step differs from C0
                e2 = SA.same_as_c0(rec, t2, c['arm']); assert e2['status'] == 'FAIL' and e2['differing_fields'] == [field], field
            t3 = copy.deepcopy(tr); t3['C0'][-1]['extra_none'] = None                 # present (None) on one side, absent on the other
            t3[c['arm']][0]['f_A_B'] = float('nan'); e3 = SA.same_as_c0(rec, t3, c['arm'])
            assert e3['status'] == 'FAIL' and e3['differing_fields'] == ['extra_none', 'f_A_B']
    moved = next(c for c in rec['cells'] if c.get('acted') and c['control_steps'] > 0); r2 = copy.deepcopy(rec); next(x for x in r2['cells'] if x['arm'] == moved['arm']).pop('first_inference_step')
    assert SA.expected_vs_actual(r2, tr, moved['arm'])['expected']['E8_policy_resumes_cleanly']['status'] == 'NOT_OBSERVABLE'
    # success never decides a status: flip the outcome, the expectations stay as they were
    r3 = copy.deepcopy(rec); x = next(x for x in r3['cells'] if x['arm'] == moved['arm']); x['y'] = 1 - x['y']
    a = SA.expected_vs_actual(rec, tr, moved['arm'])['expected']; b = SA.expected_vs_actual(r3, tr, moved['arm'])['expected']
    assert {k: v['status'] for k, v in a.items() if k != 'E10_policy_keeps_correction'} == {k: v['status'] for k, v in b.items() if k != 'E10_policy_keeps_correction'}


def test_same_records_give_the_same_metrics_figures_and_audit_outputs(c1, tmp_path):
    pytest.importorskip('matplotlib')
    from hci import spatial_figures as F
    rec, tr, fr = c1; arm = next(c['arm'] for c in rec['cells'] if c.get('acted') and c['control_steps'] > 0)
    a = json.dumps(SA.branch_metrics(rec, tr, arm), sort_keys=True); b = json.dumps(SA.branch_metrics(copy.deepcopy(rec), copy.deepcopy(tr), arm), sort_keys=True); assert a == b
    plt = F._plt(); outs = []
    for k in range(2):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80); P = F.draw_topdown(ax, rec, tr, arm, None, F.view_limits(rec, tr, [arm]), F.swept_footprint(rec, tr, arm))
        p = tmp_path / f'f{k}.png'; fig.savefig(p, metadata={'Software': None}); plt.close(fig); outs.append((p.read_bytes(), json.dumps(P, sort_keys=True, default=str)))
    assert outs[0] == outs[1]
    sw = F.swept_footprint(rec, tr, arm); t0 = rec['trigger']['step']; lims = F.view_limits(rec, tr, [arm]); n = []
    for s in (sw, None):                                                  # a frame at step t0 + 1 draws only the swept area up to t0 + 1
        fig, ax = plt.subplots(); F.draw_topdown(ax, rec, tr, arm, t0 + 1, lims, s); n.append(len(ax.patches)); plt.close(fig)
    assert n[0] - n[1] == sum(1 for q, _ in sw if q <= t0 + 1) < len(sw)


def test_the_metrics_command_on_a_c1_style_folder_is_deterministic_and_refuses_overwrite(c1, tmp_path):
    spec = importlib.util.spec_from_file_location('sa36t', Path('scripts/36_c1_spatial_audit.py')); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    spec = importlib.util.spec_from_file_location('c1_driver_sa', Path('scripts/34_vision_center_c1.py')); X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
    rec, tr, fr = c1; root = tmp_path / 'records'; X.Z.save_scene(root / 'pass_train_seed_1', 7, copy.deepcopy(rec), copy.deepcopy(tr), {k: list(v) for k, v in fr.items()})
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert M.main(['metrics', str(root), str(tmp_path / 'o1'), '--workers', '1']) == 0 and M.main(['metrics', str(root), str(tmp_path / 'o2'), '--workers', '1']) == 0
    for f in ('summary.json', 'expected_vs_actual.json', 'per_cell_summary.csv', 'contact_geometry_validation.json'):
        assert (tmp_path / 'o1' / f).read_bytes() == (tmp_path / 'o2' / f).read_bytes(), f
    with gzip.open(tmp_path / 'o1' / 'per_step_metrics.csv.gz', 'rt') as x, gzip.open(tmp_path / 'o2' / 'per_step_metrics.csv.gz', 'rt') as y: assert x.read() == y.read()
    s = json.loads((tmp_path / 'o1' / 'summary.json').read_text()); assert s['status'] == 'COMPLETE' and s['stages']['pass_train_seed_1']['triggered'] == 1
    with pytest.raises(SystemExit, match='Refused overwrite'): M.main(['metrics', str(root), str(tmp_path / 'o1'), '--workers', '1'])
    with pytest.raises(SystemExit, match='outside the records folder'): M.main(['metrics', str(root), str(root / 'out'), '--workers', '1'])
    assert {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    # a scene that cannot be read is an error on record, and the run is INCOMPLETE (exit 3), never dropped
    (root / 'pass_train_seed_1' / 'scenes' / '8.json').write_text('{broken')
    assert M.main(['metrics', str(root), str(tmp_path / 'o3'), '--workers', '1']) == 3
    e = json.loads((tmp_path / 'o3' / 'errors.json').read_text()); assert [x['seed'] for x in e] == [8] and json.loads((tmp_path / 'o3' / 'summary.json').read_text())['status'] == 'INCOMPLETE'
