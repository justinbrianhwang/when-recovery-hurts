"""The spatial audit of spatial-logger records (hci.spatial_links, scripts/38_c1_spatial_audit_v2.py): known geometric
answers, contact-list parsing, the S1-S6 items, the URDF / STL comparison, the swept volume, and the runner on C1-format
mock records. No simulator."""
import argparse, gzip, importlib.util, json, math, struct
from pathlib import Path
import numpy as np
import pytest
from hci import spatial_links as L
from hci import spatial_probe as SP

Q0 = [1., 0., 0., 0.]


def qz(deg): a = math.radians(deg) / 2; return [math.cos(a), 0., 0., math.sin(a)]


def boxd(h, p=(0, 0, 0), q=Q0): return {'type': 'PhysxCollisionShapeBox', 'local_pose': list(p) + list(q), 'half_size': list(h)}


def geometry():
    return {'panda_hand': {'pose_at_this_row': [0.] * 3 + Q0, 'shapes': [boxd((.02, .04, .01), (0, 0, .01))]}, 'panda_link3': {'pose_at_this_row': [0.] * 3 + Q0, 'shapes': [boxd((.03, .03, .05))]},
            'panda_leftfinger': {'pose_at_this_row': [0.] * 3 + Q0, 'shapes': [boxd((.005, .005, .01))]}, 'camera_link': {'pose_at_this_row': [0.] * 3 + Q0, 'shapes': [boxd((.01, .05, .01))]},
            'panda_hand_tcp': {'pose_at_this_row': [0.] * 3 + Q0, 'shapes': []}, 'cubeA': {'pose_at_this_row': [0.] * 3 + Q0, 'shapes': [boxd((.02,) * 3)]},
            'cubeB': {'pose_at_this_row': [0.] * 3 + Q0, 'shapes': [boxd((.02,) * 3)]}, 'table': {'pose_at_this_row': [0, 0, -.5] + Q0, 'shapes': [boxd((1., 1., .5))]},
            'scene-0_ground': {'pose_at_this_row': [0, 0, -1.] + Q0, 'shapes': [{'type': 'PhysxCollisionShapePlane', 'local_pose': [0.] * 3 + Q0}]}}


def row(t=1, hand=(.1, 0, .09), link3=(-.3, 0, .4), contacts=(), A=(.1, 0, .065), B=(.1, 0, .02)):
    return {'t': t, 'spatial_schema': SP.SCHEMA, 'sim_timestep_s': .01, 'tcp': list(hand), 'tcp_q': Q0, 'cubeA': list(A), 'cubeA_q': Q0, 'cubeB': list(B), 'cubeB_q': Q0,
            'link_pose': {'panda_hand': list(hand) + Q0, 'panda_link3': list(link3) + Q0, 'panda_leftfinger': [hand[0], hand[1] + .03, hand[2] - .03] + Q0, 'camera_link': [hand[0] - .05, hand[1], hand[2] + .02] + Q0, 'panda_hand_tcp': list(hand) + Q0},
            'contacts': list(contacts)}


def c(a, b, imp=(0, 0, 0), sep=0., sa=0, sb=0): return {'a': a, 'b': b, 'shape_a': sa, 'shape_b': sb, 'points': [[0, 0, 0, 0, 0, 1, *imp, sep]]}


def test_logged_shapes_are_posed_by_the_logged_poses():
    g = geometry(); W = L.world(row(), g)
    hand = W['panda_hand'][0]; assert np.allclose(hand.V.mean(0), [.1, 0, .1]) and np.allclose(hand.V.max(0) - hand.V.min(0), [.04, .08, .02])
    r = row(); r['link_pose']['panda_hand'] = [0, 0, 0] + qz(90); h2 = L.world(r, g)['panda_hand'][0]; assert np.allclose(h2.V.max(0) - h2.V.min(0), [.08, .04, .02])      # rotated 90 deg
    assert 'panda_hand_tcp' not in W and 'scene-0_ground' not in W and np.isclose(W['table'][0].V[:, 2].max(), 0.)
    assert L.gap(W['panda_hand'], W['cubeA']) == pytest.approx(.09 - .085) and L.gap(W['cubeA'], W['cubeB']) == pytest.approx(.005)


def test_contact_pairs_keep_observed_contact_and_proximity_apart():
    r = row(contacts=[c('panda_hand', 'cubeB', imp=(0, 0, .02)), c('cubeB', 'panda_hand', imp=(0, 0, -.01)), c('panda_link3', 'table', sep=.006), c('cubeA', 'cubeB', imp=(0, 0, .001))])
    p = L.contact_pairs(r); e = p[('cubeB', 'panda_hand')]
    assert e['observed'] and e['force_N'] == pytest.approx(3.) and e['points'] == 2 and p[('panda_link3', 'table')]['proximity_only'] and p[('panda_link3', 'table')]['min_separation_m'] == pytest.approx(.006)
    assert L.category('panda_hand', 'cubeB') == 'hand-cubeB' and L.category('table', 'panda_link3') == 'arm_links-table' and L.category('camera_link', 'cubeA') == 'camera-cubeA'
    assert L.category('panda_link3', 'scene-0_ground') == 'arm_links-other' and L.category('cubeB', 'table') == 'cubeB-table' and L.category('panda_hand', 'panda_leftfinger') == 'robot-self'
    m = L.row_metrics(r, geometry()); assert m['observed_contact_hand-cubeB'] and not m['observed_contact_arm_links-table'] and m['listed_arm_links-table'] and m['observed_contact_cubeA-cubeB']


def test_row_metrics_give_exact_gaps_per_group_and_compare_physx_separation():
    g = geometry(); r = row(contacts=[c('panda_link3', 'table', sep=.006)], link3=(-.3, 0, .056))
    m = L.row_metrics(r, g)
    assert m['d_hand_cubeA'] == pytest.approx(.005) and m['d_arm_links_table'] == pytest.approx(.006) and m['d_hand_cubeB'] == pytest.approx(.09 - .04)
    assert m['d_robot_cubeB'] == min(m[f'd_{k}_cubeB'] for k in L.GROUPS) and m['robot_lowest_z'] == pytest.approx(.006) and m['table_top_z'] == pytest.approx(0.)
    s = m['separation_vs_reconstruction']; assert len(s) == 1 and s[0]['reconstructed_m'] == pytest.approx(.006) and not s[0]['observed']
    assert set(m['link_aabb']) == {'panda_hand', 'panda_link3', 'panda_leftfinger', 'camera_link'}


def rec_of(rows, acted=True, k=3, trig=2):
    return {'scene': 1, 'triggered': True, 'trigger': {'step': trig}, 'initial_state': {'collision_geometry': geometry(), **rows[0]},
            'cells': [{'arm': 'C0', 'acted': False, 'control_steps': 0}, {'arm': 'X', 'acted': acted, 'control_steps': k if acted else 0}]}


def test_spatial_items_pass_fail_not_applicable_and_not_observable():
    trunk = [row(t) for t in (1, 2)]; arm = [row(t, hand=(.1, 0, .09 + .001 * t)) for t in (3, 4, 5, 6)]
    tr = {'trunk': trunk, 'X': arm, 'C0': arm}; b = L.branch(rec_of(trunk), tr, 'X'); s = L.spatial_items(b, True)
    assert all(s[k]['status'] == 'PASS' for k in list(s)[:4]) and s['S4_robot_clear_of_target_at_logged_poses']['min_distance_m'] > 0
    assert s['S5_recorded_contacts_agree_with_geometry']['status'] == s['S6_physx_separation_agrees_with_geometry']['status'] == 'NOT_OBSERVABLE'      # no contact listed: nothing compared (review a1)
    assert 'EXPLORATORY' in s['S1_no_robot_contact_with_target_during_move']['threshold']
    arm2 = [dict(r) for r in arm]; arm2[1] = row(4, contacts=[c('panda_hand', 'cubeB', imp=(0, 0, .01))]); arm2[2] = row(5, contacts=[c('panda_link3', 'table', imp=(0, 0, .01))])
    s2 = L.spatial_items(L.branch(rec_of(trunk), {'trunk': trunk, 'X': arm2}, 'X'), True)
    assert s2['S1_no_robot_contact_with_target_during_move']['status'] == 'FAIL' and s2['S1_no_robot_contact_with_target_during_move']['steps'] == [4]
    assert s2['S2_no_robot_contact_with_table_during_move']['status'] == 'FAIL' and s2['S5_recorded_contacts_agree_with_geometry']['status'] == 'FAIL'     # a contact 4.5 cm away is inconsistent
    s3 = L.spatial_items(L.branch(rec_of(trunk, acted=False), {'trunk': trunk, 'X': arm}, 'X'), False); assert s3['S1_no_robot_contact_with_target_during_move']['status'] == 'NOT_APPLICABLE'
    old = [{k: v for k, v in r.items() if k not in SP.STEP_FIELDS} for r in arm]
    assert L.branch(rec_of(trunk), {'trunk': trunk, 'X': old}, 'X') is None and all(v['status'] == 'NOT_OBSERVABLE' for v in L.spatial_items(None, True).values())
    req = L.requirements_v2({'initial_state': {}}, {'trunk': old}); assert not req['spatial_logger_record'] and {i['status'] for i in req['items'][:-2]} == {'NOT_OBSERVABLE'}
    assert req['items'][-2]['status'] == 'REQUIRES_SIMULATOR_INSTRUMENTATION'


def test_physx_separation_disagreeing_with_geometry_fails_s6():
    trunk = [row(t) for t in (1, 2)]; arm = [row(t, link3=(-.3, 0, .056), contacts=[c('panda_link3', 'table', sep=.006 + (.003 if t == 4 else 0))]) for t in (3, 4, 5)]
    s = L.spatial_items(L.branch(rec_of(trunk, k=2), {'trunk': trunk, 'X': arm}, 'X'), True)['S6_physx_separation_agrees_with_geometry']
    assert s['status'] == 'FAIL' and s['max_abs_difference_m'] == pytest.approx(.003) and s['n_pairs'] == 3


def test_robot_swept_volume_of_a_translating_link_matches_the_closed_form():
    g = {'panda_link3': {'pose_at_this_row': [0.] * 7, 'shapes': [boxd((.05, .05, .05))]}}
    rows = [{'link_pose': {'panda_link3': [x, 0, .5] + Q0}} for x in (0., .05, .1)]
    v = L.robot_swept_volume(rows, g, voxel=.005); exact = .1 * .1 * (.1 + .1)
    assert abs(v / exact - 1) < .08 and L.robot_swept_volume(rows[:1], g, voxel=.005) == pytest.approx(.001, rel=.08)


def stl_bytes(V, T):
    b = bytearray(80) + struct.pack('<I', len(T))
    for t in T: b += struct.pack('<12fH', 0, 0, 0, *V[t[0]], *V[t[1]], *V[t[2]], 0)
    return bytes(b)


def test_logged_shapes_are_compared_with_the_fixed_urdf_and_stl(tmp_path):
    from hci import spatial_geometry as G
    V = (G._SIGNS * .03).tolist(); T = G._BOX_TRI.tolist(); (tmp_path / 'm').mkdir(); (tmp_path / 'm' / 'l1.stl').write_bytes(stl_bytes(V, T))
    (tmp_path / 'r.urdf').write_text('<robot name="r"><link name="l1"><collision><origin rpy="0 0 0" xyz="0 0 0.1"/><geometry><mesh filename="m/l1.stl"/></geometry></collision></link>'
                                     '<link name="f"><collision><origin rpy="0.5235987755982988 0 0" xyz="0 0.01 0"/><geometry><box size="0.02 0.01 0.03"/></geometry></collision></link></robot>')
    a = math.radians(30) / 2; geo = {'l1': {'shapes': [{'type': 'PhysxCollisionShapeConvexMesh', 'local_pose': [0, 0, .1] + Q0, 'scale': [1., 1., 1.], 'vertices': V, 'triangles': T}]},
                                     'f': {'shapes': [boxd((.01, .005, .015), (0, .01, 0), [math.cos(a), math.sin(a), 0, 0])]}}
    r = L.geometry_against_urdf(geo, tmp_path / 'r.urdf'); assert r['pass'] and len(r['rows']) == 2 and r['rows'][0]['volume_ratio_physx_to_stl_hull'] == pytest.approx(1.)
    bad = json.loads(json.dumps(geo)); bad['l1']['shapes'][0]['vertices'] = (np.asarray(V) * 1.05).tolist(); assert not L.geometry_against_urdf(bad, tmp_path / 'r.urdf')['pass']
    bad = json.loads(json.dumps(geo)); bad['f']['shapes'][0]['half_size'] = [.01, .005, .016]; assert not L.geometry_against_urdf(bad, tmp_path / 'r.urdf')['pass']


# ---------------------------------------------------------------- the runner on C1-format records
def spatial_mock(rt, z):
    """probe2 + a consistent spatial record: the hand as a box at the TCP, both cubes and the table as their boxes."""
    from test_trajectory_audit import probe2
    r = probe2(rt, z); hand = list(r['tcp'])
    new = {'spatial_schema': SP.SCHEMA, 'elapsed_steps': rt.t, 'sim_time_s': rt.t / 20, 'sim_timestep_s': .01, 'sim_substeps_per_control': 5., 'robot_qpos': [0.] * 9, 'robot_qvel': [0.] * 9,
           'link_pose': {'panda_hand': [hand[0], hand[1], hand[2] + .03] + Q0, 'panda_link3': [-.3, 0, .4] + Q0}, 'contacts': [c('cubeB', 'table', imp=(0, 0, .002))], 'f_hand_A': 0., 'f_hand_B': 0., 'f_B_table': .2,
           'f_hand_table': 0., 'f_fingerL_table': 0., 'f_fingerR_table': 0., 'f_links_A': {}, 'f_links_B': {}, 'f_links_table': {}, 'f_links_other': {}, 'cubeA_q': Q0, 'cubeB_q': Q0}
    if z is None: new.update(collision_geometry=geometry(), robot_joint_names=['j'] * 9)
    return {**r, **new}


@pytest.fixture(scope='module')
def c1_folder(tmp_path_factory):
    torch = pytest.importorskip('torch')
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    from test_trajectory_audit import probe2
    from hci.simulation import Runtime
    from hci import vision_center_c1 as C1
    spec = importlib.util.spec_from_file_location('c1_driver_sl', Path('scripts/34_vision_center_c1.py')); X34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X34)
    root = tmp_path_factory.mktemp('c1sp'); new = root / 'new'; old = root / 'old'
    for d, pr in ((new, spatial_mock), (old, probe2)):
        r = Runtime(VisionEnv2(), BiasPolicy((.028, .012)), CFG); rec, tr, fr = C1.collect_scene(r, 7, CFG, probe=pr, arms=C1.BASE_ARMS, policy_id='train_seed_1'); X34.Z.save_scene(d, 7, rec, tr, fr)
    spec = importlib.util.spec_from_file_location('sa38t', Path('scripts/38_c1_spatial_audit_v2.py')); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    return M, new, old, root


def test_the_runner_is_deterministic_refuses_overwrite_and_marks_old_records_not_observable(c1_folder):
    M, new, old, root = c1_folder; before = {p: p.read_bytes() for p in new.rglob('*') if p.is_file()}
    ns = lambda rec, out, plan=None: argparse.Namespace(records=str(rec), replay=None, phase_b=None, urdf=None, plan=plan, out=str(out))
    assert M.cmd_metrics(ns(new, root / 'm1')) == 0 and M.cmd_metrics(ns(new, root / 'm2')) == 0
    for f in ('spatial_summary.json', 'contact_geometry_validation_v2.json'): assert (root / 'm1' / f).read_bytes() == (root / 'm2' / f).read_bytes()
    with gzip.open(root / 'm1' / 'per_step_spatial_metrics.csv.gz', 'rt') as a, gzip.open(root / 'm2' / 'per_step_spatial_metrics.csv.gz', 'rt') as b: assert a.read() == b.read()
    s = json.loads((root / 'm1' / 'spatial_summary.json').read_text())['7']; assert s['requirements']['spatial_logger_record'] and s['arms']
    acted = [a for a, v in s['arms'].items() if v['control_steps'] > 0]; assert acted and all(s['arms'][a]['spatial_items']['S1_no_robot_contact_with_target_during_move']['status'] in ('PASS', 'FAIL') for a in acted)
    with pytest.raises(SystemExit, match='Refused overwrite'): M.cmd_metrics(ns(new, root / 'm1'))
    with pytest.raises(SystemExit, match='inside or around'): M.cmd_metrics(ns(new, new / 'out'))
    assert {p: p.read_bytes() for p in new.rglob('*') if p.is_file()} == before                     # the input is never changed
    assert M.cmd_metrics(ns(old, root / 'm3')) == 0; o = json.loads((root / 'm3' / 'spatial_summary.json').read_text())['7']
    assert not o['requirements']['spatial_logger_record'] and all(v == {'status': 'NOT_OBSERVABLE', 'why': 'no spatial fields'} for v in o['arms'].values())
    assert 'WITHOUT_SPATIAL_FIELDS' in json.loads((root / 'm3' / 'MANIFEST.json').read_text())['status']                        # review a12: visible in the status
    import hashlib; plan = root / 'plan_sp.json'; plan.write_text(json.dumps({'seed_registry': {'range': [7, 7]}, 'spatial_logging': {}}))
    import shutil; d = root / 'old_bound'; shutil.copytree(old, d); r = json.loads((d / 'scenes' / '7.json').read_text()); r['plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest(); (d / 'scenes' / '7.json').write_text(json.dumps(r))
    assert M.cmd_metrics(argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=str(plan), out=str(root / 'm14'))) == 3     # a spatial-logging plan requires the fields
    assert 'has none' in json.loads((root / 'm14' / 'errors.json').read_text())['7']['error']
    r['triggered'] = False; r['cells'] = [{'arm': c['arm'], 'no_trigger': True} for c in r['cells']]; (d / 'scenes' / '7.json').write_text(json.dumps(r))      # review a14: a scene that never triggered
    assert M.cmd_metrics(argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=str(plan), out=str(root / 'm16'))) == 3
    assert 'trunk has none' in json.loads((root / 'm16' / 'errors.json').read_text())['7']['error']
    empty = root / 'empty'; (empty / 'scenes').mkdir(parents=True); assert M.cmd_metrics(ns(empty, root / 'm4')) == 3          # review a3: no scene is never COMPLETE
    assert json.loads((root / 'm4' / 'MANIFEST.json').read_text())['status'] == 'INCOMPLETE'


def test_review_a1_each_shape_pair_carries_its_own_contact_flag_and_no_comparison_is_not_observable():
    g = geometry(); g['panda_link3']['shapes'].append(boxd((.01, .01, .01), (0, 0, .3)))         # a second shape of the same link, 30 cm away
    trunk = [row(t) for t in (1, 2)]
    arm = [row(t, link3=(-.3, 0, .05), contacts=[c('panda_link3', 'table', imp=(0, 0, .01), sep=0., sa=0), c('panda_link3', 'table', sep=.02, sa=1)]) for t in (3, 4, 5)]      # shape 0 touches the table top
    rec = rec_of(trunk, k=2); rec['initial_state']['collision_geometry'] = g
    m = L.row_metrics(arm[0], g); flags = {s['shape_a']: s['observed'] for s in m['separation_vs_reconstruction']}
    assert flags == {0: True, 1: False}                                                               # the far shape is listed, not observed
    s5 = L.spatial_items(L.branch(rec, {'trunk': trunk, 'X': arm}, 'X'), True)['S5_recorded_contacts_agree_with_geometry']; assert s5['status'] == 'PASS' and s5['n_pairs'] == 3
    bare = [row(t) for t in (3, 4, 5)]
    s = L.spatial_items(L.branch(rec_of(trunk, k=2), {'trunk': trunk, 'X': bare}, 'X'), True)
    assert s['S5_recorded_contacts_agree_with_geometry']['status'] == 'NOT_OBSERVABLE' and s['S6_physx_separation_agrees_with_geometry']['status'] == 'NOT_OBSERVABLE'


def test_review_a5_contacts_with_unreconstructed_shapes_make_s5_not_observable():
    trunk = [row(t) for t in (1, 2)]
    arm = [row(t, link3=(-.3, 0, .05), contacts=[c('panda_link3', 'table', imp=(0, 0, .01)), c('panda_link3', 'scene-0_ground', imp=(0, 0, .01))]) for t in (3, 4, 5)]
    m = L.row_metrics(arm[0], geometry()); assert [x['b'] for x in m['contact_shape_pairs_not_reconstructed']] == ['scene-0_ground']
    s5 = L.spatial_items(L.branch(rec_of(trunk, k=2), {'trunk': trunk, 'X': arm}, 'X'), True)['S5_recorded_contacts_agree_with_geometry']
    assert s5['status'] == 'NOT_OBSERVABLE' and s5['n_not_reconstructed'] == 3 and s5['n_pairs'] == 3


def test_review_a5_a_scene_with_an_error_record_makes_the_audit_incomplete(c1_folder):
    import shutil
    M, new, old, root = c1_folder; d = root / 'with_error'; shutil.copytree(new, d); (d / 'scenes' / '8.error.json').write_text('{"scene": 8, "completed": false}')
    ns = argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=None, out=str(root / 'm5'))
    assert M.cmd_metrics(ns) == 3; man = json.loads((root / 'm5' / 'MANIFEST.json').read_text())
    assert man['status'] == 'INCOMPLETE' and '8' in json.loads((root / 'm5' / 'errors.json').read_text())


def test_review_a6_a_contradicting_separation_fails_s6_and_a_definite_violation_fails_s5():
    trunk = [row(t) for t in (1, 2)]
    pen = [row(t, link3=(-.3, 0, .045), contacts=[c('panda_link3', 'table', sep=.0005)]) for t in (3, 4, 5)]      # PhysX: 0.5 mm apart; reconstruction: 5 mm overlap
    s6 = L.spatial_items(L.branch(rec_of(trunk, k=2), {'trunk': trunk, 'X': pen}, 'X'), True)['S6_physx_separation_agrees_with_geometry']
    assert s6['status'] == 'FAIL' and s6['n_pairs'] == 3 and s6['max_abs_difference_m'] == pytest.approx(.0055)
    mix = [row(t, link3=(-.3, 0, .4), contacts=[c('panda_hand', 'cubeB', imp=(0, 0, .01)), c('panda_link3', 'scene-0_ground', imp=(0, 0, .01))]) for t in (3, 4, 5)]
    s5 = L.spatial_items(L.branch(rec_of(trunk, k=2), {'trunk': trunk, 'X': mix}, 'X'), True)['S5_recorded_contacts_agree_with_geometry']
    assert s5['status'] == 'FAIL' and s5['n_violations'] == 3 and s5['n_not_reconstructed'] == 3


def test_review_a7_separated_pairs_with_unreconstructed_shapes_make_s6_not_observable():
    trunk = [row(t) for t in (1, 2)]
    arm = [row(t, link3=(-.3, 0, .056), contacts=[c('panda_link3', 'table', sep=.006), c('panda_link3', 'scene-0_ground', sep=.01)]) for t in (3, 4, 5)]
    s6 = L.spatial_items(L.branch(rec_of(trunk, k=2), {'trunk': trunk, 'X': arm}, 'X'), True)['S6_physx_separation_agrees_with_geometry']
    assert s6['status'] == 'NOT_OBSERVABLE' and s6['n_not_reconstructed'] == 3 and s6['n_pairs'] == 3


def test_review_a8_a_missing_arm_trace_makes_the_audit_incomplete(c1_folder):
    import shutil
    M, new, old, root = c1_folder; d = root / 'missing_arm'; shutil.copytree(new, d)
    rec = json.loads((d / 'scenes' / '7.json').read_text()); tp = d / rec['trace']
    with gzip.open(tp, 'rt', encoding='utf-8') as f: data = json.load(f)
    gone = rec['cells'][-1]['arm']; data['traces'].pop(gone); tp.unlink()
    with gzip.open(tp, 'wt', encoding='utf-8') as f: json.dump(data, f)
    import hashlib; rec['trace_sha256'] = hashlib.sha256(tp.read_bytes()).hexdigest(); (d / 'scenes' / '7.json').write_text(json.dumps(rec))      # a consistent record whose trace lacks one arm
    assert M.cmd_metrics(argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=None, out=str(root / 'm6'))) == 3
    err = json.loads((root / 'm6' / 'errors.json').read_text()); assert gone in err['7']['error'] and json.loads((root / 'm6' / 'MANIFEST.json').read_text())['status'] == 'INCOMPLETE'


def test_review_a9_a_changed_trace_or_a_missing_scene_of_the_plan_is_never_complete(c1_folder):
    import shutil
    M, new, old, root = c1_folder; ns = lambda d, out, plan=None: argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=plan, out=str(root / out))
    d = root / 'changed'; shutil.copytree(new, d); rec = json.loads((d / 'scenes' / '7.json').read_text()); tp = d / rec['trace']
    with gzip.open(tp, 'rt', encoding='utf-8') as f: data = json.load(f)
    data['traces']['trunk'][3]['cubeB'][0] += 1e-3; tp.unlink()
    with gzip.open(tp, 'wt', encoding='utf-8') as f: json.dump(data, f)
    assert M.cmd_metrics(ns(d, 'm7')) == 3 and 'sha256' in json.loads((root / 'm7' / 'errors.json').read_text())['7']['error']
    import hashlib
    def bound_copy(name, plan):                                           # the records of a run of `plan` (plan_sha256 as the driver writes it)
        d2 = root / name; shutil.copytree(new, d2); r = json.loads((d2 / 'scenes' / '7.json').read_text()); r['plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest()
        (d2 / 'scenes' / '7.json').write_text(json.dumps(r)); return d2
    plan = root / 'plan.json'; plan.write_text(json.dumps({'seed_registry': {'range': [7, 9]}}))
    assert M.cmd_metrics(ns(bound_copy('b1', plan), 'm8', str(plan))) == 3; e = json.loads((root / 'm8' / 'errors.json').read_text()); assert set(e) == {'8', '9'}
    plan.write_text(json.dumps({'seed_registry': {'range': [7, 7]}})); assert M.cmd_metrics(ns(bound_copy('b2', plan), 'm9', str(plan))) == 0
    assert json.loads((root / 'm9' / 'MANIFEST.json').read_text())['status'] == 'COMPLETE'
    assert M.cmd_metrics(ns(new, 'm11', str(plan))) == 3 and 'another plan' in json.loads((root / 'm11' / 'errors.json').read_text())['7']['error']      # review a10
    plan.write_text(json.dumps({'seed_registry': {'range': [8, 9]}})); assert M.cmd_metrics(ns(bound_copy('b3', plan), 'm12', str(plan))) == 3
    assert 'outside the plan' in json.loads((root / 'm12' / 'errors.json').read_text())['7']['error']
    d4 = bound_copy('b4', plan); r = json.loads((d4 / 'scenes' / '7.json').read_text()); r['completed'] = False; (d4 / 'scenes' / '7.json').write_text(json.dumps(r))
    plan.write_text(json.dumps({'seed_registry': {'range': [7, 7]}})); r['plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest(); (d4 / 'scenes' / '7.json').write_text(json.dumps(r))
    assert M.cmd_metrics(ns(d4, 'm13', str(plan))) == 3 and 'not a completed record' in json.loads((root / 'm13' / 'errors.json').read_text())['7']['error']
    assert M.cmd_metrics(ns(new, 'm10')) == 0 and json.loads((root / 'm10' / 'MANIFEST.json').read_text())['status'].startswith('COMPLETE_FOR_THE_SCENES_FOUND')


def test_e1_e10_statuses_are_read_from_the_unchanged_earlier_audit():
    spec = importlib.util.spec_from_file_location('sa38e', Path('scripts/38_c1_spatial_audit_v2.py')); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    ev = {'scene': 1, 'arm': 'X', 'acted': True, 'control_steps': 3, 'expected': {'E1_alignment_error_reduced': {'status': 'PASS'}, 'E7_target_not_moved': {'status': 'FAIL', 'x': 1}}, 'actual': {}}
    assert M.e_status(ev) == {'E1_alignment_error_reduced': 'PASS', 'E7_target_not_moved': 'FAIL'}
    with pytest.raises(ValueError, match='no E1-E10'): M.e_status({'scene': 1, 'expected': {}})


def test_review_a13_a_scene_that_fails_in_a_later_arm_adds_nothing_to_the_aggregates(c1_folder, monkeypatch):
    M, new, old, root = c1_folder; real = L.branch; seen = []
    def flaky(rec, tr, arm, geometry=None):
        seen.append(arm)
        if len(seen) == 3: raise RuntimeError('injected by the test in the third arm')
        return real(rec, tr, arm, geometry)
    monkeypatch.setattr(M.L, 'branch', flaky)
    assert M.cmd_metrics(argparse.Namespace(records=str(new), replay=None, phase_b=None, urdf=None, plan=None, out=str(root / 'm15'))) == 3
    with gzip.open(root / 'm15' / 'per_step_spatial_metrics.csv.gz', 'rt') as f: assert len(f.read().strip().splitlines()) == 1        # the header only
    v = json.loads((root / 'm15' / 'contact_geometry_validation_v2.json').read_text()); assert v['observed_shape_pair_steps'] == 0 and v['separated_shape_pair_steps'] == 0


def test_review_a15_a_scene_that_never_triggered_is_audited_as_its_trunk(c1_folder):
    import shutil
    M, new, old, root = c1_folder; d = root / 'untriggered'; shutil.copytree(new, d)
    r = json.loads((d / 'scenes' / '7.json').read_text()); r['triggered'] = False; r['trigger'] = None; r['cells'] = [{'arm': c['arm'], 'no_trigger': True} for c in r['cells']]
    (d / 'scenes' / '7.json').write_text(json.dumps(r))
    assert M.cmd_metrics(argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=None, out=str(root / 'm17'))) == 0
    sc = json.loads((root / 'm17' / 'spatial_summary.json').read_text())['7']; assert list(sc['arms']) == ['trunk'] and sc['arms']['trunk']['control_steps'] == 0
    assert sc['arms']['trunk']['spatial_items']['S1_no_robot_contact_with_target_during_move']['status'] == 'NOT_APPLICABLE'
    with gzip.open(root / 'm17' / 'per_step_spatial_metrics.csv.gz', 'rt') as f: lines = f.read().strip().splitlines()
    assert len(lines) - 1 == len(r and json.load(gzip.open(d / r['trace'], 'rt'))['traces']['trunk']) and all(l.split(',')[1] == 'trunk' for l in lines[1:])


def test_review_a16_a_record_without_the_planned_arms_is_an_error(c1_folder):
    import shutil, hashlib
    M, new, old, root = c1_folder; base = json.loads((new / 'scenes' / '7.json').read_text()); arms = [c['arm'] for c in base['cells']]
    plan = root / 'plan_arms.json'; plan.write_text(json.dumps({'seed_registry': {'range': [7, 7]}, 'arm_registry': {'arms': arms}}))
    def run(name, cells):
        d = root / name; shutil.copytree(new, d); r = dict(base); r['cells'] = cells; r['plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest(); (d / 'scenes' / '7.json').write_text(json.dumps(r))
        return M.cmd_metrics(argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=str(plan), out=str(root / ('o_' + name))))
    assert run('arms_ok', base['cells']) == 0
    for name, cells in (('arms_empty', []), ('arms_short', base['cells'][:-1]), ('arms_twice', base['cells'] + base['cells'][-1:]), ('arms_order', base['cells'][1:] + base['cells'][:1])):
        assert run(name, cells) == 3, name
        assert 'planned arms' in json.loads((root / ('o_' + name) / 'errors.json').read_text())['7']['error'], name


def test_review_a17_a_shortened_trace_is_an_error_even_with_a_matching_hash(c1_folder):
    import shutil, hashlib
    M, new, old, root = c1_folder
    for name, cut in (('short_arm', lambda tr, rec: tr[rec['cells'][1]['arm']].pop()), ('short_trunk', lambda tr, rec: tr['trunk'].pop(3))):
        d = root / name; shutil.copytree(new, d); rec = json.loads((d / 'scenes' / '7.json').read_text()); tp = d / rec['trace']
        with gzip.open(tp, 'rt', encoding='utf-8') as f: data = json.load(f)
        cut(data['traces'], rec); tp.unlink()
        with gzip.open(tp, 'wt', encoding='utf-8') as f: json.dump(data, f)
        rec['trace_sha256'] = hashlib.sha256(tp.read_bytes()).hexdigest(); (d / 'scenes' / '7.json').write_text(json.dumps(rec))
        assert M.cmd_metrics(argparse.Namespace(records=str(d), replay=None, phase_b=None, urdf=None, plan=None, out=str(root / ('o_' + name)))) == 3, name
        assert 'does not cover steps' in json.loads((root / ('o_' + name) / 'errors.json').read_text())['7']['error'], name


def test_review_a18_the_whole_trunk_is_audited_once_and_an_early_disagreement_fails_s5():
    trunk = [row(t) for t in range(1, 16)]; trunk[1] = row(2, contacts=[c('panda_hand', 'cubeB', imp=(0, 0, .01))])     # step 2: a contact 5 cm away, long before trigger - 10
    rec = rec_of(trunk, k=2, trig=14); arm = [row(t) for t in (15, 16, 17)]
    b = L.branch(rec, {'trunk': trunk, 'X': arm}, 'X'); assert b['rows'][0]['t'] == 4                                    # the arm branch starts at trigger - 10
    t = L.branch(rec, {'trunk': trunk, 'X': arm}, 'trunk'); assert [r['t'] for r in t['rows']] == list(range(1, 16)) and {m['phase'] for m in t['metrics']} == {'trunk'}
    s = L.spatial_items(t, False); assert s['S5_recorded_contacts_agree_with_geometry']['status'] == 'FAIL' and s['S1_no_robot_contact_with_target_during_move']['status'] == 'NOT_APPLICABLE'
    spec = importlib.util.spec_from_file_location('sa38t18', Path('scripts/38_c1_spatial_audit_v2.py')); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    assert M.arms_of({'triggered': True, 'cells': [{'arm': 'C0'}, {'arm': 'X'}]}, {'trunk': [], 'C0': [], 'X': []}) == ['trunk', 'C0', 'X'] and M.arms_of({'triggered': False, 'cells': []}, {}) == ['trunk']


def test_review_a19_s5_is_one_sided_and_reports_the_penetration_depth():
    trunk = [row(t) for t in (1, 2)]
    deep = [row(t, link3=(-.3, 0, .02), contacts=[c('panda_link3', 'table', imp=(0, 0, .01))]) for t in (3, 4, 5)]      # link 3 sunk 30 mm into the table
    s5 = L.spatial_items(L.branch(rec_of(trunk, k=2), {'trunk': trunk, 'X': deep}, 'X'), True)['S5_recorded_contacts_agree_with_geometry']
    assert s5['status'] == 'PASS' and s5['max_penetration_m'] == pytest.approx(.03) and s5['n_overlapping'] == 3 and 'no threshold' in s5['penetration']
    assert 'one-sided' in dict(L.SPATIAL_ITEMS)['S5_recorded_contacts_agree_with_geometry']
