"""The logging-only spatial probe (hci.spatial_probe) and the driver's logging ON/OFF check, without a simulator: a fake
ManiSkill scene with recorded accessor calls, the mock C1 runtime of the other C1 tests, and the real C1 collector.
The same probe on the real pinned simulator: tests/test_spatial_probe_sim.py (skipped where ManiSkill is not installed)."""
import argparse, copy, importlib.util, json, random, sys, types
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from hci import spatial_probe as P
from hci import recovery_v1 as RV


# ---------------------------------------------------------------- a fake scene
class Pose:
    def __init__(self, p, q): self.p = torch.tensor([p], dtype=torch.float32); self.q = torch.tensor([q], dtype=torch.float32)


class SPose:                                                             # sapien.Pose: numpy float32
    def __init__(self, p, q): self.p = np.asarray(p, np.float32); self.q = np.asarray(q, np.float32)


class Box:
    def __init__(self, h, p=(0, 0, 0)): self.half_size = np.asarray(h, np.float32); self._lp = SPose(p, (1, 0, 0, 0))
    def get_local_pose(self): return self._lp


class Convex:
    def __init__(self): self.scale = np.ones(3, np.float32)
    def get_local_pose(self): return SPose((0, 0, .01), (1, 0, 0, 0))
    def get_vertices(self): return np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]], np.float32)
    def get_triangles(self): return np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.uint32)


class Plane:
    def get_local_pose(self): return SPose((0, 0, 0), (1, 0, 0, 0))


class Ent:
    def __init__(self, name, comp): self.name = name; self.components = [comp]

    @property
    def pose(self): return self.components[0].entity_pose                 # sapien.Entity.pose


class Comp:
    def __init__(self, shapes): self.collision_shapes = shapes


class Obj:
    def __init__(self, name, p=(0, 0, 0), q=(1, 0, 0, 0), shapes=None):
        self.name = name; comp = Comp(shapes if shapes is not None else [Box((.02, .02, .02))]); self.entity = Ent('scene-0_' + name, comp); comp.entity = self.entity
        comp.entity_pose = SPose(p, q); self._bodies = [comp]; self.pose = Pose(p, q)
        self.linear_velocity = torch.zeros(1, 3); self.angular_velocity = torch.zeros(1, 3)


class Log(list):
    def wrap(self, name, fn):
        def g(*a, **k): self.append(name); return fn(*a, **k)
        return g


def pt(imp, pos=(0, 0, 0), n=(0, 0, 1), sep=0.): return types.SimpleNamespace(impulse=np.array(imp, np.float32), position=np.array(pos, np.float32), normal=np.array(n, np.float32), separation=np.float32(sep))


def contact(a, b, imps, i=0, j=0, **k):
    ba, bb = a._bodies[0], b._bodies[0]
    return types.SimpleNamespace(bodies=(ba, bb), shapes=(ba.collision_shapes[i], bb.collision_shapes[j]), points=[pt(x, **k) for x in imps])


LINKS = [f'panda_link{i}' for i in range(9)] + ['panda_hand', 'panda_hand_tcp', 'panda_leftfinger', 'panda_rightfinger', 'camera_base_link', 'camera_link']


def fake(extra_contacts=()):
    log = Log(); A = Obj('cubeA', (.1, 0, .06)); B = Obj('cubeB', (.1, 0, .02)); table = Obj('table-workspace', shapes=[Box((.6, .6, .01), (0, 0, -.01))]); ground = Obj('ground', shapes=[Plane()])
    links = {n: Obj(n, (.01 * i, 0, .3), shapes=[] if n in ('panda_link8', 'panda_hand_tcp', 'camera_base_link') else ([Box((.01, .005, .02)), Box((.01, .005, .005))] if 'finger' in n else [Convex()]))
             for i, n in enumerate(LINKS)}
    dt = .01
    contacts = [contact(links['panda_hand'], B, [[0, 0, .03]]), contact(links['panda_leftfinger'], A, [[.2, 0, 0], [.1, 0, 0]], j=0, i=1), contact(A, B, [[0, 0, .005]]),
                contact(B, table, [[0, 0, .004]]), contact(links['panda_link5'], table, [[0, .01, 0]]), contact(table, links['panda_rightfinger'], [[0, 0, .002]]),
                contact(links['panda_link3'], ground, [[0, 0, .001]]), contact(links['panda_link6'], A, [[0, 0, 0]], sep=.004), *(extra_contacts(links, A, B, table) if extra_contacts else ())]
    from mani_skill_fake import sapien_utils as su

    def pairwise(a, b): return torch.tensor([su.get_pairwise_contact_impulse(contacts, a._bodies[0].entity, b._bodies[0].entity) / dt], dtype=torch.float32)
    scene = types.SimpleNamespace(gpu_sim_enabled=False, px=types.SimpleNamespace(timestep=dt), get_contacts=log.wrap('scene.get_contacts', lambda: list(contacts)),
                                  get_pairwise_contact_forces=log.wrap('scene.get_pairwise_contact_forces', pairwise),
                                  get_all_actors=log.wrap('scene.get_all_actors', lambda: [table.entity, A.entity, B.entity, ground.entity]))
    qpos = torch.tensor([[0., .1, 0., -2., 0., 2., .8, .02, .02]]); qvel = torch.full((1, 9), .5)
    joints = [types.SimpleNamespace(name=f'panda_joint{i}') for i in range(1, 8)] + [types.SimpleNamespace(name=f'panda_finger_joint{i}') for i in (1, 2)]
    robot = types.SimpleNamespace(get_qpos=log.wrap('robot.get_qpos', lambda: qpos.clone()), get_qvel=log.wrap('robot.get_qvel', lambda: qvel.clone()), get_links=log.wrap('robot.get_links', lambda: list(links.values())),
                                  get_active_joints=log.wrap('robot.get_active_joints', lambda: list(joints)))
    agent = types.SimpleNamespace(tcp=Obj('panda_hand_tcp', (.1, 0, .06), (0, 1, 0, 0)), robot=robot, finger1_link=links['panda_leftfinger'], finger2_link=links['panda_rightfinger'])
    flags = {k: torch.tensor([v]) for k, v in (('is_cubeA_grasped', True), ('is_cubeA_on_cubeB', False), ('is_cubeA_static', False), ('success', False))}
    u = types.SimpleNamespace(scene=scene, agent=agent, cubeA=A, cubeB=B, table_scene=types.SimpleNamespace(table=table), evaluate=log.wrap('env.evaluate', lambda: flags),
                              elapsed_steps=torch.tensor([5], dtype=torch.int32), control_freq=20, sim_freq=100)
    return u, log, contacts, dict(links=links, A=A, B=B, table=table, ground=ground, dt=dt)


@pytest.fixture(autouse=True)
def fake_maniskill(monkeypatch):
    """The upstream helpers (mani_skill/utils/sapien_utils.py at a4a4f92) as pure Python, and the SAPIEN shape types."""
    su = types.ModuleType('sapien_utils')

    def get_pairwise_contacts(contacts, a0, a1):
        out = []
        for c in contacts:
            if c.bodies[0].entity == a0 and c.bodies[1].entity == a1: out.append((c, True))
            elif c.bodies[0].entity == a1 and c.bodies[1].entity == a0: out.append((c, False))
        return out

    def get_pairwise_contact_impulse(contacts, a0, a1):
        tot = np.zeros(3)
        for c, flag in get_pairwise_contacts(contacts, a0, a1): tot += np.sum([p.impulse for p in c.points], axis=0) * (1 if flag else -1)
        return tot
    su.get_pairwise_contacts = get_pairwise_contacts; su.get_pairwise_contact_impulse = get_pairwise_contact_impulse
    pkg = types.ModuleType('mani_skill'); utils = types.ModuleType('mani_skill.utils'); utils.sapien_utils = su; pkg.utils = utils
    fakepkg = types.ModuleType('mani_skill_fake'); fakepkg.sapien_utils = su
    sap = types.ModuleType('sapien'); phx = types.ModuleType('sapien.physx'); sap.physx = phx
    for n, c in (('PhysxCollisionShapeBox', Box), ('PhysxCollisionShapeConvexMesh', Convex), ('PhysxCollisionShapePlane', Plane)): setattr(phx, n, c)
    for n in ('PhysxCollisionShapeSphere', 'PhysxCollisionShapeCapsule', 'PhysxCollisionShapeCylinder', 'PhysxCollisionShapeTriangleMesh'): setattr(phx, n, type(n, (), {}))
    for k, v in (('mani_skill', pkg), ('mani_skill.utils', utils), ('mani_skill.utils.sapien_utils', su), ('mani_skill_fake', fakepkg), ('sapien', sap), ('sapien.physx', phx)): monkeypatch.setitem(sys.modules, k, v)


# ---------------------------------------------------------------- the probe
def test_the_spatial_probe_keeps_every_existing_field_and_adds_only_the_declared_ones():
    u, _, _, w = fake(); base = RV.probe(u, 5, [0, 0, 0, -1]); u2, _, _, _ = fake(); ext = P.probe_spatial(u2, 5, [0, 0, 0, -1])
    assert {k: ext[k] for k in base} == base and set(ext) - set(base) == set(P.STEP_FIELDS)
    u3, _, _, _ = fake(); ini = P.probe_spatial(u3, 0, None); assert set(ini) - set(RV.probe(fake()[0], 0, None)) == set(P.ADDED_FIELDS)
    assert ext['spatial_schema'] == P.SCHEMA and ext['elapsed_steps'] == 5 and ext['sim_time_s'] == .25 and ext['sim_substeps_per_control'] == 5. and ext['sim_timestep_s'] == .01
    assert ext['robot_qpos'] == pytest.approx([0., .1, 0., -2., 0., 2., .8, .02, .02]) and ext['robot_qvel'] == [.5] * 9 and set(ext['link_pose']) == set(LINKS)
    assert ext['link_pose']['panda_link3'] == pytest.approx([.03, 0, .3, 1, 0, 0, 0])
    assert ini['robot_joint_names'][:2] == ['panda_joint1', 'panda_joint2'] and len(ini['robot_joint_names']) == 9
    g = {k: v['shapes'] for k, v in ini['collision_geometry'].items()}; assert set(g) == set(LINKS) | {'cubeA', 'cubeB', 'table', 'scene-0_ground'} and g['panda_link8'] == [] and g['scene-0_ground'][0]['type'] == 'Plane'
    assert g['panda_hand'][0]['type'] == 'Convex' and len(g['panda_hand'][0]['vertices']) == 4 and g['panda_hand'][0]['triangles'][0] == [0, 2, 1] and g['table'][0]['half_size'] == pytest.approx([.6, .6, .01])
    assert g['table'][0]['local_pose'] == pytest.approx([0, 0, -.01, 1, 0, 0, 0]) and len(g['panda_leftfinger']) == 2
    assert ini['collision_geometry']['cubeB']['pose_at_this_row'] == pytest.approx([.1, 0, .02, 1, 0, 0, 0]) and ini['collision_geometry']['panda_link2']['pose_at_this_row'] == pytest.approx([.02, 0, .3, 1, 0, 0, 0])


def test_contacts_are_kept_as_physx_gives_them_and_forces_follow_the_upstream_definition():
    u, _, contacts, w = fake(); r = P.probe_spatial(u, 5, [0, 0, 0, -1])
    assert [(c['a'], c['b']) for c in r['contacts']] == [('cubeA', 'cubeB'), ('cubeB', 'table'), ('panda_hand', 'cubeB'), ('panda_leftfinger', 'cubeA'), ('panda_link3', 'scene-0_ground'), ('panda_link5', 'table'),
                                                         ('panda_link6', 'cubeA'), ('table', 'panda_rightfinger')]       # stored sorted; orientation as PhysX gives it
    c1 = r['contacts'][3]; assert c1['shape_a'] == 1 and c1['shape_b'] == 0 and c1['points'] == [[0., 0., 0., 0., 0., 1., .2, 0., 0., 0.], [0., 0., 0., 0., 0., 1., .1, 0., 0., 0.]] and len(r['contacts'][-1]['points'][0]) == 10
    assert r['contacts'][6]['points'][0][-1] == pytest.approx(.004)                       # proximity within the contact offset: kept, with zero impulse
    u2, _, contacts2, _ = fake(); contacts2.reverse(); assert P.probe_spatial(u2, 5, [0, 0, 0, -1])['contacts'] == r['contacts']     # the PhysX order does not reach the record
    from mani_skill_fake import sapien_utils as su
    L = w['links']; bodies = {**{n: o._bodies[0].entity for n, o in L.items()}, **{n: w[n]._bodies[0].entity for n in ('A', 'B', 'table', 'ground')}}
    imps = P.pair_impulses(contacts)
    for a in bodies.values():
        for b in bodies.values(): assert P.pair_force(contacts, imps, a, b, w['dt']) == float(np.linalg.norm(su.get_pairwise_contact_impulse(contacts, a, b)) / w['dt'])
    assert r['f_hand_B'] == pytest.approx(3.) and r['f_hand_A'] == 0. and r['f_B_table'] == pytest.approx(.4) and r['f_fingerR_table'] == pytest.approx(.2) and r['f_fingerL_table'] == 0.
    assert r['f_links_B'] == {'panda_hand': pytest.approx(3.)} and r['f_links_A'] == {'panda_leftfinger': pytest.approx(30.)} and r['f_links_table'] == {'panda_link5': pytest.approx(1.), 'panda_rightfinger': pytest.approx(.2)}
    assert r['f_links_other'] == {'panda_link3|scene-0_ground': pytest.approx(.1)} and 'panda_link6' not in r['f_links_A']      # zero impulse: no force entry
    assert r['f_A_B'] == pytest.approx(.5) and r['f_A_left'] == pytest.approx(30.)          # same definition as the existing fields


def test_values_are_stored_as_the_exact_float32_the_engine_holds():
    x = np.array([0.1, 1 / 3, 2.3841858e-07, -3.2782554626464844e-07, 1e-30, 12345.678, 0.], np.float32)
    assert np.array_equal(np.asarray(P.f32(x), np.float32), x) and all(isinstance(v, float) for v in P.f32(x)) and P.f32(np.float32(.1)) == [.1]
    assert json.loads(json.dumps(P.f32(x))) == P.f32(x)


def test_the_spatial_probe_only_reads_the_simulator():
    u, log, contacts, _ = fake(); before = copy.deepcopy([[p.impulse.tolist() for p in c.points] for c in contacts]); P.probe_spatial(u, 0, None); P.probe_spatial(u, 5, [0, 0, 0, 1.])
    assert set(log) <= {'env.evaluate', 'scene.get_pairwise_contact_forces', 'robot.get_qpos', 'robot.get_qvel', 'robot.get_links', 'robot.get_active_joints', 'scene.get_contacts', 'scene.get_all_actors'}
    assert [[p.impulse.tolist() for p in c.points] for c in contacts] == before and log.count('scene.get_contacts') == 2          # once per call
    u.scene.gpu_sim_enabled = True
    with pytest.raises(RuntimeError, match='GPU backend'): P.probe_spatial(u, 5, None)
    src = Path('src/hci/spatial_probe.py').read_text(encoding='utf-8').split('# ----------------------------------------------------------------')[0]
    for word in ('set_', '.step(', 'random', 'render', 'reset(', 'set_state', 'add_force', 'set_pose', 'set_qpos'): assert word not in src.split('"""', 2)[2], word


def test_a_logging_failure_is_an_error_never_a_row_without_the_fields():
    u, _, contacts, w = fake(); contacts[0].shapes = (Box((.1, .1, .1)), contacts[0].shapes[1])                  # a shape that its body does not hold
    with pytest.raises(RuntimeError, match='contact shape not found'): P.probe_spatial(u, 5, None)
    u, _, _, w = fake(); w['A']._bodies[0].collision_shapes.append(object())
    with pytest.raises(RuntimeError, match='unknown collision shape'): P.probe_spatial(u, 0, None)
    u, _, _, _ = fake(); real = RV.probe
    try:
        RV.probe = lambda *a, **k: {**real(*a, **k), 'contacts': []}                                             # an existing field of the same name
        with pytest.raises(RuntimeError, match='overwrite'): P.probe_spatial(u, 5, None)
    finally:
        RV.probe = real


# ---------------------------------------------------------------- C1 with the spatial fields
def mock_spatial(rng=None):
    """probe2 (the mock probe of the C1 tests) + every spatial field, with arbitrary values (random when rng is given)."""
    from test_trajectory_audit import probe2
    def pr(rt, z):
        v = (lambda n: [float(x) for x in rng.normal(size=n)]) if rng is not None else (lambda n: [0.] * n)
        new = {'spatial_schema': P.SCHEMA, 'elapsed_steps': rt.t, 'sim_time_s': rt.t / 20, 'sim_timestep_s': .01, 'sim_substeps_per_control': 5., 'robot_qpos': v(9), 'robot_qvel': v(9),
               'link_pose': {'panda_hand': v(7)}, 'contacts': [{'a': 'cubeA', 'b': 'table', 'shape_a': 0, 'shape_b': 0, 'points': [v(10)]}], 'f_hand_A': v(1)[0], 'f_hand_B': 1.5, 'f_B_table': 2.,
               'f_hand_table': 0., 'f_fingerL_table': 0., 'f_fingerR_table': 0., 'f_links_A': {}, 'f_links_B': {'panda_hand': 1.5}, 'f_links_table': {}, 'f_links_other': {}}
        if z is None: new.update(collision_geometry={'cubeA': {'pose_at_this_row': [0.] * 7, 'shapes': [{'type': 'Box', 'half_size': [.02] * 3}]}}, robot_joint_names=['j'] * 9)
        return {**probe2(rt, z), **new}
    return pr


def c1_scene(probe, bias=(.028, .012), seed=7):
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    from hci.simulation import Runtime
    from hci import vision_center_c1 as C1
    r = Runtime(VisionEnv2(), BiasPolicy(bias), CFG); return C1.collect_scene(r, seed, CFG, probe=probe, arms=C1.BASE_ARMS, policy_id='train_seed_1')


def test_extra_trace_fields_change_no_c1_result_whatever_their_values():
    """The same mock scene with the earlier probe, and with the spatial fields filled with two different random contents:
    record (apart from durations), frames and earlier trace fields identical. No controller, memory, abstention rule or
    queue reads the added fields."""
    from test_trajectory_audit import probe2
    off = c1_scene(probe2); on1 = c1_scene(mock_spatial(np.random.default_rng(1))); on2 = c1_scene(mock_spatial(np.random.default_rng(2)))
    assert off[0]['triggered'] and any(c.get('acted') for c in off[0]['cells'])
    for on in (on1, on2): assert P.compare_on_off(off, on) == {'record_equal_except_durations': True, 'frames_bitwise_equal': True, 'earlier_trace_fields_equal': True, 'spatial_fields_in_every_on_row': True,
                                                               'no_spatial_field_without_the_logger': True, 'pass': True}
    assert on1[1]['C0'][3]['robot_qpos'] != on2[1]['C0'][3]['robot_qpos']                  # the added values really differed


def test_the_on_off_comparison_detects_every_kind_of_difference():
    from test_trajectory_audit import probe2
    off = c1_scene(probe2); on = c1_scene(mock_spatial())
    def bad(edit):
        x = copy.deepcopy(on); edit(*x); return P.compare_on_off(off, x)
    assert bad(lambda r, t, f: r['cells'][1].update(y=1 - r['cells'][1]['y']))['record_equal_except_durations'] is False
    assert bad(lambda r, t, f: r['trunk']['inference_calls'][0].update(seed=1))['record_equal_except_durations'] is False        # policy noise keys
    assert bad(lambda r, t, f: t['C0'][4].update(tcp=[t['C0'][4]['tcp'][0] + 1e-12, *t['C0'][4]['tcp'][1:]]))['earlier_trace_fields_equal'] is False
    assert bad(lambda r, t, f: t['trunk'][2].update(action=[t['trunk'][2]['action'][0] + 1e-3, *t['trunk'][2]['action'][1:]]))['earlier_trace_fields_equal'] is False   # an executed command
    k = next(k for k in on[2] if on[2][k]); assert bad(lambda r, t, f: f[k][0].__setitem__((0, 0, 0), f[k][0][0, 0, 0] ^ 1))['frames_bitwise_equal'] is False
    assert bad(lambda r, t, f: t['trunk'][5].pop('contacts'))['spatial_fields_in_every_on_row'] is False
    assert bad(lambda r, t, f: r['initial_state'].pop('collision_geometry'))['spatial_fields_in_every_on_row'] is False
    assert P.compare_on_off(on, on)['no_spatial_field_without_the_logger'] is False
    try: res = c1_scene(lambda rt, z: _nudge(rt) or mock_spatial()(rt, z))          # a logger that moves the target by a micrometre
    except RuntimeError as e: assert 'differs' in str(e) or 'reproduce' in str(e)       # the collector's own restore / trunk checks catch it first
    else: assert not P.compare_on_off(off, res)['pass']


def _nudge(rt):
    """Move the mock target cube by 1 micrometre: a logger with a side effect on the simulation."""
    rt.env.B = rt.env.B + torch.tensor([[1e-6, 0., 0.]])


def test_global_rng_helpers_restore_and_detect_a_draw():
    s = P.rng_snapshot(); d0 = P.rng_digest(); np.random.rand(); random.random(); torch.rand(1); assert P.rng_digest() != d0
    P.rng_restore(s); assert P.rng_digest() == d0
    np.random.rand(); assert P.rng_digest() != d0; P.rng_restore(s); torch.rand(1); assert P.rng_digest() != d0; P.rng_restore(s); assert P.rng_digest() == d0


# ---------------------------------------------------------------- the driver
@pytest.fixture(scope='module')
def X34():
    spec = importlib.util.spec_from_file_location('c1_driver_sp', Path('scripts/34_vision_center_c1.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_every_c1_scene_is_logged_with_the_spatial_probe(X34):
    src = Path('scripts/34_vision_center_c1.py').read_text(encoding='utf-8')
    assert X34.PROBE is P.gt_probe_spatial and src.count('probe=PROBE') == 2 and 'probe=X.gt_probe' not in src and src.count('X.gt_probe') == 2      # the definition comment and probe-check
    assert "('earlier_logger', X.gt_probe), ('spatial_logger', PROBE)" in src and {'src/hci/spatial_probe.py', 'tests/test_spatial_probe.py'} <= set(X34.CODE)
    sh = Path('scripts/vision_center_c1/run.sh').read_text(encoding='utf-8')
    assert sh.index('probecheck)') < sh.index('  audit)') and 'grep -q "^PROBE_CHECK_DONE" $ST' in sh and '|| true' not in sh


def probe_check_rig(X34, tmp_path, monkeypatch, spatial):
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    from test_trajectory_audit import probe2
    from hci.simulation import Runtime
    from hci import vision_center_c1 as C1
    plan = tmp_path / 'plan.json'; plan.write_text('{"pass": true}'); h = X34.sha256(plan); run = tmp_path / 'run'; run.mkdir()
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 6.0, 'plan_sha256': h}))
    (run / 'scenes.json').write_text(json.dumps({'pass': True, 'plan_sha256': h}))
    class Env(VisionEnv2):
        def close(self): pass
    monkeypatch.setattr(X34, 'bound', lambda p: ({}, CFG, tuple(C1.BASE_ARMS))); monkeypatch.setattr(X34, 'runtime_for', lambda c, pid, a: Runtime(Env(), BiasPolicy((.028, .012)), CFG))
    monkeypatch.setattr(X34.X, 'gt_probe', probe2); monkeypatch.setattr(X34, 'PROBE', spatial); monkeypatch.setattr(X34, 'PROBE_CHECK_SEEDS', (7, 8))
    return plan, run, h


def test_probe_check_passes_saves_both_runs_and_is_required_before_the_audit(X34, tmp_path, monkeypatch):
    plan, run, h = probe_check_rig(X34, tmp_path, monkeypatch, mock_spatial(np.random.default_rng(3)))
    ns = argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu')
    X34.cmd_probe_check(ns); rep = json.loads((run / 'probe_check' / 'probe_check.json').read_text())
    assert rep['pass'] and rep['status'] == 'OK' and rep['plan_sha256'] == h and [x['scene'] for x in rep['scenes']] == [7, 8] and rep['arm_runs_that_acted'] >= 1 and rep['tolerance'] == 'none: bitwise equality'
    assert all(x['global_rng_after_equal'] and x['record_equal_except_durations'] for x in rep['scenes']) and 'logging_time_overhead_fraction' in rep
    for name in ('earlier_logger', 'spatial_logger'): assert sorted(p.name for p in (run / 'probe_check' / name / 'scenes').iterdir()) == ['7.json', '8.json']
    assert X34.probe_check_passed(run, h) and not X34.probe_check_passed(run, '0' * 64)
    with pytest.raises(FileExistsError): X34.cmd_probe_check(ns)                                             # never repeated into the same folder
    (run / 'probe_check' / 'probe_check.json').write_text(json.dumps({**rep, 'pass': False}))
    with pytest.raises(SystemExit, match='probe_check missing or not passed'): X34.cmd_audit(ns)
    assert not (run / 'audit').exists()


def test_probe_check_blocks_a_logger_with_a_side_effect(X34, tmp_path, monkeypatch):
    def drawing(rt, z): np.random.rand(); return mock_spatial()(rt, z)                     # consumes the global RNG
    plan, run, h = probe_check_rig(X34, tmp_path, monkeypatch, drawing); ns = argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu')
    with pytest.raises(SystemExit): X34.cmd_probe_check(ns)
    rep = json.loads((run / 'probe_check' / 'probe_check.json').read_text()); assert not rep['pass'] and rep['status'] == 'TECHNICAL_BLOCKED' and not rep['scenes'][0]['global_rng_after_equal']
    (tmp_path / 'b').mkdir(); plan2, run2, _ = probe_check_rig(X34, tmp_path / 'b', monkeypatch, lambda rt, z: _nudge(rt) or mock_spatial()(rt, z))
    with pytest.raises(SystemExit): X34.cmd_probe_check(argparse.Namespace(plan=str(plan2), run=str(run2), upstream='x', device='cpu'))
    rep = json.loads((run2 / 'probe_check' / 'probe_check.json').read_text()); s0 = rep['scenes'][0]
    assert not rep['pass'] and rep['status'] == 'TECHNICAL_BLOCKED' and (s0.get('error_in') == 'spatial_logger' or not s0['earlier_trace_fields_equal'])


def test_probe_check_refuses_without_a_passed_placement_check(X34, tmp_path, monkeypatch):
    plan, run, h = probe_check_rig(X34, tmp_path, monkeypatch, mock_spatial()); (run / 'scenes.json').write_text(json.dumps({'pass': True, 'plan_sha256': '0' * 64}))
    with pytest.raises(SystemExit, match='scenes.json'): X34.cmd_probe_check(argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu'))
    assert not (run / 'probe_check').exists()


def test_review_a1_an_error_in_the_comparison_or_the_save_leaves_a_failure_record(X34, tmp_path, monkeypatch):
    plan, run, h = probe_check_rig(X34, tmp_path, monkeypatch, mock_spatial())
    def broken(*a, **k): raise OSError('disk full (injected by the test)')
    monkeypatch.setattr(X34.Z, 'save_scene', broken)
    with pytest.raises(SystemExit, match='PROBE CHECK FAILED'): X34.cmd_probe_check(argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu'))
    rep = json.loads((run / 'probe_check' / 'probe_check.json').read_text()); assert not rep['pass'] and rep['status'] == 'TECHNICAL_BLOCKED' and 'disk full' in rep['scenes'][-1]['error']
    assert not X34.probe_check_passed(run, h)


def test_review_a2_the_plan_fails_without_a_passed_local_invariance_check_of_the_current_logger(X34, tmp_path, monkeypatch):
    ev = tmp_path / 'evidence'; ev.mkdir()
    (ev / 'timing.json').write_text(json.dumps({'ms_added_by_spatial_over_earlier_logger': 1., 'ms_earlier_logger_over_none': 1.}))
    (ev / 'replay_summary.json').write_text(json.dumps({'scenes': {'1': {'fidelity': {'trunk': {'steps': 9}}, 'trace_bytes': 10000}}}))
    br = {'state_obs_reward_info_bitwise_equal_all_4_runs': True, 'rng_changed_by_probe_at_steps': [], 'probe_rows_equal_between_the_two_on_runs': True, 'steps': 9}
    good = {'pass': True, 'scenes': {'1': {'branches': {'trunk': br}}}, 'spatial_probe_sha256': X34.sha256('src/hci/spatial_probe.py')}
    monkeypatch.setattr(X34, 'EVIDENCE', str(ev))
    def problems(inv):
        (ev / 'invariance.json').write_text(json.dumps(inv)); out = tmp_path / f'p{len(list(tmp_path.iterdir()))}'
        try: X34.cmd_plan(argparse.Namespace(out=str(out)))
        except SystemExit: pass
        return json.loads((out / 'plan.json').read_text(encoding='utf-8'))['problems']
    msg = 'local logging ON/OFF invariance'
    assert not any(msg in p or 'invariance check was not made' in p for p in problems(good))
    assert any(msg in p for p in problems({**good, 'pass': False}))
    assert any(msg in p for p in problems({**good, 'scenes': {'1': {'branches': {'trunk': {**br, 'rng_changed_by_probe_at_steps': [3]}}}}}))
    assert any('invariance check was not made with the current' in p for p in problems({**good, 'spatial_probe_sha256': '0' * 64}))
    cover = 'does not cover scene'
    assert sum(cover in p for p in problems(good)) == len(X34.INVARIANCE_SEEDS)                       # review a3: one branch of one scene is not enough
    full = {'pass': True, 'spatial_probe_sha256': good['spatial_probe_sha256'], 'scenes': {}}
    for sd in X34.INVARIANCE_SEEDS:
        part = next(p for p in ('shadow', 'pilot') if Path(f'results/vision_center_v2_001/execution/{p}/scenes/{sd}.json').exists())
        r = json.loads(Path(f'results/vision_center_v2_001/execution/{part}/scenes/{sd}.json').read_text(encoding='utf-8'))
        full['scenes'][str(sd)] = {'branches': {n: br for n in ['trunk'] + [c['arm'] for c in r['cells']]}}
    assert not any(cover in p or msg in p for p in problems(full))
    miss = json.loads(json.dumps(full)); miss['scenes'][str(X34.INVARIANCE_SEEDS[0])]['branches'].pop('C0'); assert sum(cover in p for p in problems(miss)) == 1
    zero = json.loads(json.dumps(full)); zero['scenes'][str(X34.INVARIANCE_SEEDS[1])]['branches']['trunk']['steps'] = 0; assert sum(cover in p for p in problems(zero)) == 1


def test_review_a4_an_error_while_writing_the_report_leaves_a_failure_record(X34, tmp_path, monkeypatch):
    plan, run, h = probe_check_rig(X34, tmp_path, monkeypatch, mock_spatial()); real = X34.write_json
    def flaky(p, x):
        if Path(p).name == 'probe_check.json' and x.get('pass') is not False: raise OSError('cannot write (injected by the test)')
        return real(p, x)
    monkeypatch.setattr(X34, 'write_json', flaky)
    with pytest.raises(SystemExit, match='finalisation'): X34.cmd_probe_check(argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu'))
    rep = json.loads((run / 'probe_check' / 'probe_check.json').read_text()); assert rep['pass'] is False and 'cannot write' in rep['scenes'][-1]['error'] and not X34.probe_check_passed(run, h)
