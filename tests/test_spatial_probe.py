"""The logging-only probe proposal (hci.spatial_probe): it keeps every existing field, only reads simulator state, and
extra trace fields change no C1 result. No simulator: a fake ManiSkill scene with recorded accessor calls."""
import copy, json, sys, types
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from hci import spatial_probe as P
from hci import recovery_v1 as RV


class Pose:
    def __init__(self, p, q): self.p = torch.tensor([p], dtype=torch.float32); self.q = torch.tensor([q], dtype=torch.float32)


class Ent:
    def __init__(self, name): self.name = name


class Obj:
    def __init__(self, name, p=(0, 0, 0), q=(1, 0, 0, 0)):
        self.name = name; self.entity = Ent(name); self._bodies = [types.SimpleNamespace(entity=self.entity)]; self.pose = Pose(p, q)
        self.linear_velocity = torch.zeros(1, 3); self.angular_velocity = torch.zeros(1, 3)


class Log(list):
    def wrap(self, name, fn):
        def g(*a, **k): self.append(name); return fn(*a, **k)
        return g


def contact(a, b, imp): return types.SimpleNamespace(bodies=(types.SimpleNamespace(entity=a.entity), types.SimpleNamespace(entity=b.entity)), points=[types.SimpleNamespace(impulse=np.array(imp, float))])


def fake():
    log = Log(); A = Obj('cubeA', (.1, 0, .06)); B = Obj('cubeB', (.1, 0, .02)); table = Obj('table-workspace')
    names = [f'panda_link{i}' for i in range(9)] + ['panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'panda_hand_tcp']; links = {n: Obj(n) for n in names}
    dt = .01; contacts = [contact(links['panda_hand'], B, [0, 0, .03]), contact(links['panda_leftfinger'], A, [.2, 0, 0]), contact(A, B, [0, 0, .005]), contact(B, table, [0, 0, .004]), contact(links['panda_link5'], table, [0, .01, 0])]
    from mani_skill_fake import sapien_utils as su

    def pairwise(a, b): return torch.tensor([su.get_pairwise_contact_impulse(contacts, a._bodies[0].entity, b._bodies[0].entity) / dt], dtype=torch.float32)
    scene = types.SimpleNamespace(gpu_sim_enabled=False, px=types.SimpleNamespace(timestep=dt), get_contacts=log.wrap('scene.get_contacts', lambda: list(contacts)),
                                  get_pairwise_contact_forces=log.wrap('scene.get_pairwise_contact_forces', pairwise))
    qpos = torch.tensor([[0., .1, 0., -2., 0., 2., .8, .02, .02]])
    robot = types.SimpleNamespace(get_qpos=log.wrap('robot.get_qpos', lambda: qpos.clone()), get_links=log.wrap('robot.get_links', lambda: list(links.values())))
    agent = types.SimpleNamespace(tcp=Obj('panda_hand_tcp', (.1, 0, .06), (0, 1, 0, 0)), robot=robot, finger1_link=links['panda_leftfinger'], finger2_link=links['panda_rightfinger'])
    flags = {k: torch.tensor([v]) for k, v in (('is_cubeA_grasped', True), ('is_cubeA_on_cubeB', False), ('is_cubeA_static', False), ('success', False))}
    u = types.SimpleNamespace(scene=scene, agent=agent, cubeA=A, cubeB=B, table_scene=types.SimpleNamespace(table=table), evaluate=log.wrap('env.evaluate', lambda: flags))
    return u, log, contacts


@pytest.fixture(autouse=True)
def fake_maniskill(monkeypatch):
    """The upstream helper functions (mani_skill/utils/sapien_utils.py at a4a4f92), as pure Python over the contact list."""
    su = types.ModuleType('sapien_utils')

    def get_pairwise_contacts(contacts, a0, a1):
        return [(c, True) if (c.bodies[0].entity == a0 and c.bodies[1].entity == a1) else (c, False) for c in contacts if {c.bodies[0].entity, c.bodies[1].entity} == {a0, a1}]

    def get_pairwise_contact_impulse(contacts, a0, a1):
        tot = np.zeros(3)
        for c, flag in get_pairwise_contacts(contacts, a0, a1): tot += np.sum([p.impulse for p in c.points], axis=0) * (1 if flag else -1)
        return tot
    su.get_pairwise_contacts = get_pairwise_contacts; su.get_pairwise_contact_impulse = get_pairwise_contact_impulse
    pkg = types.ModuleType('mani_skill'); utils = types.ModuleType('mani_skill.utils'); utils.sapien_utils = su; pkg.utils = utils
    fakepkg = types.ModuleType('mani_skill_fake'); fakepkg.sapien_utils = su
    for k, v in (('mani_skill', pkg), ('mani_skill.utils', utils), ('mani_skill.utils.sapien_utils', su), ('mani_skill_fake', fakepkg)): monkeypatch.setitem(sys.modules, k, v)


def test_the_extended_probe_keeps_every_existing_field_and_adds_only_new_ones():
    u, log, _ = fake(); base = RV.probe(u, 5, [0, 0, 0, -1]); u2, log2, _ = fake(); ext = P.probe_spatial(u2, 5, [0, 0, 0, -1])
    assert {k: ext[k] for k in base} == base and set(ext) - set(base) == set(P.ADDED_FIELDS)
    assert ext['robot_qpos'] == pytest.approx([0., .1, 0., -2., 0., 2., .8, .02, .02]) and ext['f_hand_B'] == pytest.approx(3.) and ext['f_hand_A'] == 0.
    assert ext['f_B_table'] == pytest.approx(.4) and ext['f_links_B'] == {'panda_hand': pytest.approx(3.)} and ext['f_links_table'] == {'panda_link5': pytest.approx(1.)}
    assert ext['f_A_B'] == base['f_A_B'] == pytest.approx(.5)                                # same force definition as the existing field


def test_the_extended_probe_only_reads_the_simulator():
    u, log, contacts = fake(); before = copy.deepcopy([(c.points[0].impulse.tolist()) for c in contacts]); P.probe_spatial(u, 5, None)
    assert set(log) <= {'env.evaluate', 'scene.get_pairwise_contact_forces', 'robot.get_qpos', 'robot.get_links', 'scene.get_contacts'}
    assert [(c.points[0].impulse.tolist()) for c in contacts] == before and log.count('scene.get_contacts') == 1
    u.scene.gpu_sim_enabled = True
    with pytest.raises(RuntimeError, match='GPU backend'): P.probe_spatial(u, 5, None)


def test_extra_trace_fields_change_no_c1_result():
    """The same mock scene with the existing probe and with extra fields added to every row: identical record (apart from
    timings), identical frames, identical traces apart from the added fields."""
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    from test_trajectory_audit import probe2
    from hci.simulation import Runtime
    from hci import vision_center_c1 as C1
    extra = lambda rt, z: {**probe2(rt, z), 'robot_qpos': [0.] * 9, 'f_hand_A': 0., 'f_hand_B': 1.5, 'f_B_table': 2., 'f_links_B': {'panda_hand': 1.5}, 'f_links_table': {}}
    out = []
    for pr in (probe2, extra):
        r = Runtime(VisionEnv2(), BiasPolicy((.028, .012)), CFG); out.append(C1.collect_scene(r, 7, CFG, probe=pr, arms=C1.BASE_ARMS, policy_id='train_seed_1'))
    add = {'robot_qpos', 'f_hand_A', 'f_hand_B', 'f_B_table', 'f_links_B', 'f_links_table'}

    def strip(rec):                                                       # timings differ run to run; the added fields also appear in initial_state
        rec = json.loads(json.dumps(rec)); rec.pop('wall_seconds'); rec['memory'].pop('seconds')
        rec['initial_state'] = {k: v for k, v in rec['initial_state'].items() if k not in add}
        for k in ('reading_seconds', 'memory_seconds'): rec['counts'].pop(k, None)
        for c in rec['cells']: c.pop('wall_seconds', None)
        return rec
    (r0, t0, f0), (r1, t1, f1) = out; assert r0['triggered'] and strip(r0) == strip(r1)
    assert set(f0) == set(f1) and all(np.array_equal(np.stack(f0[k]), np.stack(f1[k])) for k in f0 if f0[k])
    assert t0.keys() == t1.keys() and all([{k: v for k, v in x.items() if k not in add} for x in t1[a]] == t0[a] for a in t0)
