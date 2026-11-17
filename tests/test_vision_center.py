"""Vision-CENTER: perception on synthetic and stored real frames, the act/abstain decision, the controller, accounting.

The simulator-side tests use mock physics and a synthetic renderer written here (flat colours, the module's own camera
constants). They say nothing about ManiSkill, the GPU or the research policy. The real-frame test uses three stored hand-camera
frames of the consumed diagnostic set with their logged ground truth as the label.
"""
import argparse, importlib.util, inspect, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_recovery_v1_oracle import PlaceEnv, PlacePolicy, CFG, probe, B0
from hci.simulation import Runtime
from hci import vision_center as VC
from hci import recovery_v1_oracle as O

TABLE = np.array([200, 120, 75], np.uint8); DOWN = np.array([0., 1., 0., 0.])       # TCP pointing down: the hand camera looks at the table
FIX = Path(__file__).parent / 'fixtures' / 'vision_center_frames.npz'


def render(tcp_pose, cubes, finger=False):
    """Hand-camera image of axis-aligned-height cubes [(centre xyz, yaw, colour index)], far to near; the top face is lit."""
    cam = VC.cam_hand(tcp_pose); img = np.empty((VC.IMG, VC.IMG, 3), np.uint8); img[:] = TABLE; h = VC.CUBE_HALF_M
    for c, yaw, col in sorted(cubes, key=lambda x: -np.linalg.norm(np.asarray(x[0]) - cam[1])):
        cs, sn = np.cos(yaw), np.sin(yaw); P = []
        for dz in (h, -h):
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p, z = VC.project(cam, [c[0] + h * (a * cs - b * sn), c[1] + h * (a * sn + b * cs), c[2] + dz])
                if z <= 1e-3: P = None; break
                P.append(p)
            if P is None: break
        if P is None: continue
        P = np.array(P); sil = np.zeros((VC.IMG, VC.IMG), bool)
        for f in VC._FACES: sil |= VC.raster(P[list(f)])
        dark = np.zeros(3, np.uint8); dark[col] = 147; lit = np.zeros(3, np.uint8); lit[col] = 230; img[sil] = dark; img[VC.raster(P[:4])] = lit
    if finger: img[100:, :12] = 128
    return img


def pose(x=0., y=0., z=.065): return np.array([x, y, z, *DOWN])


def held(p, hold=(0, 0, 0), yaw=0.): return ((np.asarray(p[:3]) + VC.quat_R(p[3:]) @ np.asarray(hold, float)), yaw, 0)


def target(x, y, yaw=0.): return ((x, y, .02), yaw, 1)


# ---------------------------------------------------------------- perception
@pytest.mark.parametrize('dx,dy,yaw', [(.04, .0, 0.), (.05, .03, .5), (.03, -.04, -.3), (.09, .0, .2)])
def test_target_is_recovered_from_a_full_view(dx, dy, yaw):
    p = pose(); r = VC.target_partial_fit(render(p, [target(dx, dy, yaw)]), VC.cam_hand(p), 'hand_camera')
    assert r['valid'] and np.linalg.norm(r['xy'] - [dx, dy]) < 0.001
    fv = VC.target_full_view(render(p, [target(dx, dy, yaw)]), VC.cam_hand(p), 'hand_camera')
    assert fv['valid'] and np.linalg.norm(fv['xy'] - [dx, dy]) < 0.001


def test_target_is_recovered_when_the_held_cube_hides_part_of_it_and_refused_when_hidden():
    p = pose(); cam = VC.cam_hand(p)
    r = VC.target_partial_fit(render(p, [target(.028, .012, .3), held(p)]), cam, 'hand_camera')
    assert r['valid'] and np.linalg.norm(r['xy'] - [.028, .012]) < 0.002
    hidden = VC.target_partial_fit(render(p, [target(-.02, .0), held(p)]), cam, 'hand_camera')          # behind the held cube as seen from the camera
    assert hidden is None or not hidden['valid']
    assert VC.target_partial_fit(render(p, [held(p)]), cam, 'hand_camera') is None


def test_centroid_of_a_partial_view_is_not_accepted_as_a_full_view():
    p = pose(); fv = VC.target_full_view(render(p, [target(.028, .012, .3), held(p)]), VC.cam_hand(p), 'hand_camera')
    assert fv is not None and not fv['valid']


@pytest.mark.parametrize('hold,yaw', [((0, 0, 0), 0.), ((.010, -.002, -.006), -.3), ((-.007, 0, .005), 0.), ((.017, -.003, -.005), -.27), ((-.002, 0, .001), .6)])
def test_hold_offset_is_recovered_with_fingers_in_view(hold, yaw):
    p = pose(.03, -.02); r = VC.hold_fit(render(p, [held(p, hold, yaw)], finger=True))
    assert r['valid'] and np.linalg.norm((r['h'] - hold)[:2]) < 0.003 and abs(r['h'][2] - hold[2]) < 0.003


def test_hold_fit_refuses_an_image_without_the_held_cube():
    p = pose(); assert VC.hold_fit(render(p, [target(.05, 0)])) is None


def test_reading_gives_the_vector_the_tcp_has_to_move():
    p = pose(.02, .03); hold = (.008, 0, 0); A = held(p, hold); off = np.array([.026, -.008]); rd = VC.reading(render(p, [target(*(A[0][:2] + off)), A]), p)
    assert rd['target_fresh'] and np.linalg.norm(rd['offset_xy'] - off) < 0.003 and VC.decide(rd, 100) == {'act': True, 'reason': 'ACT'}
    json.dumps(VC.summary(rd))


def test_abstain_reasons():
    p = pose(); A = held(p); near = VC.reading(render(p, [target(.028, .0), A]), p); assert VC.decide(near, 100)['reason'] == 'ACT'
    assert VC.decide(near, VC.STEPS_LEFT_MIN - 1)['reason'] == 'TOO_FEW_STEPS'
    assert VC.decide(VC.reading(render(p, [target(-.02, 0), A]), p), 100)['reason'] == 'NO_TARGET_ESTIMATE'
    assert VC.decide(VC.reading(render(p, [target(.06, 0)]), p), 100)['reason'] == 'NO_HOLD_ESTIMATE'
    fake = dict(near); fake['offset_norm_m'] = VC.ACT_MIN_M - 1e-4; assert VC.decide(fake, 100)['reason'] == 'ALIGNED'
    low = pose(z=.065); B = target(.03, .0); Alow = held(low, (0, 0, .012))                              # the cube has slipped 12 mm down in the hand
    assert VC.decide(VC.reading(render(low, [B, Alow]), low), 100)['reason'] == 'LOW_CARRY'
    for rd in (near,): assert all(not VC.decide(rd, 100)['act'] or VC.decide(rd, 100)['reason'] == 'ACT' for _ in [0])


def test_stored_real_frames():
    """Three trigger frames of failed diagnostic episodes; ground truth is the label."""
    d = np.load(FIX); seen = {}
    for i, seed in enumerate(d['seeds']):
        rd = VC.reading(d['frames'][i], d['tcp_pose'][i]); dec = VC.decide(rd, 100)['reason']; seen[int(seed)] = dec
        if dec == 'ACT': assert np.linalg.norm(rd['offset_xy'] - d['true_offset_xy'][i]) < 0.002
        if rd['hold_fit'] is not None and rd['hold_fit']['valid']: assert np.linalg.norm((rd['hold_fit']['h'] - d['true_hold'][i])[:2]) < 0.0025
    assert seen == {50611: 'ACT', 50664: 'ACT', 50613: 'NO_TARGET_ESTIMATE'}


def test_perception_and_control_do_not_touch_object_state():
    for f in (VC.reading, VC.decide, VC.target_partial_fit, VC.target_full_view, VC.hold_fit, VC.observation, VC.vision_center):
        src = inspect.getsource(f)
        for word in ('cubeA', 'cubeB', 'get_state', 'is_grasp', 'success', 'oracle_offset', 'probe'): assert word not in src, (f.__name__, word)
    assert set(inspect.signature(VC.reading).parameters) == {'img_hand', 'tcp_pose', 'target'}


# ---------------------------------------------------------------- simulator side (mock physics, synthetic images)
class VisionEnv(PlaceEnv):
    """PlaceEnv with a hand-camera image rendered from its state. The TCP points down at carry height."""
    def _boot(self):
        super()._boot(); self.p = torch.tensor([[0., 0., .065]])

    def obs(self):
        o = super().obs(); o['extra']['tcp_pose'] = torch.cat([self.p, torch.tensor([DOWN], dtype=torch.float32)], -1); p = o['extra']['tcp_pose'][0].numpy().astype(float)
        A = (self.A[0].numpy().astype(float) if not self.held else np.array([p[0], p[1], .065]), 0., 0); B = (self.B[0].numpy().astype(float), 0., 1)
        o['sensor_data'] = {'hand_camera': {'rgb': torch.from_numpy(render(p, [B, A]))[None]}}; return o


class BiasPolicy(PlacePolicy):
    def __init__(self, bias): super().__init__(); self._bias = np.asarray(bias, float)
    def bias(self, episode): return self._bias


def vscene(bias, arms=VC.ARMS, seed=7):
    env = VisionEnv(); r = Runtime(env, BiasPolicy(bias), CFG); return (*VC.collect_scene(r, seed, CFG, probe=probe, arms=arms), env)


def cell(rec, arm): return next(x for x in rec['cells'] if x['arm'] == arm)


def test_scene_where_vision_acts():
    rec, traces, frames, env = vscene((-.028, .0)); v = cell(rec, 'VISION_CENTER'); c = v['control']
    assert rec['triggered'] and v['acted'] and c['decision'] == 'ACT' and 1 <= c['steps_used'] <= O.MAX_STEPS and v['oracle_reads'] == 0 and cell(rec, 'CENTER_ORACLE')['oracle_reads'] > 0
    assert c['end_reason'] in ('within_tolerance', 'step_budget', 'target_cap') and len(frames['VISION_CENTER']) == len(c['readings']) == c['steps_used'] + 1
    assert v['gt_reading_error']['offset_error_m'] < 0.004 and abs(v['gt_reading_error']['true_offset_m'] - .028) < 0.004 and not v['gt_reading_error']['unnecessary_act']
    assert v['gt_effect_of_control']['offset_xy_after_m'] < v['gt_effect_of_control']['offset_xy_before_m'] and all(np.hypot(*s['commanded_xy_m']) <= O.STEP_CMD_M + 1e-9 for s in c['steps'])
    assert VC.validate_scene(rec, CFG) and set(traces) == {'trunk', *VC.ARMS}; json.dumps(rec)


def test_vision_reads_no_cube_pose():
    env = VisionEnv(); r = Runtime(env, BiasPolicy((-.028, .0)), CFG); r.reset(3)
    while not O.trigger_reading(r)['fires']: r.step_base()
    before = env.reads; info = VC.vision_center(r); assert info['acted'] and env.reads == before and len(r.policy.queue) == 0


@pytest.mark.parametrize('bias,reason', [((.028, .012), 'NO_TARGET_ESTIMATE'), ((-.004, .0), None)])
def test_abstaining_scene_is_the_uninterrupted_one(bias, reason):
    rec, traces, frames, env = vscene(bias, arms=VC.SHADOW_ARMS); v = cell(rec, 'VISION_CENTER'); c0 = cell(rec, 'C0')
    assert not v['acted'] and v['control']['decision'] in VC.ABSTAIN and (reason is None or v['control']['decision'] == reason)
    assert (v['y'], v['steps']) == (c0['y'], c0['steps']) and v['reproduces_trunk']['match'] and v['intervention'] is None and rec['oracle_reads'] == 0 and len(frames['VISION_CENTER']) == 1


def test_abstain_leaves_the_queue_untouched():
    env = VisionEnv(); r = Runtime(env, BiasPolicy((.028, .012)), CFG); r.reset(3)
    while not O.trigger_reading(r)['fires']: r.step_base()
    q = [x.copy() for x in r.policy.queue]; t = r.t; info = VC.vision_center(r)
    assert not info['acted'] and r.t == t and r.intervention is None and len(r.policy.queue) == len(q) and all((a == b).all() for a, b in zip(q, r.policy.queue))


def test_validate_scene_rejects_inconsistent_records():
    rec, *_ = vscene((-.028, .0)); bad = json.loads(json.dumps(rec)); cell(bad, 'VISION_CENTER')['oracle_reads'] = 1
    with pytest.raises(ValueError, match='ground truth'): VC.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); cell(bad, 'VISION_CENTER')['control']['readings'][0]['offset_norm_m'] = .012
    with pytest.raises(ValueError, match='threshold'): VC.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); cell(bad, 'VISION_CENTER')['control']['steps'][0]['commanded_xy_m'] = [.02, 0]
    with pytest.raises(ValueError, match='bound'): VC.validate_scene(bad, CFG)
    with pytest.raises(ValueError, match='C0'): VC.collect_scene(Runtime(VisionEnv(), BiasPolicy((0, 0)), CFG), 1, CFG, arms=('VISION_CENTER',))


# ---------------------------------------------------------------- accounting
def fake(scene, y0, yv, yo=None, acted=True, err=.001, true=.03, decision=None):
    d = decision or ('ACT' if acted else 'ALIGNED'); ge = {'offset_error_m': err, 'true_offset_m': true, 'unnecessary_act': acted and true <= .01, 'missed_large_offset': (not acted) and true >= .02}
    cells = [{'arm': 'C0', 'y': y0}, {'arm': 'VISION_CENTER', 'y': yv, 'acted': acted, 'control': {'decision': d, 'end_reason': 'within_tolerance', 'tcp_xy_displacement_m': .01}, 'control_steps': 4, 'gt_reading_error': ge,
                                      'gt_effect_of_control': {'cubeB_xy_displacement_m': 0.}}]
    if yo is not None: cells.append({'arm': 'CENTER_ORACLE', 'y': yo})
    return {'scene': scene, 'triggered': True, 'cells': cells}


def test_shadow_verdict_criteria():
    ok = [fake(i, 0, 1) for i in range(6)] + [fake(10 + i, 1, 1) for i in range(6)] + [fake(30 + i, 1, 1, acted=False, true=.005) for i in range(20)]
    v = VC.shadow_verdict(ok); assert v['pass'] and v['label'] == 'SHADOW_PASS' and v['acts'] == 12 and v['paired_VISION_CENTER_vs_C0']['net'] == 6
    assert not VC.shadow_verdict(ok[:8])['pass']                                                                    # too few acts
    assert not VC.shadow_verdict(ok + [fake(90 + i, 1, 0) for i in range(3)])['criteria']['harm_at_most']
    assert not VC.shadow_verdict(ok + [fake(95 + i, 1, 1, true=.008) for i in range(3)])['criteria']['unnecessary_acts_at_most']
    assert not VC.shadow_verdict(ok + [fake(99, 1, 1, err=.009)])['criteria']['max_error_at_act']


def test_summarize_needs_every_planned_scene_and_applies_the_gate():
    recs = [fake(i, 0, 1, 1) for i in range(5)] + [fake(10 + i, 1, 1, 1, acted=False, true=.004) for i in range(5)]; s = VC.summarize(recs, [r['scene'] for r in recs])
    assert s['success']['C0']['k'] == 5 and s['success']['VISION_CENTER']['k'] == 10 and s['followup_gate']['label'] == 'VISION_GATE_PASS' and s['oracle_rescues_also_rescued_by_vision'] == {'both': 5, 'oracle': 5}
    with pytest.raises(ValueError, match='planned'): VC.summarize(recs[:-1], [r['scene'] for r in recs])
    assert VC.summarize([fake(i, 0, 0, 1) for i in range(6)], list(range(6)))['followup_gate']['label'] == 'NO_VISION_GATE_PASS'


def test_projected_cost_and_seed_set():
    kw = dict(seconds_per_env_step=.03, seconds_per_reading=1.5, usd_per_hour=.45, spent_usd=.2, scenes=100, arms=3)
    a = VC.projected_cost(cap_usd=2.5, **kw); assert a['fits_cap'] and not VC.projected_cost(cap_usd=.5, **kw)['fits_cap']
    assert VC.projected_cost(cap_usd=2.5, **{**kw, 'seconds_per_reading': 3.})['projected_total_usd'] > a['projected_total_usd']
    man = json.loads(Path('configs/vision_center_splits.json').read_text(encoding='utf-8')); assert VC.pilot_seeds(man, {'x': [50799, 50900]}) == list(range(50800, 50900))
    with pytest.raises(ValueError, match='recorded use'): VC.pilot_seeds(man, {'x': [50850]})
    rv = json.loads(Path('configs/recovery_v1_splits.json').read_text(encoding='utf-8'))['splits']
    assert not set(man['splits']['vision_center_pilot']) & {s for v in rv.values() for s in v}


@pytest.fixture(scope='module')
def X32():
    spec = importlib.util.spec_from_file_location('vc_driver', Path('scripts/32_vision_center.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_every_server_command_refuses_without_an_explicit_authorization(X32, tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir()
    ns = lambda: argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu', usd_per_hour=.4, spent_usd=0., stage='shadow')
    had_ledger = Path(X32.LEDGER).exists()                       # since the pilot pass was executed (2026-11-17, run2) the machine that ran it holds the ledger; a refused command must not create or touch one
    for cmd in (X32.cmd_scenes, X32.cmd_audit, X32.cmd_budget, X32.cmd_shadow, X32.cmd_run):
        with pytest.raises(SystemExit, match='no authorization'): cmd(ns())
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'agent', 'paid_compute_authorized': True, 'cap_usd': 2.5, 'plan_sha256': X32.sha256(plan)}))
    for cmd in (X32.cmd_scenes, X32.cmd_audit, X32.cmd_budget, X32.cmd_shadow, X32.cmd_run):
        with pytest.raises(SystemExit, match='Refused'): cmd(ns())
    assert not [p for p in run.iterdir() if p.is_dir()] and Path(X32.LEDGER).exists() == had_ledger


def test_driver_constants_and_file_list(X32):
    assert X32.shadow_seeds() == list(range(50600, 50800)) and not set(X32.shadow_seeds()) & set(range(50800, 50900))
    assert all(Path(f).exists() for f in X32.CODE) and 'src/hci/vision_center.py' in X32.CODE and X32.MANIFEST in X32.SERVER_READS
    src = Path('scripts/32_vision_center.py').read_text(encoding='utf-8'); assert 'authorization.json' in src and "write_json(run / 'authorization" not in src
    assert X32.SEMANTICS['VISION_CENTER']['act_min_m'] == VC.ACT_MIN_M and X32.ANALYSIS['stage_B1_shadow']['pass_criteria'] == VC.SHADOW_PASS
