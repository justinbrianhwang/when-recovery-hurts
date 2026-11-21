"""Integration tests of the C1 run entry point `cmd_run` (scripts/34_vision_center_c1.py), from the authorization check to the
summary, including a real SIGALRM budget deadline.

What runs unchanged: require_authorization, the gate-file checks, the ledger and pass-folder checks, the start projection,
the exclusive ledger, run_passes (arm_deadline with signal.setitimer and the real handler), passes (the per-scene budget
check, error files), C1.collect_scene and Z.save_scene, cmd_summarize (check_run_gates, load_records, C1.summarize).

What is replaced, and only this:
  * LEDGER points into the test's temporary folder (the real artifacts/ledger is never touched; checked);
  * runtime_for returns the mock runtime of the other C1 tests (mock physics, synthetic renderer, scripted policy);
  * X.gt_probe and the driver's PROBE (the spatial logger) are the mock probe; C1.pilot_seeds returns synthetic seeds 7, 8, 9 (no C1 scene 51100-51299 is used);
  * bound() returns the test plan (a copy of plan_v2.json), the mock config and the nine arms; the real bound() is
    tested separately on the real plan;
  * the placement check, the logging ON/OFF check and the server audit (they need ManiSkill and the policy) are written by
    the test as passed gate files (the ON/OFF check itself is tested in tests/test_spatial_probe.py); the
    session clock and the budget gate are written by the real cmd_clock and cmd_budget.

Linux only where SIGALRM is needed (signal.setitimer); elsewhere those tests are SKIPPED, never passed.
Set C1_IT_EVIDENCE=<dir> to keep a file list with sha256 of every run folder the tests create.
"""
import argparse, hashlib, importlib.util, json, os, signal, time, traceback
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from test_recovery_v1_oracle import CFG
from test_vision_center import BiasPolicy
from test_vision_center_v2 import VisionEnv2
from test_trajectory_audit import probe2
from hci.simulation import Runtime
from hci import vision_center_c1 as C1

LINUX = hasattr(signal, 'setitimer')
needs_alarm = pytest.mark.skipif(not LINUX, reason='SIGALRM deadline needs signal.setitimer (Linux); not verified on this platform')
SEEDS = [7, 8, 9]
REAL_PLAN = Path('results/vision_center_v2_c1_001/preparation_v3/plan.json')        # the current C1 plan (Option A: spatial logging); plan_v2.json (dd3af27e) is superseded
REAL_LEDGER = Path('artifacts/ledger/vision_center_c1.json')
REAL_RUN = Path('runs/vision_center_v2_c1_001')


class Env(VisionEnv2):
    closed = 0
    def close(self): Env.closed += 1


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree(d): return {p.relative_to(d).as_posix(): sha(p) for p in sorted(Path(d).rglob('*')) if p.is_file()} if Path(d).exists() else {}
def ns(plan, run, **k): return argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu', **k)


@pytest.fixture(scope='module')
def X34():
    spec = importlib.util.spec_from_file_location('c1_driver_it', Path('scripts/34_vision_center_c1.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture
def rig(X34, tmp_path, monkeypatch, request):
    """A temporary run folder with every gate before `run` passed, for the test plan. Returns a small helper object."""
    before = {'ledger': REAL_LEDGER.exists(), 'real_run': tree(REAL_RUN)}
    plan_path = tmp_path / 'plan.json'; plan_path.write_bytes(REAL_PLAN.read_bytes() + b'\n')          # a different file: its own hash
    plan = json.loads(plan_path.read_text(encoding='utf-8')); arms = tuple(C1.BASE_ARMS); assert list(arms) == plan['arm_registry']['arms']
    ledger = tmp_path / 'artifacts' / 'ledger' / 'vision_center_c1.json'
    monkeypatch.setattr(X34, 'LEDGER', str(ledger)); monkeypatch.setattr(X34, 'bound', lambda p: (json.loads(Path(p).read_text(encoding='utf-8')), CFG, arms))
    monkeypatch.setattr(X34, 'runtime_for', lambda c, pid, a: Runtime(Env(), BiasPolicy((.028, .012) if pid == 'train_seed_1' else (-.028, .0)), CFG))
    monkeypatch.setattr(X34.X, 'gt_probe', probe2); monkeypatch.setattr(X34, 'PROBE', probe2); monkeypatch.setattr(C1, 'pilot_seeds', lambda manifest, used: list(rig.seeds))
    run = tmp_path / 'run'; run.mkdir(); events = []; real_collect = C1.collect_scene

    def collect(r, s, c, **kw):                                                   # calls the real collector; records where an exception left it
        events.append(('start', int(s), time.time()))
        if rig.inject.get(('start', int(s))): rig.inject[('start', int(s))]()
        try: out = real_collect(r, s, c, **kw)
        except BaseException as e:
            tb = traceback.extract_tb(e.__traceback__); events.append(('interrupted', int(s), type(e).__name__, sum(1 for f in tb if '/src/hci/' in f.filename.replace('\\', '/')))); raise
        events.append(('end', int(s), time.time()))
        if rig.inject.get(('end', int(s))): rig.inject[('end', int(s))]()
        return out
    monkeypatch.setattr(C1, 'collect_scene', collect)

    class Rig:
        pass
    rig = Rig(); rig.plan, rig.run, rig.ledger, rig.events, rig.inject, rig.seeds, rig.arms, rig.h = plan_path, run, ledger, events, {}, SEEDS[:2], arms, X34.sha256(plan_path)

    def ready(cap=6.0, usd_per_hour=.46, seeds=None):
        if seeds: rig.seeds = seeds
        (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': cap, 'plan_sha256': rig.h}))
        (run / 'scenes.json').write_text(json.dumps({'pass': True, 'plan_sha256': rig.h, 'note': 'written by the test: the placement check needs ManiSkill'}))
        (run / 'probe_check').mkdir(); (run / 'probe_check' / 'probe_check.json').write_text(json.dumps({'pass': True, 'plan_sha256': rig.h, 'note': 'written by the test: the ON/OFF check needs the policy'}))
        (run / 'audit').mkdir(); (run / 'audit' / 'audit.json').write_text(json.dumps({'pass': True, 'plan_sha256': rig.h, 'arms': list(arms), 'seconds_per_env_step': 1e-6, 'seconds_per_reading': 1e-6, 'seconds_per_memory_step': 1e-6,
                                                                                    'note': 'written by the test: the server audit needs ManiSkill'}))
        X34.cmd_clock(ns(plan_path, run, usd_per_hour=usd_per_hour, spent_before_usd=0.)); X34.cmd_budget(ns(plan_path, run))
        assert json.loads((run / 'budget_gate.json').read_text())['fits_cap']
    rig.ready = ready
    yield rig
    if LINUX: signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, signal.SIG_DFL)
    assert REAL_LEDGER.exists() == before['ledger'] and tree(REAL_RUN) == before['real_run'], 'the test touched the real C1 ledger or run folder'
    ev = os.environ.get('C1_IT_EVIDENCE')
    if ev:
        Path(ev).mkdir(parents=True, exist_ok=True)
        (Path(ev) / f'{request.node.name}.json').write_text(json.dumps({'test': request.node.name, 'events': [[e[0], e[1], *[x for x in e[2:] if not isinstance(x, float)]] for e in events], 'files_sha256': {'run/' + k: v for k, v in tree(run).items()},
                                                                       'ledger_in_tmp': sha(ledger) if ledger.exists() else None}, indent=1))


def timer_left(): return signal.getitimer(signal.ITIMER_REAL)[0] if LINUX else 0.


def scenes_written(out): return sorted(int(p.stem) for p in (Path(out) / 'scenes').glob('*.json') if not p.name.endswith('.error.json'))


# ---------------------------------------------------------------- 1, 7: a normal mock run, then every way of running it again
@needs_alarm
def test_1_normal_run_ends_with_a_summary_and_7_a_second_run_is_refused(X34, rig):
    rig.ready(seeds=[7, 8]); X34.cmd_run(ns(rig.plan, rig.run))
    res = json.loads((rig.run / 'result.json').read_text()); assert res['plan_sha256'] == rig.h and list(res['per_policy']) == list(C1.POLICIES)
    for pid in C1.POLICIES:
        assert scenes_written(rig.run / f'pass_{pid}') == [7, 8] and all(res['per_policy'][pid]['success'][a]['n'] == 2 for a in rig.arms)
    assert not list(rig.run.rglob('budget_stop.json')) and not list(rig.run.rglob('*.error.json')) and timer_left() == 0.
    assert json.loads((rig.run / 'budget_deadline.json').read_text())['seconds_left_at_start'] > 3600 and (rig.run / 'ledger.json').read_text() == rig.ledger.read_text()
    assert [e[0] for e in rig.events] == ['start', 'end'] * 4 and Env.closed >= 2
    before = tree(rig.run)
    with pytest.raises(SystemExit, match='exists'): X34.cmd_run(ns(rig.plan, rig.run))                       # the ledger exists
    rig.ledger.unlink()
    with pytest.raises(SystemExit, match='never resumed'): X34.cmd_run(ns(rig.plan, rig.run))                 # without the ledger, the pass folders still block
    assert tree(rig.run) == before and not rig.ledger.exists() and len(rig.events) == 8


# ---------------------------------------------------------------- 2, 5, 6: with honest timing the gates stop the run before the clock deadline
@needs_alarm
def test_2_5_6_with_real_scene_times_the_five_scene_gate_stops_the_run_before_the_clock_deadline(X34, rig):
    """cap 1.0, US$2.10/h: the start gate passes (0.30 + 1125 s fixed overhead = 0.96) and the clock deadline is 1200 s away.
    After five scenes the gate projects the remaining 155 scenes from their measured time (it blocks for any mock scene time above 6.2 s; they take 25-70 s) and stops
    the run INCOMPLETE. In this fixture the gates stop the run before the clock deadline; that holds only while later scenes
    are not slower than the first five, so it is not a proof that the gates always come first. The deadline is the backstop
    for scenes that become slower after the gate, and its in-scene handling is tested in test_3_4. Nothing is shortened or
    changed by this test."""
    rig.ready(cap=1.0, usd_per_hour=(1.0 - X34.RESERVE_USD) * 3600 / 1200., seeds=list(range(7, 87)))
    with pytest.raises(SystemExit, match='BUDGET_BLOCKED after five scenes.*INCOMPLETE'): X34.cmd_run(ns(rig.plan, rig.run))
    out = rig.run / 'pass_train_seed_1'; g = json.loads((out / 'budget_after_five_scenes.json').read_text())
    assert not g['fits_cap'] and g['scene_runs_still_to_do'] == 155 and g['plan_sha256'] == rig.h and scenes_written(out) == [7, 8, 9, 10, 11]
    assert [e[:2] for e in rig.events] == [(k, s) for s in range(7, 12) for k in ('start', 'end')] and not (rig.run / 'pass_train_seed_2').exists() and timer_left() == 0.
    assert json.loads((rig.run / 'budget_deadline.json').read_text())['seconds_left_at_start'] > 1100 and rig.ledger.exists() and not (rig.run / 'result.json').exists()
    with pytest.raises(SystemExit, match='INCOMPLETE: no record'): X34.cmd_summarize(ns(rig.plan, rig.run, out=str(rig.run / 'result.json')))
    with pytest.raises(SystemExit, match='exists'): X34.cmd_run(ns(rig.plan, rig.run))


@needs_alarm
def test_2_a_clock_deadline_that_the_gates_would_not_allow_never_starts(X34, rig):
    """A deadline 12 s after the clock: the budget gate refuses (fixed overhead), so no ledger, no scene, no timer."""
    with pytest.raises(SystemExit): rig.ready(cap=1.0, usd_per_hour=(1.0 - X34.RESERVE_USD) * 3600 / 12., seeds=[7, 8])
    assert not json.loads((rig.run / 'budget_gate.json').read_text())['fits_cap']
    with pytest.raises(SystemExit, match='not passed for this plan'): X34.cmd_run(ns(rig.plan, rig.run))
    assert not rig.ledger.exists() and not list(rig.run.glob('pass_*')) and not rig.events and timer_left() == 0.


# ---------------------------------------------------------------- 3, 4: a deadline inside the second scene keeps the first
@needs_alarm
def test_3_4_a_deadline_inside_the_second_scene_keeps_the_first_and_starts_no_other(X34, rig):
    """The run's own timer is armed by run_passes for hours; at the start of the second scene the test shortens it to 4 s, so
    SIGALRM lands inside that scene. The handler, the exception path and every file write are the code under test."""
    rig.ready(seeds=[7, 8, 9]); armed = {}
    def shorten(): armed['left'] = signal.getitimer(signal.ITIMER_REAL)[0]; armed['at'] = time.time(); signal.setitimer(signal.ITIMER_REAL, 4.0)
    rig.inject[('start', 8)] = shorten
    with pytest.raises(SystemExit, match='BUDGET_STOP.*INCOMPLETE'): X34.cmd_run(ns(rig.plan, rig.run))
    out = rig.run / 'pass_train_seed_1'; stop = json.loads((rig.run / 'budget_stop.json').read_text())
    k = json.loads((rig.run / 'session_clock.json').read_text()); expect = X34.deadline_seconds_left(k, 6.0, now=armed['at'])
    assert abs(armed['left'] - expect) < 2. and json.loads((rig.run / 'budget_deadline.json').read_text())['seconds_left_at_start'] > armed['left']      # the run's own timer came from the session clock
    assert stop['scene'] == 8 and stop['scenes_finished_in_pass'] == 1 and [e[:2] for e in rig.events] == [('start', 7), ('end', 7), ('start', 8), ('interrupted', 8)] and rig.events[3][2] == 'BudgetDeadline' and rig.events[3][3] > 0
    assert scenes_written(out) == [7] and not (rig.run / 'pass_train_seed_2').exists() and timer_left() == 0.
    rec = json.loads((out / 'scenes' / '7.json').read_text()); assert rec['completed'] and rec['plan_sha256'] == rig.h and X34.sha256(out / rec['trace']) == rec['trace_sha256']
    with pytest.raises(SystemExit, match='budget stop'): X34.cmd_summarize(ns(rig.plan, rig.run, out=str(rig.run / 'result.json')))
    assert not (rig.run / 'result.json').exists()


# ---------------------------------------------------------------- the per-scene budget check before a scene
@needs_alarm
def test_the_budget_check_before_a_scene_stops_the_run_and_keeps_it_incomplete(X34, rig):
    """After the first scene the test moves the session clock's start back by 13 h (as if the scene had taken that long at
    US$0.46/h): the check before the next scene must stop the run without starting it."""
    rig.ready(seeds=[7, 8, 9])
    def age():
        p = rig.run / 'session_clock.json'; k = json.loads(p.read_text()); k['started_epoch'] -= 13 * 3600; p.write_text(json.dumps(k))
    rig.inject[('end', 7)] = age
    with pytest.raises(SystemExit, match='BUDGET_STOP: one more scene'): X34.cmd_run(ns(rig.plan, rig.run))
    out = rig.run / 'pass_train_seed_1'; stop = json.loads((out / 'budget_stop.json').read_text())
    assert stop['before_scene'] == 8 and stop['scenes_finished'] == 1 and [e[:2] for e in rig.events] == [('start', 7), ('end', 7)] and scenes_written(out) == [7] and timer_left() == 0.
    with pytest.raises(SystemExit, match='budget stop'): X34.cmd_summarize(ns(rig.plan, rig.run, out=str(rig.run / 'result.json')))


# ---------------------------------------------------------------- 9: any other exception fails closed the same way
@needs_alarm
def test_9_an_ordinary_error_in_a_scene_fails_closed(X34, rig):
    rig.ready(seeds=[7, 8, 9])
    def boom(): raise RuntimeError('injected by the test: a technical error in scene 8')
    rig.inject[('start', 8)] = boom
    with pytest.raises(RuntimeError, match='injected'): X34.cmd_run(ns(rig.plan, rig.run))
    out = rig.run / 'pass_train_seed_1'; err = json.loads((out / 'scenes' / '8.error.json').read_text())
    assert err['completed'] is False and 'injected' in err['error'] and err['scenes_finished_before'] == 1 and scenes_written(out) == [7]
    assert [e[:2] for e in rig.events] == [('start', 7), ('end', 7), ('start', 8)] and not (rig.run / 'pass_train_seed_2').exists() and timer_left() == 0. and rig.ledger.exists()
    assert not (rig.run / 'result.json').exists() and not (rig.run / 'budget_stop.json').exists()
    with pytest.raises(SystemExit, match='INCOMPLETE: error files'): X34.cmd_summarize(ns(rig.plan, rig.run, out=str(rig.run / 'result.json')))
    with pytest.raises(SystemExit, match='exists'): X34.cmd_run(ns(rig.plan, rig.run))


@needs_alarm
def test_a_logging_error_inside_a_scene_fails_the_run_closed(X34, rig, monkeypatch):
    """The spatial logger raises during scene 8 (as hci.spatial_probe does for a shape it cannot read): the scene gets an
    error file, no later scene starts, the run is INCOMPLETE and is not summarised; no row without the logged fields."""
    rig.ready(seeds=[7, 8, 9]); state = {}
    def flaky(rt, z):
        if rt.episode == 8 and rt.t == 5: raise RuntimeError('probe_spatial: contact shape not found (injected by the test)')
        return probe2(rt, z)
    monkeypatch.setattr(X34, 'PROBE', flaky)
    with pytest.raises(RuntimeError, match='contact shape not found'): X34.cmd_run(ns(rig.plan, rig.run))
    out = rig.run / 'pass_train_seed_1'; err = json.loads((out / 'scenes' / '8.error.json').read_text())
    assert err['completed'] is False and 'contact shape not found' in err['error'] and scenes_written(out) == [7] and not (out / 'traces' / 'trace_8.json.gz').exists()
    assert [e[:2] for e in rig.events if e[0] != 'interrupted'] == [('start', 7), ('end', 7), ('start', 8)] and not (rig.run / 'pass_train_seed_2').exists() and not (rig.run / 'result.json').exists()
    with pytest.raises(SystemExit, match='INCOMPLETE: error files'): X34.cmd_summarize(ns(rig.plan, rig.run, out=str(rig.run / 'result.json')))


# ---------------------------------------------------------------- 8: a missing or wrong approval, a wrong plan
def test_8_no_approval_a_foreign_plan_hash_or_a_changed_plan_is_refused_before_anything_is_written(X34, rig):
    rig.ready(seeds=[7, 8])
    good = (rig.run / 'authorization.json').read_text(); before = tree(rig.run)
    cases = [('missing', None, 'no authorization'),
             ('another plan', {**json.loads(good), 'plan_sha256': '0' * 64}, 'another plan'),
             ('not the user', {**json.loads(good), 'approved_by': 'agent'}, 'not an explicit user approval'),
             ('no cap', {**json.loads(good), 'cap_usd': 0}, 'not an explicit user approval'),
             ('not authorized', {**json.loads(good), 'paid_compute_authorized': False}, 'not an explicit user approval')]
    for name, auth, msg in cases:
        if auth is None: (rig.run / 'authorization.json').unlink()
        else: (rig.run / 'authorization.json').write_text(json.dumps(auth))
        with pytest.raises(SystemExit, match=msg): X34.cmd_run(ns(rig.plan, rig.run))
        assert not rig.ledger.exists() and not list(rig.run.glob('pass_*')) and not rig.events, name
    (rig.run / 'authorization.json').write_text(good)
    rig.plan.write_bytes(rig.plan.read_bytes() + b' ')                                                    # the plan changed after the approval
    with pytest.raises(SystemExit, match='another plan'): X34.cmd_run(ns(rig.plan, rig.run))
    rig.plan.write_bytes(rig.plan.read_bytes()[:-1]); (rig.run / 'scenes.json').write_text(json.dumps({'pass': True, 'plan_sha256': '1' * 64}))
    with pytest.raises(SystemExit, match='not passed for this plan'): X34.cmd_run(ns(rig.plan, rig.run))     # a gate of another plan
    assert not rig.ledger.exists() and not list(rig.run.glob('pass_*')) and not rig.events and set(tree(rig.run)) == set(before)


def test_8_the_real_bound_refuses_a_plan_whose_code_hash_differs(X34, tmp_path):
    """The real bound() on the real plan, then on a copy with one code hash changed. Needs the frozen policy files."""
    plan = json.loads(REAL_PLAN.read_text(encoding='utf-8'))
    if not all((Path(p['path']) / 'weights.pt').exists() for p in plan['policies'].values() if isinstance(p, dict) and 'path' in p): pytest.skip('frozen policy files are not on this machine')
    X34.bound(str(REAL_PLAN))                                                                              # the real plan binds the working tree
    bad = dict(plan); bad['code'] = {**plan['code'], 'src/hci/vision_center_v2.py': '0' * 64}; p = tmp_path / 'plan.json'; p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match='differ from the plan'): X34.bound(str(p))
