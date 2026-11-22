"""Generalization experiment gen_arch_plainconv_001 (PREPARED, NOT RUN): does the UNCHANGED Vision-CENTER V2 rescue a
frozen policy of a DIFFERENT architecture on the same task?

Policy: the Stage 1D official-baseline Diffusion Policy (PlainConv encoder, 500 clean demonstrations, train seed 1, 80k,
`artifacts/policies/stack_1d_s1_d500_clean_v1_i080000`) — frozen since Stage 1D, never adapted to V2. V2, its constants,
its trigger and its rules are the C1 code, unchanged (hci.vision_center_c1.collect_scene).
Scenes: new development seeds 51300-51499 (unused; not calibration or test).
Arms: C0, V2_SINGLE_FRAME, VISION_CENTER_V2, CENTER_ORACLE (privileged reference).

Hypothesis fixed before any rollout (one test, alpha 0.05):
  H_arch: with the PlainConv policy, VISION_CENTER_V2 succeeds in more scenes than C0
          (exact one-sided paired test on discordant scenes).
Reported without a test: success with Wilson intervals, rescue/harm/net per arm vs C0, trigger and act rates, abstention
reasons, oracle rescues recovered by V2, V2 vs single frame, target pushed, development safety criteria of C1
(harm <= 2, memory errors <= 2, target pushed > 1 mm <= 1).

  plan     local: writes <run>/plan.json (hashes of code, policy, config, seeds, hypothesis, cost cap)
  run      server: needs <run>/authorization.json written by the user ({'approved_by': 'user', 'plan_sha256', 'cap_usd'});
           exclusive ledger; one pass; never resumed
  analyze  anywhere: refuses a partial run
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from hci import v2_confirm as V  # noqa: E402  (hashing, exclusive files, exact test, Wilson)
from hci.statistics import wilson_interval  # noqa: E402

WORK = 'gen_arch_plainconv_001'
POLICY = 'artifacts/policies/stack_1d_s1_d500_clean_v1_i080000'
SEEDS = (51300, 51499)
ARMS = ('C0', 'V2_SINGLE_FRAME', 'VISION_CENTER_V2', 'CENTER_ORACLE')
ALPHA = 0.05
SAFETY = {'harm_max': 2, 'memory_errors_max': 2, 'target_pushed_over_1mm_max': 1}
LEDGER = 'artifacts/ledger/gen_arch_plainconv_001.json'
CONFIG = 'configs/stackcube.yaml'
CODE = ['scripts/41_gen_arch_plainconv.py', 'tests/test_gen_arch_plainconv.py', 'src/hci/v2_confirm.py', 'src/hci/vision_center_c1.py', 'src/hci/vision_center_v2.py', 'src/hci/vision_center.py',
        'src/hci/recovery_v1_oracle.py', 'src/hci/recovery_v1.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/vision_v1.py', 'src/hci/statistics.py', 'src/hci/spatial_probe.py',
        'scripts/33_vision_center_v2.py', CONFIG]
COST = {'scene_runs': 200, 'seconds_per_scene_run_assumed': 14.0, 'setup_hours': 0.5, 'usd_per_hour_assumed': 0.36, 'cap_usd': 1.50,
        'basis': 'C1 measured 22.6 s per scene-run with 9 arms; 4 arms assumed 14 s (trunk ~6 s + 4 x ~1.9 s); measured in the first 5 scenes'}


def make_runtime(c, policy_path, upstream, device):
    from hci import vision_v1 as VV
    return VV.runtime(c, policy_path, upstream, device)


def probe():
    from hci import spatial_probe as SP
    return SP.gt_probe_spatial


def load_config():
    from hci.io import load_config as lc
    return lc(CONFIG)


def save_scene(out, seed, rec, traces, frames):
    import importlib.util
    spec = importlib.util.spec_from_file_location('vc2_save', REPO / 'scripts' / '33_vision_center_v2.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.save_scene(out, seed, rec, traces, frames)


def seeds(): return list(range(SEEDS[0], SEEDS[1] + 1))


def cmd_plan(a):
    run = Path(a.run); p = run / 'plan.json'
    if p.exists(): raise SystemExit(f'{p} exists')
    files = [*CODE, f'{POLICY}/weights.pt', f'{POLICY}/policy_meta.json']
    plan = V.manifest(files, {'work': WORK, 'policy': POLICY, 'seeds': list(SEEDS), 'arms': list(ARMS), 'hypothesis': {'H_arch': 'with the PlainConv policy, VISION_CENTER_V2 succeeds in more scenes than C0; exact one-sided paired test; alpha 0.05'},
                              'safety_development_criteria': SAFETY, 'cost': COST, 'data_role': 'development (new seeds 51300-51499); not calibration, not test', 'status': 'PREPARED, NOT RUN'})
    run.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(plan, indent=1), encoding='utf-8'); print(json.dumps({'plan': str(p), 'sha256': V.sha256(p)}))


def require_authorization(run):
    p = Path(run) / 'authorization.json'
    if not p.is_file(): raise SystemExit('Refused: no authorization.json (paid compute not authorized)')
    a = json.loads(p.read_text(encoding='utf-8'))
    if a.get('approved_by') != 'user' or a.get('plan_sha256') != V.sha256(Path(run) / 'plan.json') or not float(a.get('cap_usd', 0)) > 0: raise SystemExit('Refused: the authorization does not name this plan with a positive cap')
    return a


def cmd_placement(a):
    run = Path(a.run); require_authorization(run); out = run / 'placement.json'
    if out.exists(): raise SystemExit(f'{out} exists')
    import importlib.util
    spec = importlib.util.spec_from_file_location('x40p', REPO / 'scripts' / '40_v2_confirm.py'); X40 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X40)
    rep = X40.placement_check(seeds(), WORK); out.write_text(json.dumps(rep, indent=1), encoding='utf-8'); print(json.dumps({k: rep[k] for k in ('pass', 'seeds_checked', 'references')}))
    if not rep['pass']: raise SystemExit(2)


def cmd_run(a):
    run = Path(a.run); require_authorization(run); plan = json.loads((run / 'plan.json').read_text(encoding='utf-8')); V.verify_manifest(plan)
    ss = seeds() if a.seeds is None else [int(s) for s in a.seeds]
    if a.seeds is None and not ((run / 'placement.json').is_file() and json.loads((run / 'placement.json').read_text())['pass']): raise SystemExit('the placement check has not passed')
    if a.seeds is not None and (set(ss) & set(range(30000, 30400)) or set(ss) & set(range(90000, 90400))): raise SystemExit('--seeds may not name calibration or test seeds')
    out = run / 'pass'
    if out.exists(): raise SystemExit(f'{out} exists: never resumed or repeated')
    V.create_exclusive(a.ledger or LEDGER, {'work': WORK, 'seeds': [ss[0], ss[-1], len(ss)], 'plan_sha256': V.sha256(run / 'plan.json'), 'used_utc': V.now()})
    (out / 'scenes').mkdir(parents=True)
    from hci import vision_center_c1 as C1
    c = load_config(); r = make_runtime(c, POLICY, a.upstream, a.device); pr = probe(); t0 = time.time(); done = 0
    try:
        for s in ss:
            try:
                rec, traces, frames = C1.collect_scene(r, s, c, probe=pr, arms=ARMS, role=WORK, policy_id='stage1d_plainconv_s1_80k'); save_scene(out, s, rec, traces, frames)
            except Exception as e:
                (out / 'scenes' / f'{s}.error.json').write_text(json.dumps({'scene': s, 'error': repr(e), 'traceback': traceback.format_exc(), 'finished_before': done}), encoding='utf-8')
                raise SystemExit(f'INCOMPLETE: scene {s} raised {e!r}')
            done += 1; print(f'{WORK} scene={s} done={done}/{len(ss)} wall={time.time() - t0:.0f}s', flush=True)
    finally:
        r.env.close()
    (out / 'PASS_DONE.json').write_text(json.dumps({'scenes': done, 'seconds': time.time() - t0}), encoding='utf-8')


def cmd_analyze(a):
    run = Path(a.run); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    ss = seeds() if a.seeds is None else [int(s) for s in a.seeds]; d = run / 'pass'; V.check_complete(d, ss)
    if not (d / 'PASS_DONE.json').is_file(): raise SystemExit('INCOMPLETE: no PASS_DONE.json')
    from hci import vision_center_c1 as C1
    recs = []
    for s in ss:
        rec = json.loads((d / 'scenes' / f'{s}.json').read_text(encoding='utf-8')); C1.validate_scene(rec)
        if V.sha256(d / rec['trace']) != rec['trace_sha256']: raise SystemExit(f'trace of {s} changed')
        recs.append(rec)
    y = {arm: {r['scene']: int(next(c for c in r['cells'] if c['arm'] == arm)['y']) for r in recs} for arm in ARMS}
    acted = {arm: {r['scene']: bool(next(c for c in r['cells'] if c['arm'] == arm).get('acted')) for r in recs} for arm in ARMS}
    n = len(recs); res = {'work': WORK, 'scenes': n, 'triggered': sum(1 for r in recs if r['triggered']), 'arms': {}}
    for arm in ARMS:
        s_ = sum(y[arm].values()); e = {'success': s_, 'wilson95': list(wilson_interval(s_, n)), 'acted': sum(acted[arm].values())}
        if arm != 'C0':
            e['rescue'] = sum(1 for k in y['C0'] if y['C0'][k] == 0 and y[arm][k] == 1); e['harm'] = sum(1 for k in y['C0'] if y['C0'][k] == 1 and y[arm][k] == 0); e['net'] = e['rescue'] - e['harm']
        res['arms'][arm] = e
    b = res['arms']['VISION_CENTER_V2']['rescue']; c = res['arms']['VISION_CENTER_V2']['harm']; p = V.exact_one_sided(b, c)
    res['H_arch'] = {'v2_only': b, 'c0_only': c, 'p_one_sided': p, 'alpha': ALPHA, 'rejected_null': p <= ALPHA}
    orc = {k for k in y['C0'] if y['C0'][k] == 0 and y['CENTER_ORACLE'][k] == 1}; res['oracle_rescues'] = len(orc); res['oracle_rescues_recovered_by_v2'] = sum(1 for k in orc if y['VISION_CENTER_V2'][k] == 1)
    bb = sum(1 for k in y['C0'] if y['V2_SINGLE_FRAME'][k] == 0 and y['VISION_CENTER_V2'][k] == 1); cc = sum(1 for k in y['C0'] if y['V2_SINGLE_FRAME'][k] == 1 and y['VISION_CENTER_V2'][k] == 0)
    res['v2_vs_single_frame_descriptive'] = {'v2_only': bb, 'single_frame_only': cc}
    res['safety_development_criteria'] = {'harm_v2': c, 'limit': SAFETY, 'note': 'memory errors and target push are read from the records by the C1 mechanism analysis; not recomputed here'}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps({'H_arch': res['H_arch'], 'success': {k: v['success'] for k, v in res['arms'].items()}}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--run', required=True)
    q = s.add_parser('placement'); q.add_argument('--run', required=True)
    q = s.add_parser('run'); q.add_argument('--run', required=True); q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda'); q.add_argument('--seeds', nargs='*', default=None); q.add_argument('--ledger')
    q = s.add_parser('analyze'); q.add_argument('--run', required=True); q.add_argument('--out', required=True); q.add_argument('--seeds', nargs='*', default=None)
    a = p.parse_args(); {'plan': cmd_plan, 'placement': cmd_placement, 'run': cmd_run, 'analyze': cmd_analyze}[a.cmd](a)


if __name__ == '__main__':
    main()
