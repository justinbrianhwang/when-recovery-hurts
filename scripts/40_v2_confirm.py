"""Vision-CENTER V2 confirmatory Calibration / Final Test driver (work ras_v2_confirm_001). PREPARED, NOT RUN.

  freeze     local: writes frozen/<name>/manifest.json (calibration stage: amendment approval, candidate, family,
             epsilon/delta, code, both policies, configs, splits; test stage additionally the calibration analysis and
             the calibration freeze). Refuses without the user's amendment approval record.
  verify     anywhere: every frozen file still has its frozen sha256
  placement  server, before calibration (simulator, NO policy): initial cube placements of the stage's seeds against
             every placement on record; writes <run>/placement_<stage>.json
  run        server: one pass of one policy over the stage's seeds, arms C0 and VISION_CENTER_V2 from one snapshot per
             scene (hci.vision_center_c1.collect_scene, unchanged). Calibration: exclusive ledger per (stage, policy).
             Test: additionally --approval RUN_FROZEN_TEST and an exclusive TEST_STARTED marker in the frozen folder.
             Never resumed; an error writes <seed>.error.json and stops the pass INCOMPLETE.
  analyze    anywhere: refuses a partial run; calibration -> harm certificate (Holm over the frozen family);
             test -> certificate on the Test scenes (reported) and the efficacy tests

The approval token is supplied by the user for the Final Test only; nothing here creates it.
"""
from __future__ import annotations

import argparse, gzip, importlib.util, json, sys, time, traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from hci import v2_confirm as V  # noqa: E402

CONFIG = 'configs/stackcube.yaml'; SPLITS_JSON = 'configs/splits.json'
POLICY_PATHS = {'train_seed_1': 'artifacts/policies/stack_v1_s1_d500_clean_resnet18_i080000', 'train_seed_2': 'artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000'}
LEDGER_DIR = 'artifacts/ledger'
CODE = ['src/hci/v2_confirm.py', 'scripts/40_v2_confirm.py', 'src/hci/vision_center_c1.py', 'src/hci/vision_center_v2.py', 'src/hci/vision_center.py', 'src/hci/recovery_v1_oracle.py',
        'src/hci/recovery_v1.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/vision_v1.py', 'src/hci/statistics.py', 'src/hci/spatial_probe.py', 'src/hci/io.py', 'src/hci/trees.py',
        'scripts/33_vision_center_v2.py', 'scripts/32_vision_center.py', 'scripts/31_recovery_v1_oracle_xy.py', 'scripts/30_recovery_v1_diag.py', CONFIG, SPLITS_JSON, 'tests/test_v2_confirm.py']


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / file); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def make_runtime(c, policy_path, upstream, device):                   # replaced by the tests with the mock runtime
    from hci import vision_v1 as VV
    return VV.runtime(c, policy_path, upstream, device)


def probe():                                                          # the C1 spatial logger (logging only)
    from hci import spatial_probe as SP
    return SP.gt_probe_spatial


def load_config():
    from hci.io import load_config as lc
    return lc(CONFIG)


def save_scene(out, seed, rec, traces, frames):
    Z = _load('vc2_driver_save', '33_vision_center_v2.py'); Z.save_scene(out, seed, rec, traces, frames)


# ---------------------------------------------------------------- freeze / verify
def cmd_freeze(a):
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists: a freeze is never overwritten')
    appr = V.require_amendment_approval(a.approval_record, a.amendment)
    fam = V.family(a.policies)
    files = [*CODE, a.amendment, a.approval_record, *(f'{POLICY_PATHS[p]}/{f}' for p in a.policies for f in ('weights.pt', 'policy_meta.json'))]
    extra = {'stage': a.stage, 'candidate': V.CANDIDATE, 'arms': list(V.ARMS), 'family': fam, 'epsilon': V.EPSILON, 'delta': V.DELTA, 'alpha_efficacy': V.ALPHA_EFFICACY,
             'splits': {k: list(v) for k, v in V.SPLITS.items()}, 'policies': {p: POLICY_PATHS[p] for p in a.policies}, 'amendment_approval': appr, 'upstream_commit': a.upstream_commit,
             'pip_freeze_sha256': V.sha256(a.pip_freeze) if a.pip_freeze else None, 'note': 'Operational lock against accidental contamination, not access control.'}
    if a.stage == 'test':
        if not (a.calibration_freeze and a.calibration_analysis): raise SystemExit('a test freeze needs the calibration freeze and the calibration analysis')
        files += [a.calibration_freeze, a.calibration_analysis]
        cal = json.loads(Path(a.calibration_analysis).read_text(encoding='utf-8'))
        extra['calibration_result'] = {'joint_statement_all_policies': cal['certificate']['joint_statement_all_policies'], 'members': cal['certificate']['members']}
    if a.pip_freeze: files.append(a.pip_freeze)
    man = V.manifest(files, extra); out.mkdir(parents=True); (out / 'manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    print(json.dumps({'frozen': str(out), 'files': len(man['files_sha256']), 'stage': a.stage}))


def frozen(path):
    man = json.loads((Path(path) / 'manifest.json').read_text(encoding='utf-8')); V.verify_manifest(man); return man


def cmd_verify(a): frozen(a.frozen); print('VERIFIED')


# ---------------------------------------------------------------- run
def cmd_run(a):
    man = frozen(a.frozen); stage = man['stage']; pid = a.policy
    if pid not in man['policies']: raise SystemExit(f'{pid} is not a frozen policy')
    if stage == 'test': V.require_test_approval(a.approval)
    elif a.approval is not None: raise SystemExit('the approval token belongs to the Final Test only')
    seeds = V.split_seeds(SPLITS_JSON, stage) if a.seeds is None else [int(s) for s in a.seeds]          # --seeds only for the tests (synthetic seeds)
    if a.seeds is not None and set(seeds) & set(range(30000, 30400)) | set(seeds) & set(range(90000, 90400)): raise SystemExit('--seeds may not name calibration or test seeds')
    out = Path(a.out) / f'pass_{pid}'
    if out.exists(): raise SystemExit(f'{out} exists: a pass is never resumed or repeated')
    placement = Path(a.out) / f'placement_{stage}.json'
    if a.seeds is None and not (placement.is_file() and json.loads(placement.read_text())['pass']): raise SystemExit('the placement check of this stage has not passed')
    ledger = Path(a.ledger_dir or LEDGER_DIR) / f'ras_v2_confirm_{stage}_{pid}.json'
    if stage == 'test':
        V.create_exclusive(Path(a.frozen) / f'TEST_STARTED_{pid}.json', {'policy': pid, 'out': str(out), 'started_utc': V.now()})
    V.create_exclusive(ledger, {'set': f'{stage}', 'policy': pid, 'seeds': [seeds[0], seeds[-1], len(seeds)], 'frozen_manifest_sha256': V.sha256(Path(a.frozen) / 'manifest.json'), 'used_utc': V.now(),
                                'note': 'single use; written before the first rollout'})
    (out / 'scenes').mkdir(parents=True)
    from hci import vision_center_c1 as C1
    c = load_config(); r = make_runtime(c, man['policies'][pid], a.upstream, a.device); pr = probe(); started = time.time(); done = 0
    try:
        for s in seeds:
            try:
                rec, traces, frames = C1.collect_scene(r, s, c, probe=pr, arms=V.ARMS, role=V.ROLE[stage], policy_id=pid); save_scene(out, s, rec, traces, frames)
            except Exception as e:
                (out / 'scenes' / f'{s}.error.json').write_text(json.dumps({'scene': s, 'completed': False, 'error': repr(e), 'traceback': traceback.format_exc(), 'scenes_finished_before': done}), encoding='utf-8')
                raise SystemExit(f'INCOMPLETE: scene {s} raised {e!r}; the pass stops and is never resumed')
            done += 1; print(f'{stage} {pid} scene={s} done={done}/{len(seeds)} wall={time.time() - started:.0f}s', flush=True)       # outcomes are not printed
    finally:
        r.env.close()
    (out / 'PASS_DONE.json').write_text(json.dumps({'policy': pid, 'scenes': done, 'seconds': time.time() - started, 'frozen_manifest_sha256': V.sha256(Path(a.frozen) / 'manifest.json')}), encoding='utf-8')


# ---------------------------------------------------------------- analyze
def cmd_analyze(a):
    man = frozen(a.frozen); stage = man['stage']; out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    seeds = V.split_seeds(SPLITS_JSON, stage) if a.seeds is None else [int(s) for s in a.seeds]
    from hci import vision_center_c1 as C1
    by = {}; hashes = {}
    for pid in man['policies']:
        d = Path(a.run) / f'pass_{pid}'; V.check_complete(d, seeds)
        if not (d / 'PASS_DONE.json').is_file(): raise SystemExit(f'INCOMPLETE: {d} has no PASS_DONE.json')
        recs = []
        for s in seeds:
            p = d / 'scenes' / f'{s}.json'; rec = json.loads(p.read_text(encoding='utf-8')); C1.validate_scene(rec)
            if V.sha256(d / rec['trace']) != rec['trace_sha256']: raise SystemExit(f'trace of {pid} {s} changed')
            recs.append(rec); hashes[f'{pid}/{s}'] = V.sha256(p)
        by[pid] = V.outcomes(recs)
    res = {'work': V.WORK, 'stage': stage, 'frozen_manifest_sha256': V.sha256(Path(a.frozen) / 'manifest.json'), 'certificate': V.certificate(by, man['family'], man['epsilon'], man['delta']),
           'record_sha256': hashes}
    if stage == 'test': res['efficacy'] = V.efficacy(by, list(man['policies']), man['alpha_efficacy'])
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({'stage': stage, 'joint': res['certificate']['joint_statement_all_policies'], 'members': [(m['policy'], m['n_plus'], m['k_harm'], round(m['p_value'], 6), m['certified']) for m in res['certificate']['members']]}))


# ---------------------------------------------------------------- placement (needs the simulator, no policy)
def placement_check(seeds, label):
    """Initial cube placements of `seeds` (env reset only, no policy) against every placement on record, exactly as the C1
    placement check (scripts/34 cmd_scenes) plus the C1 development scenes. Returns the report dict."""
    X34 = _load('c1_driver_for_keys', '34_vision_center_c1.py')
    from hci.simulation import make_env
    from hci.trees import as_numpy
    from hci.io import read_json
    c = load_config(); env = make_env(c)

    def key(seed):
        env.reset(seed=int(seed)); return np.array([*as_numpy(env.cubeA.pose.p)[0], *as_numpy(env.cubeB.pose.p)[0]], float)
    try:
        D = X34.D; refs = {n: np.array(k) for n, k in read_json(X34.KEYS)['keys'].items()}; sp = read_json(D.SEED_SOURCES['splits'])['splits']
        sets = {'splits/audit': sp['audit'], 'splits/dev': sp['dev'], **{f'policy_eval/{k}': v for k, v in read_json(D.SEED_SOURCES['policy_eval'])['splits'].items()},
                **{f'vision_v1/{k}': v for k, v in read_json(D.SEED_SOURCES['vision_v1'])['splits'].items()}, **{f'{n}/{k}': v for n, p in X34.USED.items() for k, v in read_json(p)['splits'].items()},
                **{f'c1/{k}': v for k, v in read_json(X34.MANIFEST)['splits'].items()}}
        for name, ids in sets.items():
            for s in ids:
                if int(s) not in set(seeds): refs[f'{name}/{s}'] = key(s)
        names = list(refs); table = np.stack([refs[n] for n in names]); own = {}; clashes = []
        for s in seeds:
            k = key(s); dd = np.abs(table - k).max(1); j = int(dd.argmin()); twin = [t for t, v in own.items() if np.abs(v - k).max() <= X34.X.IDENTICAL_M]
            if dd[j] <= X34.X.IDENTICAL_M or twin: clashes.append({'seed': s, 'identical_to': names[j] if dd[j] <= X34.X.IDENTICAL_M else twin[0]})
            own[str(s)] = k
        return {'pass': not clashes, 'set': label, 'identical_placements': clashes, 'seeds_checked': len(own), 'references': len(names), 'identical_threshold_m': X34.X.IDENTICAL_M,
                'scope': 'initial cube placement only; no policy was run'}
    finally:
        env.close()


def cmd_placement(a):
    man = frozen(a.frozen); stage = man['stage']; out = Path(a.out) / f'placement_{stage}.json'
    if out.exists(): raise SystemExit(f'{out} exists')
    rep = placement_check(V.split_seeds(SPLITS_JSON, stage), stage)
    Path(a.out).mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, indent=1), encoding='utf-8'); print(json.dumps({k: rep[k] for k in ('pass', 'seeds_checked', 'references')}))
    if not rep['pass']: raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('freeze'); q.add_argument('--stage', choices=('calibration', 'test'), required=True); q.add_argument('--out', required=True); q.add_argument('--amendment', required=True)
    q.add_argument('--approval-record', required=True); q.add_argument('--policies', nargs='+', default=list(V.POLICY_IDS)); q.add_argument('--upstream-commit', required=True)
    q.add_argument('--pip-freeze'); q.add_argument('--calibration-freeze'); q.add_argument('--calibration-analysis')
    q = s.add_parser('verify'); q.add_argument('--frozen', required=True)
    q = s.add_parser('placement'); q.add_argument('--frozen', required=True); q.add_argument('--out', required=True)
    q = s.add_parser('run'); q.add_argument('--frozen', required=True); q.add_argument('--policy', required=True); q.add_argument('--out', required=True); q.add_argument('--approval')
    q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda'); q.add_argument('--seeds', nargs='*', default=None); q.add_argument('--ledger-dir')
    q = s.add_parser('analyze'); q.add_argument('--frozen', required=True); q.add_argument('--run', required=True); q.add_argument('--out', required=True); q.add_argument('--seeds', nargs='*', default=None)
    a = p.parse_args(); {'freeze': cmd_freeze, 'verify': cmd_verify, 'placement': cmd_placement, 'run': cmd_run, 'analyze': cmd_analyze}[a.cmd](a)


if __name__ == '__main__':
    main()
