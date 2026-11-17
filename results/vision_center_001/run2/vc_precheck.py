"""Thread-limit check before stage B1 (run2). The unchanged perception function is applied to the three stored fixture frames
(policy episodes of the used diagnostic set). Mode `ref`: write the outputs. Mode `check REF OUT`: compare with the reference
(outputs of the same code on the development machine); any different decision, validity flag or a number differing by more
than 1e-9 is a technical failure (exit 2). No simulator, no scene, no outcome."""
import json, os, sys, time
import numpy as np
from hci import vision_center as VC
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'); TOL = 1e-9; STEPS_LEFT = 50


def outputs():
    d = np.load('tests/fixtures/vision_center_frames.npz'); rows = []; secs = []
    for rep in range(2):
        for i in range(3):
            t = time.perf_counter(); rd = VC.reading(d['frames'][i], d['tcp_pose'][i]); secs.append(time.perf_counter() - t)
            if rep == 0: rows.append({'seed': int(d['seeds'][i]), 'decision': VC.decide(rd, STEPS_LEFT)['reason'], **VC.summary(rd)})
    return rows, secs


def diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [x for k in sorted(set(a) | set(b)) for x in (diff(a[k], b[k], f'{path}/{k}') if k in a and k in b else [f'{path}/{k}: missing'])]
    if isinstance(a, list) and isinstance(b, list):
        return [f'{path}: length'] if len(a) != len(b) else [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f'{path}[{i}]')]
    if isinstance(a, float) or isinstance(b, float):
        return [] if (a is not None and b is not None and abs(float(a) - float(b)) <= TOL) else [f'{path}: {a} != {b}']
    return [] if a == b else [f'{path}: {a} != {b}']


if sys.argv[1] == 'ref':
    rows, secs = outputs(); json.dump({'rows': rows, 'made_on': 'development machine, default threads', 'seconds': secs}, open(sys.argv[2], 'w'), indent=1); print(json.dumps([(r['seed'], r['decision'], r['offset_norm_m']) for r in rows]))
else:
    env = {k: os.environ.get(k) for k in THREAD_VARS}; ref = json.load(open(sys.argv[2])); rows, secs = outputs(); problems = [f'{k} is {v}, not 1' for k, v in env.items() if v != '1'] + diff(ref['rows'], json.loads(json.dumps(rows)))
    out = {'pass': not problems, 'problems': problems, 'thread_environment': env, 'cpu_count': os.cpu_count(), 'load_average': getattr(os, 'getloadavg', lambda: None)(), 'tolerance': TOL, 'seconds_per_reading': secs, 'rows': rows,
           'reference': ref['made_on'], 'scope': 'stored frames only; no simulator, no scene, no outcome'}
    json.dump(out, open(sys.argv[3], 'w'), indent=1); print(json.dumps({k: out[k] for k in ('pass', 'problems', 'seconds_per_reading', 'load_average')}))
    if problems: raise SystemExit(2)
