"""Offline feasibility only: can the horizontal cubeA-cubeB offset be read from the wrist camera while the cube is held near the target?
Frames exist for the failed episodes only. Ground truth is used as the regression target (calibration), never as an input."""
import json, gzip, sys, math
import numpy as np
sys.path.insert(0, 'src')
from hci import recovery_v1 as RV
X = sys.argv[1]
eps = [json.loads(l) for l in open(X + '/episodes.jsonl', encoding='utf-8')]
data = []
for e in eps:
    if e['success_once'] or not e['frames']: continue
    with gzip.open(X + '/' + e['trace'], 'rt', encoding='utf-8') as f: t = json.load(f)
    rows = t['rows']; z0 = t['initial']['cubeA'][2]; fr = np.load(X + '/' + e['frames'])['frames'][:, :, 128:, :].astype(int)   # hand camera
    for i, r in enumerate(rows):
        d = RV.derived(r)
        if not r['is_cubeA_grasped'] or r['cubeA'][2] - z0 < 0.01 or d['xy'] > 0.06 or d['h'] > 0.03: continue
        im = fr[i + 1]; R, G, B = im[..., 0], im[..., 1], im[..., 2]
        g = (G > 120) & (R < 90) & (B < 90); rd = (R > 120) & (G < 60) & (B < 60)
        if g.sum() < 5 or rd.sum() < 20: feats = None
        else:
            v, u = np.nonzero(g); rv, ru = np.nonzero(rd)
            feats = [u.mean(), v.mean(), g.sum(), u.min(), u.max(), v.min(), v.max(), ru.mean(), rv.mean(), rd.sum()]
        data.append((e['seed'], r['t'], feats, r['cubeA'][0] - r['cubeB'][0], r['cubeA'][1] - r['cubeB'][1], d['h'], r['tcp_q']))
print('frames while holding within 6 cm and 3 cm above rest:', len(data), 'episodes', len({d[0] for d in data}), 'green not visible:', sum(1 for d in data if d[2] is None))
q = np.array([d[6] for d in data]); print('gripper orientation spread (max abs dev of quaternion):', float(np.abs(q - q.mean(0)).max()))
D = [d for d in data if d[2] is not None]; seeds = sorted({d[0] for d in D})
def design(F):
    F = np.asarray(F, float); F = (F - np.array([64, 64, 300, 64, 64, 64, 64, 64, 64, 3000])) / np.array([64, 64, 300, 64, 64, 64, 64, 64, 64, 3000])
    return np.hstack([np.ones((len(F), 1)), F, F[:, :2] ** 2, F[:, :1] * F[:, 1:2]])
err = []; per = {}
for s in seeds:
    tr_ = [d for d in D if d[0] != s]; te = [d for d in D if d[0] == s]
    A = design([d[2] for d in tr_]); Y = np.array([[d[3], d[4]] for d in tr_]); W = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y)
    P = design([d[2] for d in te]) @ W; E_ = np.linalg.norm(P - np.array([[d[3], d[4]] for d in te]), axis=1); err += list(E_); per[s] = (len(te), float(np.median(E_)))
err = np.array(err); print('leave-one-episode-out error of the estimated xy offset (m): median %.4f, 75%% %.4f, 90%% %.4f, max %.4f, n=%d' % (np.median(err), np.quantile(err, .75), np.quantile(err, .9), err.max(), len(err)))
print('true offset size in these frames: median %.4f' % np.median([math.hypot(d[3], d[4]) for d in D]))
print(per)
