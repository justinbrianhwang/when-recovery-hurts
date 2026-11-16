import json, gzip, sys, math, statistics as st
sys.path.insert(0, 'src')
from hci import recovery_v1 as RV
X = sys.argv[1]
eps = [json.loads(l) for l in open(X + '/episodes.jsonl', encoding='utf-8')]
def tr(e):
    with gzip.open(X + '/' + e['trace'], 'rt', encoding='utf-8') as f: return json.load(f)
rel = []
for e in eps:
    t = tr(e); rows = t['rows']; z0 = t['initial']['cubeA'][2]
    seg = RV.holds(rows); car = [(a, b) for a, b in seg if any(rows[i]['cubeA'][2] - z0 >= 0.01 for i in range(a, b + 1))]
    if not car: continue
    a, b = car[-1]; r = rows[b]; d = RV.derived(r)
    # offset of cubeA from cubeB and of TCP from cubeA at the last grasped step; descent speed of cubeA over the last 3 grasped steps
    vz = (rows[b]['cubeA'][2] - rows[max(a, b - 3)]['cubeA'][2]) / max(1, b - max(a, b - 3)) * 20
    tcpxy = math.hypot(r['tcp'][0] - r['cubeB'][0], r['tcp'][1] - r['cubeB'][1])
    grasp_off = math.hypot(r['tcp'][0] - r['cubeA'][0], r['tcp'][1] - r['cubeA'][1])
    # first open command at or after hold start
    op = next((rows[i]['t'] for i in range(a, len(rows)) if rows[i]['action'][-1] > 0), None)
    io = None if op is None else op - 1                       # index of the row of the first open command
    pre = None if io is None or io < 1 else RV.derived(rows[io - 1])   # state BEFORE the first open command was executed
    rel.append(dict(seed=e['seed'], ok=e['success_once'], label=e['label'], t_rel=r['t'], xy=d['xy'], h=d['h'], vz=vz, tcp_xy_B=tcpxy, grasp_off=grasp_off, tilt=RV.tilt_deg(r['cubeA_q']), fAB=r['f_A_B'],
                    open_step=op, xy_pre=None if pre is None else pre['xy'], h_pre=None if pre is None else pre['h'], on_pre=None if io is None or io < 1 else rows[io - 1]['is_cubeA_on_cubeB'], ended_holding=b == len(rows) - 1))
f = lambda v: 'None' if v is None else f'{v:.4f}'
S_ = [x for x in rel if x['ok']]; F = [x for x in rel if not x['ok']]
q = lambda v: [round(z, 4) for z in (min(v), st.quantiles(v, n=4)[0], st.median(v), st.quantiles(v, n=4)[2], max(v))]
print('successes with a carry', len(S_), 'failures with a carry', len(F))
for k in ('xy', 'h', 'xy_pre', 'h_pre', 'vz', 'tcp_xy_B', 'grasp_off', 'tilt'):
    print(k, 'success min/q1/med/q3/max', q([x[k] for x in S_ if x[k] is not None]), '| failure', q([x[k] for x in F if x[k] is not None]))
for thr in (0.010, 0.015, 0.0175, 0.020, 0.0225, 0.025, 0.030):
    a = [x for x in rel if x['xy_pre'] is not None and x['xy_pre'] > thr]
    print(f'xy before first open command > {thr}: episodes {len(a)}, failures {sum(1 for x in a if not x["ok"])}; below: {sum(1 for x in rel if x["xy_pre"] is not None and x["xy_pre"] <= thr)} episodes, failures {sum(1 for x in rel if x["xy_pre"] is not None and x["xy_pre"] <= thr and not x["ok"])}')
print('--- failures')
for x in sorted(F, key=lambda x: x['label']): print(x['seed'], x['label'], 't_rel', x['t_rel'], 'open', x['open_step'], 'xy', f(x['xy']), 'h', f(x['h']), 'xy_pre', f(x['xy_pre']), 'h_pre', f(x['h_pre']), 'vz', f(x['vz']), 'grasp_off', f(x['grasp_off']), 'tilt', f(x['tilt']))
print('--- successes with the largest xy before the open command')
for x in sorted(S_, key=lambda x: -(x['xy_pre'] or 0))[:12]: print(x['seed'], 't_rel', x['t_rel'], 'open', x['open_step'], 'xy_pre', f(x['xy_pre']), 'h_pre', f(x['h_pre']), 'xy', f(x['xy']), 'h', f(x['h']))
json.dump(rel, open(sys.argv[2], 'w'), indent=1)
