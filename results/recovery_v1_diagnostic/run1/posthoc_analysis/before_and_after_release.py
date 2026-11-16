import json, gzip, sys, math, statistics as st
sys.path.insert(0, 'src')
from hci import recovery_v1 as RV
X = sys.argv[1]
eps = [json.loads(l) for l in open(X + '/episodes.jsonl', encoding='utf-8')]
def tr(e):
    with gzip.open(X + '/' + e['trace'], 'rt', encoding='utf-8') as f: return json.load(f)
xy = lambda r: math.hypot(r['cubeA'][0] - r['cubeB'][0], r['cubeA'][1] - r['cubeB'][1])
out = []
for e in eps:
    t = tr(e); rows = t['rows']; z0 = t['initial']['cubeA'][2]
    seg = RV.holds(rows); car = [(a, b) for a, b in seg if any(rows[i]['cubeA'][2] - z0 >= 0.01 for i in range(a, b + 1))]
    if not car: continue
    a, b = car[-1]
    if b >= len(rows) - 1: continue
    g = lambda k: xy(rows[max(a, b - k)])
    # horizontal action commanded in the last 4 and 8 steps before the release (normalised units), and how long the xy offset has been within 2 mm of its release value
    act = lambda k: [round(sum(abs(rows[i]['action'][j]) for i in range(max(a, b - k + 1), b + 1)) / k, 3) for j in (0, 1, 2)]
    still = next((k for k in range(1, b - a) if abs(xy(rows[b - k]) - xy(rows[b])) > 0.002), None)
    minxy = min(xy(rows[i]) for i in range(a, b + 1))
    out.append(dict(seed=e['seed'], ok=e['success_once'], label=e['label'], b=rows[b]['t'], xy16=g(16), xy8=g(8), xy4=g(4), xy0=g(0), min_xy_in_hold=minxy, steps_xy_settled=still, mean_abs_action_xyz_last8=act(8),
                    width_b=sum(rows[b]['finger_qpos']), chunk_phase=(rows[b]['t']) % 8, after=len(rows) - 1 - b,
                    post_regrasp=len([s for s in seg if s[0] > b]), tcp_z_end=rows[-1]['tcp'][2], grip_end=rows[-1]['action'][-1],
                    tcp_to_A_end=rows[-1]['tcp_to_cubeA']))
F = [x for x in out if not x['ok']]; S_ = [x for x in out if x['ok']]
med = lambda v: round(st.median(v), 4)
for k in ('xy16', 'xy8', 'xy4', 'xy0', 'min_xy_in_hold'):
    print(k, 'success median', med([x[k] for x in S_]), 'failure median', med([x[k] for x in F]))
print('steps the xy offset had been settled (within 2 mm) before release: success median', med([x['steps_xy_settled'] or 99 for x in S_]), 'failure', med([x['steps_xy_settled'] or 99 for x in F]))
print('release step mod 8 (0 = last action of a chunk... ) success', sorted(set(x['chunk_phase'] for x in S_)), 'failure', [x['chunk_phase'] for x in F])
for x in F: print(x['seed'], x['label'][:22], 'b', x['b'], 'xy16/8/4/0', *(f'{x[k]:.4f}' for k in ('xy16', 'xy8', 'xy4', 'xy0')), 'min', f"{x['min_xy_in_hold']:.4f}", 'settled', x['steps_xy_settled'], 'act', x['mean_abs_action_xyz_last8'], 'after', x['after'], 'regrasps', x['post_regrasp'], 'end tcp_z', round(x['tcp_z_end'], 3), 'grip', round(x['grip_end'], 2), 'tcp-A', round(x['tcp_to_A_end'], 3))
print('success after-release steps (always 0-few):', sorted(set(x['after'] for x in S_))[:10])
# the not-static case
e = next(e for e in eps if e['seed'] == 50691); rows = tr(e)['rows']
print('50691:', [(r['t'], round(math.hypot(*r['cubeA_v'][:2]) + abs(r['cubeA_v'][2]), 4), round(math.sqrt(sum(w * w for w in r['cubeA_w'])), 3), r['is_cubeA_static'], r['is_cubeA_grasped'], round(sum(r['finger_qpos']), 3), round(r['f_A_left'] + r['f_A_right'], 2), round(r['action'][-1], 1)) for r in rows[108:200:10]])
# approach failures
for e in eps:
    if e['label'] != 'NO_APPROACH': continue
    t = tr(e); rows = t['rows']; i = min(range(len(rows)), key=lambda k: rows[k]['tcp_to_cubeA']); r = rows[i]
    d = [round(r['tcp'][k] - r['cubeA'][k], 3) for k in range(3)]; A0 = t['initial']['cubeA']
    print(e['seed'], 'closest at t', r['t'], 'tcp-A xyz', d, 'cubeA moved', round(math.hypot(rows[-1]['cubeA'][0] - A0[0], rows[-1]['cubeA'][1] - A0[1]), 3), 'close cmds', sum(1 for x in rows if x['action'][-1] < 0), 'first close', next((x['t'] for x in rows if x['action'][-1] < 0), None),
          'end tcp-A', [round(rows[-1]['tcp'][k] - rows[-1]['cubeA'][k], 3) for k in range(3)], 'end width', round(sum(rows[-1]['finger_qpos']), 3), 'tcp z min', round(min(x['tcp'][2] for x in rows), 3), 'max fingerforce', round(max(x['f_A_left'] + x['f_A_right'] for x in rows), 2))
