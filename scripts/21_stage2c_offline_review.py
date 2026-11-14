"""Stage 2C: offline review of the saved Stage 2B records (run id stage2c_001). See amendments/AMENDMENT_007.

CPU only. Reads CSV/JSON/NPZ that Stage 2B wrote; runs no simulator, no policy inference, no training, no rendering.
Post hoc development analysis, planned after the Stage 2B aggregates and some cases were seen. No p-value, no interval,
no threshold search, no selector is trained. Nothing here is the performance of an executable selector.

  manifest   hashes of every input and of this file, the fixed rules; written BEFORE any new result is computed
  run        A recount from the raw rows, B per-snapshot counts, C leave-one-repeat-out choice diagnostic,
             D event sequence around the retreat; refuses if an input or this file differs from the manifest
  verify     inputs still have the hashes of the manifest

Time axis of a trace (src/hci/pilot2b.py, Hooks.advance): row i of `trunk` is the state AFTER control step i+1
(column t = i+1); row i of branch `b_<checkpoint>_<repeat>_<arm>` is the state after step checkpoint+i+1.
The state AT the checkpoint is trunk row checkpoint-1. Flags and positions are simulator ground truth (logging only).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, platform, sys
import numpy as np

ARMS = ('C0', 'C1', 'F', 'R')
ARM_NAMES = {'C0': 'continue', 'C1': 'continue_alt_noise', 'F': 'refresh', 'R': 'retreat_retry'}
PAIRS = (('F', 'C0'), ('R', 'C0'), ('C1', 'C0'), ('F', 'C1'), ('R', 'C1'), ('R', 'F'))
REPEATS = (0, 1, 2, 3)
RULES = {'G_CF': ('C0', 'F'), 'G_CR': ('C0', 'R'), 'G_CFR': ('C0', 'F', 'R')}      # C1 is never a candidate
PRIORITY = ('C0', 'F', 'R')
ALWAYS = ('C0', 'C1', 'F', 'R')
# thresholds of the existing diagnostic labels (src/hci/diagnostics.py at the Stage 2B code hash); copied, not tuned
NEAR_M, ALIGN_M, LIFT_M = 0.03, 0.02, 0.01
B_MOVED_M = 0.002          # cube B displacement reported as "moved"; a description of position only, not a collision
COL = {n: i for i, n in enumerate(('t', 'tcp_x', 'tcp_y', 'tcp_z', 'cubeA_x', 'cubeA_y', 'cubeA_z', 'cubeB_x', 'cubeB_y', 'cubeB_z',
                                   'is_cubeA_grasped', 'is_cubeA_on_cubeB', 'is_cubeA_static', 'success', 'a_x', 'a_y', 'a_z', 'a_gripper'))}
EXPECTED = {'scenes': 24, 'snapshots': 84, 'blocks': 336, 'continuations': 1344, 'success': {'C0': 93, 'C1': 98, 'F': 94, 'R': 39},
            'sets': {'C0=1,R=0': (69, 13), 'C0=0,R=1': (15, 5), 'C0=1,F=1,R=0': (53, 11), 'C0=0,F=0,R=1': (12, 5), 'C0=C1=F=0,R=1': (8, 4)}}
EVENT_FLAGS = ('grasp_loss_during_retreat', 'grasp_kept_through_retreat', 'grasp_loss_after_return', 'grasp_loss_after_return_while_on_cubeB',
               'lift_lost_during_retreat', 'alignment_lost_during_retreat', 'near_lost_during_retreat', 'no_reapproach_before_horizon',
               'never_grasped_after_return', 'retained_grasp_no_success_by_horizon', 'aligned_after_return_no_success', 'success_after_return',
               'ended_during_retreat',
               'cubeB_moved_during_retreat', 'cubeB_moved_after_return', 'ended_at_horizon', 'insufficient_trace')
NOT_CLAIMED = ['a grasp flag that turns off is not "the cube was dropped" (a normal placement also turns it off)',
               'a failure at step 200 is not "the time spent on the retreat was the only cause"',
               'retreat steps used is not "that many extra steps would have given success"',
               'no out-of-distribution or off-manifold cause is established', 'cube B displacement is not a confirmed collision (no force or contact data)']


# ---------------------------------------------------------------- strict reading
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def bit(v):
    """0/1 only. 'False', '', 2, 0.5 ... are errors, never truthy guesses."""
    if isinstance(v, bool): raise ValueError(f'outcome must be 0 or 1, got {v!r}')
    if isinstance(v, (int, np.integer)) and int(v) in (0, 1): return int(v)
    if isinstance(v, str) and v in ('0', '1'): return int(v)
    raise ValueError(f'outcome must be 0 or 1, got {v!r}')


def boolean(v):
    if isinstance(v, bool): return v
    if v in ('True', 'False'): return v == 'True'
    raise ValueError(f'boolean must be True or False, got {v!r}')


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    p = Path(path)
    with p.open('x', encoding='utf-8') as f: f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- A: join and recount
def join_blocks(csv_rows, records):
    """One block per (scene, checkpoint, block_repeat_id), present once in the CSV and once in the scene JSON, with
    identical outcomes and snapshot. Any duplicate, gap or conflict is an error; nothing is dropped to make it fit."""
    js = {}; snap_of = {}
    for rec in records:
        ids = {s['checkpoint']: s['snapshot_id'] for s in rec['snapshots']}
        if len(ids) != len(rec['snapshots']): raise ValueError(f'scene {rec["scene"]}: a checkpoint has two snapshots')
        for t, sid in ids.items():
            if not sid.startswith(f'{rec["scene"]}:{t}:'): raise ValueError(f'snapshot id {sid} does not belong to scene {rec["scene"]} checkpoint {t}')
            snap_of[(rec['scene'], t)] = sid
        for b in rec['blocks']:
            k = (b['scene'], b['checkpoint'], b['block_repeat_id'])
            if b['scene'] != rec['scene'] or k in js: raise ValueError(f'block {k}: foreign or duplicate block in the scene record')
            if b['snapshot_id'] != ids.get(b['checkpoint']): raise ValueError(f'block {k}: tied to another snapshot')
            cells = {c['arm']: c for c in b['cells']}
            if tuple(c['arm'] for c in b['cells']) != ARMS: raise ValueError(f'block {k}: arms are not C0, C1, F, R once each')
            for a in ARMS:
                want = b['block_repeat_id'] + 4 if a == 'C1' else b['block_repeat_id']
                if cells[a]['experimental_arm'] != ARM_NAMES[a] or cells[a]['policy_noise_repeat_id'] != want or cells[a]['block_repeat_id'] != b['block_repeat_id']:
                    raise ValueError(f'block {k}: arm {a} has another name or noise key than declared')
                if bit(cells[a]['y']) != bit(b['y_' + a]): raise ValueError(f'block {k}: cell outcome differs from the block outcome')
            js[k] = (rec, b, cells)
    out = []; seen = set()
    for row in csv_rows:
        k = (int(row['scene']), int(row['checkpoint']), int(row['block_repeat_id']))
        if k in seen: raise ValueError(f'block {k}: twice in the CSV')
        seen.add(k)
        if k not in js: raise ValueError(f'block {k}: in the CSV but not in the scene records')
        rec, b, cells = js[k]
        if row['snapshot_id'] != b['snapshot_id']: raise ValueError(f'block {k}: snapshot id differs between CSV and JSON')
        y = {a: bit(row['y_' + a]) for a in ARMS}
        if any(y[a] != bit(b['y_' + a]) for a in ARMS): raise ValueError(f'block {k}: outcomes differ between CSV and JSON')
        if int(row['policy_noise_repeat_id_C1']) != k[2] + 4: raise ValueError(f'block {k}: C1 noise key is not r + 4')
        if bit(row['trunk_success']) != bit(rec['trunk']['success']): raise ValueError(f'block {k}: trunk outcome differs between CSV and JSON')
        out.append({'scene': k[0], 'checkpoint': k[1], 'repeat': k[2], 'snapshot_id': b['snapshot_id'], 'y': y, 'trunk_success': bit(rec['trunk']['success']),
                    'grasped_at_checkpoint_csv': boolean(row['grasped_at_checkpoint']), 'cells': cells, 'parity_pass': boolean(b['c0_c1_queue_prefix_parity']['pass'])})
    if seen != set(js): raise ValueError(f'{len(set(js) - seen)} blocks of the scene records are missing in the CSV')
    by = {}
    for x in out: by.setdefault((x['scene'], x['checkpoint']), []).append(x['repeat'])
    for k, reps in by.items():
        if sorted(reps) != list(REPEATS): raise ValueError(f'snapshot {k}: repeats are {sorted(reps)}, not 0,1,2,3')
        if k not in snap_of: raise ValueError(f'snapshot {k}: blocks without a snapshot record')
    if set(by) != set(snap_of): raise ValueError('a snapshot record has no blocks')
    return sorted(out, key=lambda x: (x['scene'], x['checkpoint'], x['repeat']))


def pair_counts(blocks, a, ref):
    n = {'0to1': 0, '1to0': 0, '1to1': 0, '0to0': 0}; s01 = set(); s10 = set()
    for x in blocks:
        c = f"{x['y'][ref]}to{x['y'][a]}"; n[c] += 1
        if c == '0to1': s01.add(x['scene'])
        if c == '1to0': s10.add(x['scene'])
    return {'arm': a, 'reference': ref, 'blocks': len(blocks), **n, 'reference_failure_blocks': n['0to1'] + n['0to0'], 'reference_success_blocks': n['1to0'] + n['1to1'],
            'delta_mean_outcome': (n['0to1'] - n['1to0']) / len(blocks) if blocks else 'NA', 'scenes_with_0to1': len(s01), 'scenes_with_1to0': len(s10)}


def scene_equal(values_by_scene):
    """Mean within each scene over its own rows first (scenes have different numbers of checkpoints), then over scenes."""
    per = {s: float(np.mean(v)) for s, v in values_by_scene.items() if len(v)}
    return (float(np.mean(list(per.values()))) if per else 'NA'), per


def by_scene(blocks, f):
    out = {}
    for x in blocks: out.setdefault(x['scene'], []).append(f(x))
    return out


def select(blocks, **want):
    return [x for x in blocks if all(x['y'][a] == v for a, v in want.items())]


def set_info(rows):
    return {'blocks': len(rows), 'snapshots': len({x['snapshot_id'] for x in rows}), 'scenes': len({x['scene'] for x in rows}),
            'scene_list': sorted({x['scene'] for x in rows})}


def r_sets(blocks):
    return {'C0=1,R=0': select(blocks, C0=1, R=0), 'C0=0,R=1': select(blocks, C0=0, R=1), 'C0=1,F=1,R=0': select(blocks, C0=1, F=1, R=0),
            'C0=0,F=0,R=1': select(blocks, C0=0, F=0, R=1), 'C0=C1=F=0,R=1': select(blocks, C0=0, C1=0, F=0, R=1),
            'C0=1,R=1': select(blocks, C0=1, R=1), 'C0=0,R=0': select(blocks, C0=0, R=0)}


def recount(blocks, records, old_summary=None, old_pairs=None, old_skips=None):
    scenes = sorted({x['scene'] for x in blocks}); out = {'scenes_recorded': len(records), 'scenes_with_blocks': len(scenes),
                                                         'snapshots': len({x['snapshot_id'] for x in blocks}), 'blocks': len(blocks), 'continuations': len(ARMS) * len(blocks)}
    out['success_blocks'] = {a: sum(x['y'][a] for x in blocks) for a in ARMS}
    out['pairs'] = []
    for a, ref in PAIRS:
        p = pair_counts(blocks, a, ref); mean, per = scene_equal(by_scene(blocks, lambda x: x['y'][a] - x['y'][ref]))
        p['scene_equal_delta'] = mean; p['scenes_positive'] = sum(v > 0 for v in per.values()); p['scenes_negative'] = sum(v < 0 for v in per.values())
        p['scenes_zero'] = sum(v == 0 for v in per.values()); out['pairs'].append(p)
    out['scene_equal_mean_outcome'] = {a: scene_equal(by_scene(blocks, lambda x: x['y'][a]))[0] for a in ARMS}
    sets = r_sets(blocks); out['r_sets'] = {k: set_info(v) for k, v in sets.items()}
    key = lambda x: (x['scene'], x['checkpoint'], x['repeat'])
    out['subset_checks'] = {'C0=1,F=1,R=0 within C0=1,R=0': {key(x) for x in sets['C0=1,F=1,R=0']} <= {key(x) for x in sets['C0=1,R=0']},
                            'C0=0,F=0,R=1 within C0=0,R=1': {key(x) for x in sets['C0=0,F=0,R=1']} <= {key(x) for x in sets['C0=0,R=1']},
                            'C0=C1=F=0,R=1 within C0=0,F=0,R=1': {key(x) for x in sets['C0=C1=F=0,R=1']} <= {key(x) for x in sets['C0=0,F=0,R=1']}}
    trunk_ok = {r['scene'] for r in records if bit(r['trunk']['success'])}; cross = {}
    for name in ('C0=1,R=0', 'C0=1,F=1,R=0', 'C0=0,R=1'):
        inside = [x for x in sets[name] if x['scene'] in trunk_ok]; outside = [x for x in sets[name] if x['scene'] not in trunk_ok]
        cross[name] = {'in_trunk_success_scenes': set_info(inside), 'in_trunk_failure_scenes': set_info(outside)}
    all_ok = [x for x in blocks if x['scene'] in trunk_ok]
    cross['blocks_of_trunk_success_scenes'] = {**set_info(all_ok), 'of_which_C0=1,F=1,R=0': len(select(all_ok, C0=1, F=1, R=0)),
                                               'of_which_C0=0': len(select(all_ok, C0=0)), 'of_which_R=1': len(select(all_ok, R=1))}
    cross['note'] = 'trunk_success is the outcome of the never-restored repeat-0 run of a scene; it does not stand for the C0 outcome of other repeats'
    out['trunk_success_cross'] = cross; out['trunk_success_scenes'] = sorted(trunk_ok)
    skips = [(r['scene'], s['checkpoint'], s['reason']) for r in records for s in r['skips']]; reasons = {}
    for _, _, why in skips: reasons[why] = reasons.get(why, 0) + 1
    out['skipped_checkpoints'] = {'n': len(skips), 'reasons': reasons, 'by_checkpoint': {str(t): sum(1 for s in skips if s[1] == t) for t in sorted({s[1] for s in skips})}}
    dup = [d for r in records for d in r['no_intervention_duplicates']]; rc = [d for r in records for d in r['rechecks']]
    out['reproducibility_records'] = {'trunk_duplicates': len(dup), 'trunk_duplicates_matched': sum(boolean(d['match']) for d in dup),
                                      'reruns': len(rc), 'reruns_matched': sum(boolean(d['match']) for d in rc),
                                      'reruns_by_arm': {a: sum(1 for d in rc if d['experimental_arm'] == a) for a in ('continue', 'continue_alt_noise')},
                                      'queue_parity_blocks': len(blocks), 'queue_parity_passed': sum(x['parity_pass'] for x in blocks)}
    checks = {}
    for k in ('snapshots', 'blocks', 'continuations'): checks['expected_' + k] = out[k] == EXPECTED[k]
    checks['expected_scenes'] = out['scenes_with_blocks'] == EXPECTED['scenes']; checks['expected_success'] = out['success_blocks'] == EXPECTED['success']
    for k, (nb, ns) in EXPECTED['sets'].items(): checks['expected_set_' + k] = (out['r_sets'][k]['blocks'], out['r_sets'][k]['scenes']) == (nb, ns)
    if old_summary is not None:
        checks['summary_success_blocks'] = old_summary['success_blocks_by_arm'] == out['success_blocks']
        checks['summary_counts'] = (old_summary['snapshots'], old_summary['blocks'], old_summary['continuations']) == (out['snapshots'], out['blocks'], out['continuations'])
        se = old_summary['scene_equal_weight']
        checks['summary_scene_equal_deltas'] = all(abs(se[f"d_{p['arm']}_{p['reference']}"]['mean_over_scenes'] - p['scene_equal_delta']) < 1e-12 for p in out['pairs'])
        checks['summary_skips'] = old_summary['skipped_checkpoints'] == reasons
    if old_pairs is not None:
        old = {(r['arm'], r['reference']): r for r in old_pairs if r['checkpoint'] == 'all'}
        checks['pair_summary_csv'] = all(all(int(old[(p['arm'], p['reference'])][c]) == p[c] for c in ('0to1', '1to0', '1to1', '0to0', 'blocks')) for p in out['pairs'])
    if old_skips is not None:
        checks['skip_reasons_csv'] = sorted((int(r['scene']), int(r['checkpoint']), r['reason']) for r in old_skips) == sorted(skips)
    out['checks'] = checks; out['all_checks_pass'] = all(checks.values()) and all(out['subset_checks'].values())
    return out


# ---------------------------------------------------------------- B: per-snapshot counts
def snapshots(blocks):
    out = {}
    for x in blocks: out.setdefault((x['scene'], x['checkpoint'], x['snapshot_id']), {})[x['repeat']] = x
    return out


def snapshot_rows(blocks):
    rows = []
    for (scene, t, sid), reps in sorted(snapshots(blocks).items()):
        k = {a: sum(reps[r]['y'][a] for r in reps) for a in ARMS}
        row = {'scene': scene, 'checkpoint': t, 'snapshot_id': sid, 'valid_repeats': len(reps), **{f'k_{a}': k[a] for a in ARMS},
               'trunk_success': reps[0]['trunk_success']}
        for a, ref in PAIRS:
            if (a, ref) != ('C1', 'C0'): row[f'mean_{a}_minus_{ref}'] = (k[a] - k[ref]) / len(reps)
        row['mean_C1_minus_C0'] = (k['C1'] - k['C0']) / len(reps)
        for name, want in (('R_only_vs_C0_F', dict(C0=0, F=0, R=1)), ('R_fail_vs_C0_F', dict(C0=1, F=1, R=0)), ('C0fail_Rsuccess', dict(C0=0, R=1)),
                           ('C0success_Rfail', dict(C0=1, R=0)), ('C0fail_Fsuccess', dict(C0=0, F=1)), ('C0success_Ffail', dict(C0=1, F=0)),
                           ('C0fail_C1success', dict(C0=0, C1=1)), ('C0success_C1fail', dict(C0=1, C1=0))):
            ids = [r for r in sorted(reps) if all(reps[r]['y'][a] == v for a, v in want.items())]
            row[f'n_{name}'] = len(ids); row[f'repeats_{name}'] = ' '.join(map(str, ids))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- C: leave-one-repeat-out choice diagnostic
def first_max(score, priority=PRIORITY):
    """Largest training count; ties go to C0, then F, then R. Only arms present in `score` can be chosen."""
    best = max(score.values())
    for a in priority:
        if a in score and score[a] == best: return a
    raise ValueError('no candidate arm')


def transfer_rows(blocks):
    """For every snapshot and held-out repeat: choose from the OTHER three repeats, read the choice on the held-out one.
    Uses past branch outcomes of the same state, so it cannot be deployed; it is a consistency diagnostic only."""
    rows = []
    for (scene, t, sid), reps in sorted(snapshots(blocks).items()):
        for held in REPEATS:
            train = [q for q in REPEATS if q != held]
            full = {a: sum(reps[q]['y'][a] for q in train) for a in PRIORITY}
            for rule, allowed in RULES.items():
                chosen = first_max({a: full[a] for a in allowed}); y = reps[held]['y']
                rows.append({'rule': rule, 'scene': scene, 'checkpoint': t, 'snapshot_id': sid, 'held_out_repeat': held, 'training_repeats': ' '.join(map(str, train)),
                             **{f'training_successes_{a}': full[a] for a in PRIORITY}, 'chosen_arm': chosen, 'y_chosen': y[chosen],
                             **{f'y_{a}_held_out': y[a] for a in ARMS}, 'chosen_vs_C0': f"{y['C0']}to{y[chosen]}", 'chosen_vs_C1': f"{y['C1']}to{y[chosen]}"})
    return rows


def transfer_summary(rows):
    """Scene-equal first (mean over a scene's own snapshot x held-out rows, then over scenes); block mean second."""
    scenes = sorted({r['scene'] for r in rows}); base = [r for r in rows if r['rule'] == 'G_CF']; series = {}
    for a in ALWAYS: series['always_' + a] = {(r['scene'], r['checkpoint'], r['held_out_repeat']): r[f'y_{a}_held_out'] for r in base}
    for rule in RULES: series[rule] = {(r['scene'], r['checkpoint'], r['held_out_repeat']): r['y_chosen'] for r in rows if r['rule'] == rule}
    keys = sorted(series['always_C0'])

    def per_scene(f):
        out = {s: [] for s in scenes}
        for k in keys: out[k[0]].append(f(k))
        return out
    per = {name: {s: float(np.mean(v)) for s, v in per_scene(lambda k: series[name][k]).items()} for name in series}
    scene_table = [{'scene': s, 'rows_per_rule': sum(1 for k in keys if k[0] == s), **{name: per[name][s] for name in series},
                    **{f'{rule}_minus_always_C0': per[rule][s] - per['always_C0'][s] for rule in RULES},
                    **{f'{rule}_minus_always_C1': per[rule][s] - per['always_C1'][s] for rule in RULES},
                    'G_CFR_minus_G_CF': per['G_CFR'][s] - per['G_CF'][s]} for s in scenes]
    summary = {}
    for name in series:
        vals = [series[name][k] for k in keys]; d0 = [per[name][s] - per['always_C0'][s] for s in scenes]; d1 = [per[name][s] - per['always_C1'][s] for s in scenes]
        summary[name] = {'rows': len(vals), 'successes': int(sum(vals)), 'block_mean': float(np.mean(vals)), 'scene_equal_mean': float(np.mean([per[name][s] for s in scenes])),
                         'scene_equal_minus_always_C0': float(np.mean(d0)), 'scenes_above_C0': sum(x > 0 for x in d0), 'scenes_below_C0': sum(x < 0 for x in d0),
                         'scenes_equal_C0': sum(x == 0 for x in d0),
                         'scene_equal_minus_always_C1': float(np.mean(d1)), 'scenes_above_C1': sum(x > 0 for x in d1), 'scenes_below_C1': sum(x < 0 for x in d1),
                         'scenes_equal_C1': sum(x == 0 for x in d1)}
    choices = {}
    for rule in RULES:
        mine = [r for r in rows if r['rule'] == rule]; c = {}
        for a in RULES[rule]:
            sel = [r for r in mine if r['chosen_arm'] == a]
            c[a] = {'rows': len(sel), 'snapshots': len({r['snapshot_id'] for r in sel}), 'scenes': len({r['scene'] for r in sel}),
                    'held_out_vs_C0': {k: sum(1 for r in sel if r['chosen_vs_C0'] == k) for k in ('0to1', '1to0', '1to1', '0to0')},
                    'held_out_vs_C1': {k: sum(1 for r in sel if r['chosen_vs_C1'] == k) for k in ('0to1', '1to0', '1to1', '0to0')},
                    'scenes_with_0to1_vs_C0': sorted({r['scene'] for r in sel if r['chosen_vs_C0'] == '0to1'}),
                    'scenes_with_1to0_vs_C0': sorted({r['scene'] for r in sel if r['chosen_vs_C0'] == '1to0'})}
        choices[rule] = c
    cf = {(r['scene'], r['checkpoint'], r['held_out_repeat']): r for r in rows if r['rule'] == 'G_CF'}; diff = []
    for r in rows:
        if r['rule'] != 'G_CFR': continue
        o = cf[(r['scene'], r['checkpoint'], r['held_out_repeat'])]
        if r['chosen_arm'] != o['chosen_arm']:
            diff.append({'scene': r['scene'], 'checkpoint': r['checkpoint'], 'snapshot_id': r['snapshot_id'], 'held_out_repeat': r['held_out_repeat'],
                         'G_CF_chose': o['chosen_arm'], 'G_CFR_chose': r['chosen_arm'], 'y_G_CF': o['y_chosen'], 'y_G_CFR': r['y_chosen'],
                         'difference': r['y_chosen'] - o['y_chosen']})
    gap = {'rows_where_the_choice_differs': len(diff), 'sum_of_differences': sum(d['difference'] for d in diff),
           'rows_better': sum(d['difference'] > 0 for d in diff), 'rows_worse': sum(d['difference'] < 0 for d in diff), 'rows_same_outcome': sum(d['difference'] == 0 for d in diff),
           'scenes': sorted({d['scene'] for d in diff}), 'snapshots': sorted({d['snapshot_id'] for d in diff}), 'rows': diff}
    return {'policies': summary, 'choices': choices, 'G_CFR_minus_G_CF': gap, 'dependence': 'Rows of one scene share the scene and its past; every repeat of a snapshot is '
            'used three times for choosing and once for reading. The row count is not a number of independent samples.'}, scene_table


# ---------------------------------------------------------------- D: events around the retreat
def derived(rows, z0):
    r = np.asarray(rows, float).reshape(-1, len(COL)); A = r[:, 4:7]; B = r[:, 7:10]; T = r[:, 1:4]
    lift = A[:, 2] - z0; d = np.linalg.norm(T - A, axis=1); xy = np.linalg.norm(A[:, :2] - B[:, :2], axis=1); lifted = lift >= LIFT_M
    return {'t': r[:, 0].astype(int), 'grasp': r[:, COL['is_cubeA_grasped']] > .5, 'on': r[:, COL['is_cubeA_on_cubeB']] > .5, 'success': r[:, COL['success']] > .5,
            'lift': lift, 'd': d, 'xy': xy, 'lifted': lifted, 'aligned': lifted & (xy <= ALIGN_M), 'near': d <= NEAR_M, 'A': A, 'B': B, 'grip': r[:, COL['a_gripper']]}


def first_step(t, mask):
    i = np.flatnonzero(mask)
    return int(t[i[0]]) if len(i) else None


def retreat_events(trunk, branch, checkpoint, steps_used, z0, y, steps, reason, first_inference_step, horizon=200):
    """Observed facts of one Retreat-Retry branch. Labels are observations; no cause is attached to any of them."""
    pre = derived(trunk[:checkpoint], z0); br = derived(branch, z0); k = int(steps_used); n = len(br['t'])
    bad = (len(pre['t']) != checkpoint or checkpoint < 1 or n != steps - checkpoint or k > n or (n and (br['t'][0] != checkpoint + 1 or br['t'][-1] != steps))
           or (first_inference_step is not None and first_inference_step != checkpoint + k))
    if bad: return {'insufficient_trace': True}
    at = lambda d, i, name: d[name][i]
    cp = {name: at(pre, -1, name) for name in ('grasp', 'lift', 'd', 'xy', 'near', 'lifted', 'aligned', 'on', 'grip')}; A0 = pre['A'][-1]; B0 = pre['B'][-1]
    src, i = (br, k - 1) if k else (pre, -1)
    ret = {name: at(src, i, name) for name in ('grasp', 'lift', 'd', 'xy', 'near', 'lifted', 'aligned', 'on')}; A1 = src['A'][i]; B1 = src['B'][i]
    during = {name: br[name][:k] for name in br}; after = {name: br[name][k:] for name in br}
    seq = np.concatenate([[bool(ret['grasp'])], after['grasp']]); drop = np.flatnonzero(seq[:-1] & ~seq[1:])      # True -> False at after[index]
    loss_after = int(after['t'][drop[0]]) if len(drop) else None
    loss_on = bool(after['on'][drop[0]] or (after['on'][drop[0] - 1] if drop[0] > 0 else ret['on'])) if len(drop) else False
    b_during = float(np.linalg.norm(during['B'] - B0, axis=1).max()) if k else 0.
    b_after = float(np.linalg.norm(after['B'] - B1, axis=1).max()) if len(after['t']) else 0.
    fin = {name: (br[name][-1] if n else cp[name]) for name in ('grasp', 'lift', 'd', 'xy')}
    out = {'insufficient_trace': False, 'retreat_steps_used': k, 'return_step': checkpoint + k, 'remaining_steps_at_return': horizon - (checkpoint + k),
           'cp_grasped': bool(cp['grasp']), 'cp_gripper_command': float(cp['grip']), 'cp_cube_lift_m': float(cp['lift']), 'cp_tcp_to_cube_m': float(cp['d']),
           'cp_cubeA_to_cubeB_xy_m': float(cp['xy']), 'cp_near': bool(cp['near']), 'cp_lifted': bool(cp['lifted']), 'cp_aligned': bool(cp['aligned']), 'cp_on_cubeB': bool(cp['on']),
           'ret_grasped': bool(ret['grasp']), 'ret_cube_lift_m': float(ret['lift']), 'ret_tcp_to_cube_m': float(ret['d']), 'ret_cubeA_to_cubeB_xy_m': float(ret['xy']),
           'ret_near': bool(ret['near']), 'ret_lifted': bool(ret['lifted']), 'ret_aligned': bool(ret['aligned']),
           'cubeA_displacement_during_retreat_m': float(np.linalg.norm(A1 - A0)), 'cubeA_dz_during_retreat_m': float(A1[2] - A0[2]),
           'cubeB_max_displacement_during_retreat_m': b_during, 'cubeB_max_displacement_after_return_m': b_after,
           'first_grasp_loss_step_during_retreat': first_step(during['t'], ~during['grasp']) if cp['grasp'] else None,
           'first_grasp_step_after_return': first_step(after['t'], after['grasp']), 'first_grasp_loss_step_after_return': loss_after,
           'first_near_step_after_return': first_step(after['t'], after['near']), 'first_lifted_step_after_return': first_step(after['t'], after['lifted']),
           'first_aligned_step_after_return': first_step(after['t'], after['aligned']), 'first_on_cubeB_step_after_return': first_step(after['t'], after['on']),
           'grasp_steps_after_return': int(after['grasp'].sum()), 'end_step': int(steps), 'end_reason': reason,
           'steps_from_return_to_end': int(steps - checkpoint - k), 'final_grasped': bool(fin['grasp']), 'final_cube_lift_m': float(fin['lift']),
           'final_tcp_to_cube_m': float(fin['d']), 'final_cubeA_to_cubeB_xy_m': float(fin['xy'])}
    out.update(grasp_loss_during_retreat=bool(cp['grasp'] and (~during['grasp']).any()), grasp_kept_through_retreat=bool(cp['grasp'] and during['grasp'].all()),
               grasp_loss_after_return=loss_after is not None, grasp_loss_after_return_while_on_cubeB=loss_on,
               lift_lost_during_retreat=bool(cp['lifted'] and not ret['lifted']), alignment_lost_during_retreat=bool(cp['aligned'] and not ret['aligned']),
               near_lost_during_retreat=bool(cp['near'] and not ret['near']),
               no_reapproach_before_horizon=bool(not y and not ret['near'] and not after['near'].any()),
               never_grasped_after_return=bool(not y and not ret['grasp'] and not after['grasp'].any()),
               retained_grasp_no_success_by_horizon=bool(not y and reason == 'horizon' and fin['grasp']),
               aligned_after_return_no_success=bool(not y and after['aligned'].any()),
               success_after_return=bool(y and steps > checkpoint + k), ended_during_retreat=bool(n == k and (k > 0 or n == 0)),
               cubeB_moved_during_retreat=b_during > B_MOVED_M, cubeB_moved_after_return=b_after > B_MOVED_M, ended_at_horizon=reason == 'horizon')
    return out


def reference_at(branch, checkpoint, step, z0, cp_state):
    """State of another arm at the same PHYSICAL step (its last state if it ended earlier; the checkpoint state if step == checkpoint)."""
    d = derived(branch, z0); i = min(step - checkpoint, len(d['t'])) - 1
    if i < 0: return dict(cp_state, ended_before=False)
    return {'grasped': bool(d['grasp'][i]), 'lifted': bool(d['lifted'][i]), 'aligned': bool(d['aligned'][i]), 'near': bool(d['near'][i]),
            'ended_before': bool(step - checkpoint > len(d['t']))}


def event_rows(blocks, records, trace_dir):
    recs = {r['scene']: r for r in records}; rows = []
    for scene in sorted(recs):
        mine = [x for x in blocks if x['scene'] == scene]
        if not mine: continue
        path = Path(trace_dir) / f'{scene}.npz'; z0 = float(recs[scene]['initial_state']['cubeA'][2])
        with np.load(path) as z:
            if [str(c) for c in z['columns']] != list(COL): raise ValueError(f'{path}: trace columns differ from the collector')
            trunk = z['trunk']
            for x in mine:
                t, rep = x['checkpoint'], x['repeat']; c = x['cells']; R = c['R']
                ev = retreat_events(trunk, z[f'b_{t}_{rep}_R'], t, R['retreat']['steps_used'], z0, R['y'], R['steps'], R['reason'], R['first_inference_step'])
                y = x['y']; row = {'scene': scene, 'checkpoint': t, 'repeat': rep, 'snapshot_id': x['snapshot_id'], **{f'y_{a}': y[a] for a in ARMS},
                                   'R_vs_C0': f"{y['C0']}to{y['R']}", 'in_C0=1,F=1,R=0': int((y['C0'], y['F'], y['R']) == (1, 1, 0)),
                                   'in_C0=0,F=0,R=1': int((y['C0'], y['F'], y['R']) == (0, 0, 1)), 'in_C0=C1=F=0,R=1': int((y['C0'], y['C1'], y['F'], y['R']) == (0, 0, 0, 1)),
                                   'trunk_success': x['trunk_success'], 'trace_file': path.as_posix(), 'trace_key': f'b_{t}_{rep}_R'}
                if not ev['insufficient_trace']:
                    if ev['cp_grasped'] != x['grasped_at_checkpoint_csv']: raise ValueError(f'{scene} {t} {rep}: grasp flag at the checkpoint differs between trace and CSV')
                    ref = reference_at(z[f'b_{t}_{rep}_C0'], t, ev['return_step'], z0,
                                       {'grasped': ev['cp_grasped'], 'lifted': ev['cp_lifted'], 'aligned': ev['cp_aligned'], 'near': ev['cp_near']})
                    ev.update({'C0_at_return_step_' + k: v for k, v in ref.items()})
                    ev.update(C0_end_step=c['C0']['steps'], C0_end_reason=c['C0']['reason'], C0_steps_from_checkpoint_to_end=c['C0']['steps'] - t,
                              F_end_step=c['F']['steps'], F_end_reason=c['F']['reason'])
                rows.append({**row, **ev})
    return rows


EVENT_SETS = (('R_worse_than_C0 (C0=1,R=0)', lambda r: r['R_vs_C0'] == '1to0'), ('R_better_than_C0 (C0=0,R=1)', lambda r: r['R_vs_C0'] == '0to1'),
              ('both_succeed (C0=1,R=1)', lambda r: r['R_vs_C0'] == '1to1'), ('both_fail (C0=0,R=0)', lambda r: r['R_vs_C0'] == '0to0'),
              ('C0=1,F=1,R=0', lambda r: r['in_C0=1,F=1,R=0'] == 1), ('C0=0,F=0,R=1', lambda r: r['in_C0=0,F=0,R=1'] == 1),
              ('C0=C1=F=0,R=1', lambda r: r['in_C0=C1=F=0,R=1'] == 1),
              ('C0=1,R=0 in trunk-success scenes', lambda r: r['R_vs_C0'] == '1to0' and r['trunk_success'] == 1),
              ('C0=1,R=0 in trunk-failure scenes', lambda r: r['R_vs_C0'] == '1to0' and r['trunk_success'] == 0), ('all blocks', lambda r: True))
STATE_FLAGS = ('cp_grasped', 'cp_near', 'cp_lifted', 'cp_aligned', 'ret_grasped', 'ret_near', 'ret_lifted', 'ret_aligned', 'final_grasped',
               'C0_at_return_step_grasped', 'C0_at_return_step_lifted', 'C0_at_return_step_aligned', 'C0_at_return_step_ended_before')
MEDIANS = ('retreat_steps_used', 'remaining_steps_at_return', 'steps_from_return_to_end', 'C0_steps_from_checkpoint_to_end', 'cp_tcp_to_cube_m', 'ret_tcp_to_cube_m',
           'cubeA_dz_during_retreat_m', 'cubeB_max_displacement_during_retreat_m', 'final_cubeA_to_cubeB_xy_m')


def event_summary(rows):
    out = []
    for name, f in EVENT_SETS:
        sel = [r for r in rows if f(r)]; ok = [r for r in sel if not r['insufficient_trace']]
        row = {'set': name, 'blocks': len(sel), 'snapshots': len({r['snapshot_id'] for r in sel}), 'scenes': len({r['scene'] for r in sel}), 'blocks_with_usable_trace': len(ok)}
        for k in EVENT_FLAGS: row[k] = sum(1 for r in (sel if k == 'insufficient_trace' else ok) if r[k])
        for k in EVENT_FLAGS[:-1]: row['scenes_' + k] = len({r['scene'] for r in ok if r[k]})
        for k in STATE_FLAGS: row[k] = sum(1 for r in ok if r[k])
        for k in MEDIANS: row['median_' + k] = float(np.median([r[k] for r in ok])) if ok else 'NA'
        out.append(row)
    return out


def case_index(rows, snaps):
    """Cases for explanation, picked AFTER the tables by a fixed rule: per set and observed flag, the first block in
    ascending (scene, checkpoint, repeat); plus the first snapshot with the pattern in two or more repeats."""
    out = []; ok = [r for r in rows if not r['insufficient_trace']]
    for name, f in EVENT_SETS[:2]:
        sel = sorted([r for r in ok if f(r)], key=lambda r: (r['scene'], r['checkpoint'], r['repeat']))
        for k in EVENT_FLAGS[:-1]:
            hit = [r for r in sel if r[k]]
            if hit: out.append({'scene': hit[0]['scene'], 'checkpoint': hit[0]['checkpoint'], 'repeat': hit[0]['repeat'], 'set': name, 'observation': k,
                                'selection_rule': 'first block of the set with this observed flag, ascending scene/checkpoint/repeat', 'blocks_with_this_flag_in_set': len(hit),
                                'evidence': f"{hit[0]['trace_file']}:{hit[0]['trace_key']}", 'post_hoc': True})
    for col, label in (('n_R_only_vs_C0_F', 'R alone succeeded (C0=0,F=0,R=1) in two or more repeats of one snapshot'),
                       ('n_C0success_Rfail', 'R failed where C0 succeeded (C0=1,R=0) in two or more repeats of one snapshot')):
        usable = {(x['scene'], x['checkpoint'], x['repeat']): x for x in ok}; multi = []
        for s in snaps:          # only repeats whose trace is usable can serve as the example
            reps = [int(v) for v in s['repeats_' + col[2:]].split() if (s['scene'], s['checkpoint'], int(v)) in usable]
            if s[col] >= 2 and reps: multi.append((s, reps[0]))
        if multi:
            s, rep = multi[0]; r = usable[(s['scene'], s['checkpoint'], rep)]
            out.append({'scene': s['scene'], 'checkpoint': s['checkpoint'], 'repeat': rep, 'set': label, 'observation': f"{s[col]} of 4 repeats ({s['repeats_' + col[2:]]})",
                        'selection_rule': 'first snapshot (ascending scene/checkpoint) with the pattern in >= 2 repeats; its lowest such repeat',
                        'blocks_with_this_flag_in_set': len(multi), 'evidence': f"{r['trace_file']}:{r['trace_key']}", 'post_hoc': True})
    return out


def case_timeseries(trace_dir, record, scene, checkpoint, repeat, steps_used, lead=8):
    """Step table of C0 and R for one block: same physical step in every row."""
    z0 = float(record['initial_state']['cubeA'][2]); rows = []
    with np.load(Path(trace_dir) / f'{scene}.npz') as z:
        pre = derived(z['trunk'][:checkpoint], z0); arms = {a: derived(z[f'b_{checkpoint}_{repeat}_{a}'], z0) for a in ('C0', 'R')}
    cell = lambda d, i: {'grasped': int(d['grasp'][i]), 'on_cubeB': int(d['on'][i]), 'cube_lift_m': round(float(d['lift'][i]), 4), 'tcp_to_cube_m': round(float(d['d'][i]), 4),
                         'cubeA_to_cubeB_xy_m': round(float(d['xy'][i]), 4), 'gripper_command': round(float(d['grip'][i]), 3)}
    for i in range(max(0, checkpoint - lead), checkpoint):
        c = cell(pre, i); rows.append({'step': i + 1, 'phase_R': 'before_checkpoint', **{f'C0_{k}': v for k, v in c.items()}, **{f'R_{k}': v for k, v in c.items()}})
    n = max(len(arms['C0']['t']), len(arms['R']['t']))
    for i in range(n):
        row = {'step': checkpoint + i + 1, 'phase_R': 'retreat' if i < steps_used else ('policy_after_return' if i < len(arms['R']['t']) else 'ended')}
        for a in ('C0', 'R'):
            c = cell(arms[a], i) if i < len(arms[a]['t']) else {k: '' for k in ('grasped', 'on_cubeB', 'cube_lift_m', 'tcp_to_cube_m', 'cubeA_to_cubeB_xy_m', 'gripper_command')}
            row.update({f'{a}_{k}': v for k, v in c.items()})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- commands
def input_files(a):
    res = Path(a.results); run = Path(a.run); files = sorted(p for p in res.rglob('*') if p.is_file())
    files += sorted((run / 'branches' / 'traces').glob('*.npz')) + sorted((run / 'branches' / 'inputs').glob('*.npz')) + sorted((run / 'branches' / 'scenes').glob('*.json'))
    files += [p for p in (Path(a.backup),) if p.exists()]
    return files


def hashes(files):
    return {p.as_posix(): sha256(p) for p in files}


def backup_check(a, h):
    """Raw files of the run against the per-file list computed on the server from the archive."""
    listed = {}
    for line in (Path(a.results) / 'backup_sha256_stage2b_001.txt').read_text().splitlines():
        v, name = line.split(None, 1); listed[name.strip().lstrip('*')] = v
    run = Path(a.run).as_posix(); mine = {k: v for k, v in h.items() if k.startswith(run + '/')}
    bad = [k for k, v in mine.items() if listed.get(k) != v]
    copies = [p.name for p in (Path(a.results) / 'scenes').glob('*.json') if h[p.as_posix()] != h.get(f'{run}/branches/scenes/{p.name}')]
    return {'raw_files_checked': len(mine), 'raw_files_differing_from_backup_list': bad, 'results_scene_copies_differing_from_raw': copies,
            'pass': bool(mine) and not bad and not copies}


def inputs_schema(a, records):
    """Presence and layout of the saved online inputs only. They are not read into any model here."""
    fields = read_json(Path(a.results) / 'observation_fields.json'); want = set(fields['online_inputs']); seen = {}; problems = []
    for rec in records:
        p = Path(a.run) / 'branches' / 'inputs' / f"{rec['scene']}.npz"
        if not p.exists(): problems.append(f'{p}: missing'); continue
        with np.load(p) as z:
            names = set(z.files)
            for s in rec['snapshots']:
                pre = f"t{s['checkpoint']}_"; got = {n[len(pre):] for n in names if n.startswith(pre)}
                if got != want: problems.append(f"{p} checkpoint {s['checkpoint']}: fields {sorted(got)}")
                for n in got: seen.setdefault(n, set()).add((tuple(z[pre + n].shape), str(z[pre + n].dtype)))
            if len(names) != len(want) * len(rec['snapshots']): problems.append(f'{p}: arrays that belong to no snapshot')
    return {'files': len(records), 'snapshot_sets': sum(len(r['snapshots']) for r in records), 'fields': {k: sorted(map(str, v)) for k, v in sorted(seen.items())},
            'cameras': fields.get('cameras'), 'problems': problems, 'pass': not problems, 'use_in_this_stage': 'existence and layout only; no model was trained or evaluated'}


def outside_inputs(a):
    """Outputs never go into a location that is read as input."""
    out = Path(a.out).resolve()
    for p in (Path(a.results), Path(a.run), Path(a.backup).parent):
        p = p.resolve()
        if out == p or p in out.parents: raise SystemExit(f'Refused: --out {out} lies inside the input location {p}')


def cmd_manifest(a):
    outside_inputs(a); out = Path(a.out); out.mkdir(parents=True, exist_ok=False); files = input_files(a); h = hashes(files); plan = read_json(Path(a.results) / 'plan.json')
    need = [Path(a.results) / n for n in ('pair_outcomes_four_arm.csv', 'scene_equal_weight_summary.csv', 'refresh_retreat_complementarity.csv', 'summary.json', 'plan.json',
                                         'observation_fields.json', 'backup_sha256_stage2b_001.txt')]
    missing = [p.as_posix() for p in need if not p.exists()]; traces = len(list((Path(a.run) / 'branches' / 'traces').glob('*.npz')))
    m = {'run_id': out.name, 'stage': 'Stage 2C offline review of saved Stage 2B records', 'created_utc': datetime.now(timezone.utc).isoformat(),
         'nature': 'Post hoc development analysis, written after the Stage 2B aggregates and several individual cases were already seen and published '
                   '(report stage2b_001, README). Not a confirmatory experiment and not an independent validation.',
         'resources': {'gpu': 'none', 'paid': 0, 'new_scenes': 0, 'simulator_rollouts': 0, 'policy_inference': 0, 'training': 0, 'rendering': 0},
         'status': {'base_policy_qualification': 'BELOW_ORIGINAL_GATE', 'stage2a': 'COMPLETED_EXPLORATORY_ONLY', 'stage2b': 'COMPLETED_EXPLORATORY_ONLY',
                    'observation_based_selector': 'NOT_TRAINED', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'},
         'data_role': {'main': 'Stage 2B only (stage2b_exploratory_dev, 24 scenes, seeds 10024-10047)', 'stage2a': 'context only, reported separately, never pooled',
                       'saved_online_inputs': 'existence and layout checked; not used for any model'},
         'arms': {**ARM_NAMES, 'note': 'C1 is an experimental control and not a deployable intervention candidate; it is never chosen by any rule'},
         'stage2b_code_at_run_time': {'plan_sha256': sha256(Path(a.results) / 'plan.json'), 'code': plan['code'], 'repository_head_then': plan['repository']['head'],
                                      'note': 'Records are read with the schema of this code, not with the current HEAD'},
         'analysis_code': {'scripts/' + Path(__file__).name: sha256(__file__)},
         'inputs': {'files': len(files), 'required_missing': missing, 'trace_files': traces, 'sha256': h},
         'rules': {'join_key': '(scene, checkpoint, block_repeat_id); duplicates, gaps, conflicts and a block tied to another snapshot are errors',
                   'outcomes': 'strictly 0 or 1; booleans strictly True or False',
                   'pairs': [f'{x} vs {y}' for x, y in PAIRS], 'primary_weighting': 'mean within a scene over its reached checkpoints and repeats, then over the 24 scenes',
                   'repeat_transfer': {'candidate_sets': {k: list(v) for k, v in RULES.items()}, 'choice': 'largest number of successes over the three other repeats',
                                       'ties': 'C0 first, then F, then R', 'held_out': 'every repeat 0..3 of every snapshot; the held-out outcome is never used for the choice',
                                       'C1': 'reported on the same held-out repeat as an additional reference only',
                                       'meaning': 'consistency of the arm ranking across noise repeats of the SAME state. Uses past branch outcomes, so it cannot be deployed. '
                                                  'Not generalisation to new scenes, not an observation-based selector, not a policy evaluation, not a causal test.'},
                   'events': {'time_axis': 'trunk row i = state after step i+1; branch row i = state after step checkpoint+i+1; checkpoint state = trunk row checkpoint-1',
                              'retreat_end': 'retreat.steps_used of the block, cross-checked with the first inference step of the R arm; never a fixed number',
                              'thresholds': {'NEAR_M': NEAR_M, 'ALIGN_M': ALIGN_M, 'LIFT_M': LIFT_M, 'source': 'src/hci/diagnostics.py at the Stage 2B code hash (existing labels)',
                                             'B_MOVED_M': B_MOVED_M, 'B_MOVED_M_note': 'set here before the run; describes displacement only'},
                              'flags': list(EVENT_FLAGS), 'applied_to': 'all blocks; sets are formed from outcomes afterwards',
                              'not_claimed': NOT_CLAIMED},
                   'cases': 'picked after the tables by a fixed rule (first ascending block per set and flag; first snapshot with a pattern in >= 2 repeats)',
                   'excluded': ['p-values', 'confidence intervals', 'LTT', 'risk guarantees', 'threshold search on these data', 'best-of-arms presented as executable']},
         'expected_for_recount_only': EXPECTED,
         'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}}
    m['pass'] = not missing; write_json(out / 'analysis_manifest.json', m)
    print(json.dumps({'pass': m['pass'], 'inputs': len(files), 'trace_files': traces, 'missing': missing}))
    if missing: raise SystemExit(2)


def cmd_verify(a):
    outside_inputs(a); m = read_json(Path(a.out) / 'analysis_manifest.json'); now = hashes(input_files(a)); old = m['inputs']['sha256']
    changed = sorted(k for k in old if now.get(k) != old[k]); new = sorted(set(now) - set(old))
    rep = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'files': len(old), 'changed_or_missing': changed, 'new_files_in_input_locations': new,
           'unchanged': not changed and not new}
    name = Path(a.out) / (a.label + '.json'); write_json(name, rep); print(json.dumps({k: rep[k] for k in ('files', 'changed_or_missing', 'new_files_in_input_locations', 'unchanged')}))
    if not rep['unchanged']: raise SystemExit(2)
    return rep


def cmd_run(a):
    outside_inputs(a); out = Path(a.out); m = read_json(out / 'analysis_manifest.json')
    if list(m['analysis_code'].values()) != [sha256(__file__)]: raise SystemExit('Refused: this file differs from the manifest (write a new manifest under a new run id)')
    now = hashes(input_files(a))
    if now != m['inputs']['sha256']: raise SystemExit('Refused: inputs differ from the manifest')
    res = Path(a.results); run = Path(a.run); trace_dir = run / 'branches' / 'traces'
    records = [read_json(p) for p in sorted((res / 'scenes').glob('*.json'))]
    blocks = join_blocks(read_csv(res / 'pair_outcomes_four_arm.csv'), records)
    rc = recount(blocks, records, read_json(res / 'summary.json'), read_csv(res / 'pair_summary.csv'), read_csv(res / 'skip_reasons.csv'))
    rc['backup_list'] = backup_check(a, now); rc['saved_online_inputs'] = inputs_schema(a, records)
    rc['raw_traces'] = {'files': len(list(trace_dir.glob('*.npz'))), 'scenes_recorded': len(records)}
    rc['r_sets'] = {k: v for k, v in rc['r_sets'].items()}
    write_json(out / 'integrity_and_recount.json', rc)
    print(json.dumps({'all_checks_pass': rc['all_checks_pass'], 'checks_failed': [k for k, v in rc['checks'].items() if not v], 'backup': rc['backup_list']['pass']}))
    if not rc['all_checks_pass'] or not rc['backup_list']['pass']: raise SystemExit('Stopped: the recount or the raw-file check did not reproduce the Stage 2B records')
    snaps = snapshot_rows(blocks); write_csv(out / 'snapshot_outcome_counts.csv', snaps)
    tr = transfer_rows(blocks); write_csv(out / 'repeat_transfer_rows.csv', tr); summary, scene_table = transfer_summary(tr)
    total = {'scene': 'ALL_scene_equal_weight', 'rows_per_rule': sum(s['rows_per_rule'] for s in scene_table),
             **{k: float(np.mean([s[k] for s in scene_table])) for k in scene_table[0] if k not in ('scene', 'rows_per_rule')}}
    block = {'scene': 'ALL_row_mean', 'rows_per_rule': total['rows_per_rule'], **{k: summary['policies'][k]['block_mean'] for k in summary['policies']}}
    for rule in RULES:
        block[f'{rule}_minus_always_C0'] = block[rule] - block['always_C0']; block[f'{rule}_minus_always_C1'] = block[rule] - block['always_C1']
    block['G_CFR_minus_G_CF'] = block['G_CFR'] - block['G_CF']
    write_csv(out / 'repeat_transfer_scene_summary.csv', scene_table + [total, block]); write_json(out / 'repeat_transfer_summary.json', summary)
    if trace_dir.exists() and rc['raw_traces']['files'] == len(records):
        ev = event_rows(blocks, records, trace_dir); fields = []
        for r in ev:
            for k in r:
                if k not in fields: fields.append(k)
        write_csv(out / 'retreat_event_rows.csv', [{k: r.get(k, '') for k in fields} for r in ev], fields)
        write_csv(out / 'retreat_event_summary.csv', event_summary(ev)); cases = case_index(ev, snaps); write_csv(out / 'case_index.csv', cases)
        (out / 'case_timeseries').mkdir(); recs = {r['scene']: r for r in records}; done = set()
        for c in cases:
            k = (c['scene'], c['checkpoint'], c['repeat'])
            if k in done: continue
            done.add(k); e = next(r for r in ev if (r['scene'], r['checkpoint'], r['repeat']) == k)
            write_csv(out / 'case_timeseries' / f'scene{k[0]}_t{k[1]}_r{k[2]}.csv', case_timeseries(trace_dir, recs[k[0]], *k, e['retreat_steps_used']))
        status = 'COMPLETED_POSTHOC_OFFLINE_DIAGNOSTIC'; unusable = sum(1 for r in ev if r['insufficient_trace'])
    else:
        status = 'PARTIAL_INPUT_UNAVAILABLE'; unusable = None
        write_json(out / 'retreat_events_NOT_RUN.json', {'status': 'NOT RUN / raw trace unavailable', 'trace_dir': trace_dir.as_posix()})
    write_json(out / 'run_status.json', {'stage2c': status, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'blocks': len(blocks), 'snapshots': len(snaps),
                                         'repeat_transfer_rows': len(tr), 'event_rows_with_insufficient_trace': unusable,
                                         'observation_based_selector': 'NOT_TRAINED', 'simulator_rollouts': 0, 'policy_inference_calls': 0})
    print(json.dumps({'status': status, 'blocks': len(blocks), 'snapshots': len(snaps), 'transfer_rows': len(tr), 'insufficient_trace': unusable}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    for name in ('manifest', 'run', 'verify'):
        q = s.add_parser(name); q.add_argument('--out', required=True); q.add_argument('--results', default='results/stage2b'); q.add_argument('--run', default='runs/stage2b_001')
        q.add_argument('--backup', default='data_backup/stage2b_001/stage2b_out.tgz')
        if name == 'verify': q.add_argument('--label', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
