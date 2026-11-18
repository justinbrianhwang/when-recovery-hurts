"""Vision-CENTER v2, Phase C0 (post-pass): audit of the remembered target position against the logged ground truth.

Reads the stored scene records and per-step traces of stage B1 (300 used scenes) and stage B2 (200 new scenes). Nothing is
simulated and no record is changed. Ground truth (cube positions in the traces) is used only as an after-the-fact label.
The 'what if a rule were absent' part re-applies the decision ladder of hci.vision_center_v2.decide to the LOGGED estimate
with rules switched off. It says which decision the logged numbers would have led to; it is not a rollout and gives no
outcome. Run from the repository root:

    python results/vision_center_v2_001/postpass/memory_risk.py
"""
import json, gzip, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]; EXE = ROOT / 'results/vision_center_v2_001/execution'; OUT = Path(__file__).resolve().parent / 'memory_risk.json'
sys.path.insert(0, str(ROOT / 'src'))
from hci import vision_center_v2 as V2          # constants and the unmodified decision function only

STAGES = (('B1_shadow_used', 'shadow', 50600, 50899), ('B2_pilot_new', 'pilot', 50900, 51099))
CASES = {'B2_pilot_new': [51009, 50959], 'B1_shadow_used': [50623, 50702, 50661]}
mm = lambda x: None if x is None else round(1000 * float(x), 2)


def cell(r, a): return next(x for x in r['cells'] if x['arm'] == a)


def ladder(rd, steps_left, off=()):
    """The decision ladder of V2.decide on a logged reading, with the rules named in `off` removed."""
    if rd['why'] is not None: return rd['why']
    n = rd['offset_norm_m']; u = rd['u']; gap = rd['gap_m']; rem = rd['target_src'] != 'frame'
    if 'carry_height' not in off and rd['held_xyz'][2] < V2.CARRY_Z_MIN_M: return 'LOW_CARRY'
    if 'memory_gap' not in off and rem and gap < V2.MEMORY_GAP_MIN_M: return 'LOW_CARRY'
    if 'side_contact' not in off and n + (u if rem else 0.) >= V2.SIDE_OFFSET_M and gap < V2.SIDE_GAP_MIN_M: return 'SIDE_CONTACT_RISK'
    if steps_left < V2.STEPS_LEFT_MIN: return 'TOO_FEW_STEPS'
    if n < V2.ACT_MIN_M: return 'ALIGNED'
    if 'uncertainty' not in off and n - u < V2.RESIDUAL_TOL_M: return 'ALIGNED_WITHIN_UNCERTAINTY'
    return 'ACT'


def rows_by_t(trace, initial):
    by = {int(x['t']): x for x in trace}; by.setdefault(0, initial); return by


def scene_row(stage, folder, r):
    v = cell(r, 'VISION_CENTER_V2'); ctl = v['control']; rd = ctl['readings'][0]; t = r['trigger']['step']; gt = r['trigger']['gt_for_diagnosis_only']; e = v['gt_reading_error'] or {}
    tm = rd.get('memory_target'); row = {'stage': stage, 'scene': r['scene'], 'trigger_step': t, 'decision': ctl['decision'], 'acted': v['acted'], 'target_src': rd['target_src'], 'hold_src': rd['hold_src'], 'y_C0': cell(r, 'C0')['y'], 'y_V2': v['y'],
                                         'bound_offset_m': rd['u'], 'offset_error_m': e.get('offset_error_m'), 'target_error_m': e.get('target_error_m'), 'true_offset_m': e.get('true_offset_m'), 'est_offset_m': rd['offset_norm_m'], 'gap_m': rd['gap_m'],
                                         'steps_left': r['trigger']['remaining_budget']}
    assert ladder(rd, row['steps_left']) == ctl['decision'], (r['scene'], ladder(rd, row['steps_left']), ctl['decision'])       # the re-implemented ladder reproduces every logged decision
    if tm is not None:
        tr = json.load(gzip.open(EXE / folder / 'traces' / f'trace_{r["scene"]}.json.gz'))['traces']['trunk']; by = rows_by_t(tr, r['initial_state']); tl = int(tm['t_last'])
        b_last = np.array(by[tl]['cubeB'][:2]); b_trig = np.array(gt['cubeB'][:2]); est = np.array(tm['xy'])
        path = [float(np.hypot(*(np.array(by[k]['cubeB'][:2]) - b_last))) for k in range(tl, t + 1) if k in by]
        row.update(mem={'src': tm['src'], 't_last': tl, 'age_steps': t - tl, 'exposed_steps': tm['exposed_steps'], 'confirmations': tm['confirmations'], 'bound_target_m': tm['u'],
                        'error_at_last_view_m': float(np.hypot(*(est - b_last))), 'target_moved_after_last_view_m': float(np.hypot(*(b_trig - b_last))), 'target_error_at_trigger_m': float(np.hypot(*(est - b_trig))),
                        'first_step_moved_over_1mm': next((tl + i for i, d in enumerate(path) if d > 0.001), None),
                        'within_target_bound': bool(np.hypot(*(est - b_trig)) <= tm['u']), 'last_view_is_the_trigger_step': tl == t,
                        'confirmed_only_with_the_trigger_step': bool(tl == t and tm['confirmations'] <= V2.CONFIRM_MIN),
                        'hand_check': rd.get('memory_check')})
    if v['acted']:
        srcs = [x['target_src'] for x in ctl['readings']]; eff = v['gt_effect_of_control'] or {}
        row.update(control={'steps': ctl['steps_used'], 'end_reason': ctl['end_reason'], 'target_src_per_reading': srcs, 'never_saw_target_during_move': all(s != 'frame' for s in srcs),
                            'switched_to_current_frame_at_reading': next((i for i, s in enumerate(srcs) if s == 'frame'), None) if srcs[0] != 'frame' else None,
                            'target_pushed_during_control_m': eff.get('cubeB_xy_displacement_m'), 'offset_after_m': eff.get('offset_xy_after_m')})
    row['what_if'] = {'as_logged': ctl['decision'], **{f'without_{"+".join(o)}': ladder(rd, row['steps_left'], o) for o in (('carry_height',), ('memory_gap',), ('carry_height', 'memory_gap'), ('side_contact',), ('uncertainty',),
                                                                                                               ('carry_height', 'memory_gap', 'side_contact', 'uncertainty'))}}
    return row


def case(stage, folder, scene, row):
    r = json.loads((EXE / folder / 'scenes' / f'{scene}.json').read_text(encoding='utf-8')); t = r['trigger']['step']; tr = json.load(gzip.open(EXE / folder / 'traces' / f'trace_{scene}.json.gz'))['traces']['trunk']; by = rows_by_t(tr, r['initial_state'])
    m = r['memory']; tl = row.get('mem', {}).get('t_last'); out = {'scene': scene, 'stage': stage, 'summary': row, 'outcomes': {x['arm']: x['y'] for x in r['cells']}, 'acted': {x['arm']: x.get('acted') for x in r['cells']},
                                                                     'memory_observation_steps': {'hand': [m['hand_steps'][0], m['hand_steps'][-1], len(m['hand_steps'])] if m['hand_steps'] else None,
                                                                                                  'base': [m['base_steps'][0], m['base_steps'][-1], len(m['base_steps'])] if m['base_steps'] else None, 'log': m['log']}}
    if tl is not None:
        b0 = np.array(by[tl]['cubeB'][:2]); out['timeline_from_last_view'] = [{'t': k, 'target_moved_since_last_view_mm': mm(np.hypot(*(np.array(by[k]['cubeB'][:2]) - b0))), 'tcp_to_target_xy_mm': mm(np.hypot(*(np.array(by[k]['tcp'][:2]) - np.array(by[k]['cubeB'][:2])))),
                                                                              'tcp_z_mm': mm(by[k]['tcp'][2]), 'heldA_z_mm': mm(by[k]['cubeA'][2]), 'grasped': by[k].get('is_cubeA_grasped')} for k in range(tl, t + 1) if k in by]
    v = cell(r, 'VISION_CENTER_V2')
    if v['acted']: out['control_readings'] = [{'t': x['t'], 'target_src': x['target_src'], 'offset_mm': mm(x['offset_norm_m']), 'bound_mm': mm(x['u'])} for x in v['control']['readings']]; out['effect_of_control'] = v['gt_effect_of_control']
    o = next((x for x in r['cells'] if x['arm'] == 'CENTER_ORACLE'), None)
    if o is not None: out['oracle_effect_of_control'] = o['gt_effect_of_control']
    return out


def q(a):
    a = [float(x) for x in a if x is not None]
    return {'n': len(a), 'median_mm': mm(np.median(a)), 'p90_mm': mm(np.quantile(a, 0.9)), 'max_mm': mm(max(a))} if a else {'n': 0}


def coverage(rows):
    mem = [x for x in rows if 'mem' in x]; out = {}
    for src in ('hand', 'base'):
        s = [x for x in mem if x['mem']['src'] == src]; ex = [x for x in s if x['mem']['exposed_steps'] > 0]; ne = [x for x in s if x['mem']['exposed_steps'] == 0]
        out[src] = {'estimates_at_the_trigger': len(s), 'acted_on': sum(1 for x in s if x['acted']), 'error_at_last_view': q(x['mem']['error_at_last_view_m'] for x in s), 'target_moved_after_last_view': q(x['mem']['target_moved_after_last_view_m'] for x in s),
                    'error_at_trigger': q(x['mem']['target_error_at_trigger_m'] for x in s), 'outside_own_bound': [x['scene'] for x in s if not x['mem']['within_target_bound']],
                    'moved_over_1mm_after_last_view': sum(1 for x in s if x['mem']['target_moved_after_last_view_m'] > 0.001), 'moved_over_5mm_after_last_view': [x['scene'] for x in s if x['mem']['target_moved_after_last_view_m'] > 0.005],
                    'exposure_flag': {'flagged': len(ex), 'not_flagged': len(ne), 'moved_over_1mm_among_flagged': sum(1 for x in ex if x['mem']['target_moved_after_last_view_m'] > 0.001),
                                      'moved_over_1mm_among_not_flagged': [x['scene'] for x in ne if x['mem']['target_moved_after_last_view_m'] > 0.001],
                                      'moved_over_push_allowance_among_flagged': [x['scene'] for x in ex if x['mem']['target_moved_after_last_view_m'] > V2.U_PUSH_M]},
                    'age_steps': {'median': float(np.median([x['mem']['age_steps'] for x in s])) if s else None, 'max': max((x['mem']['age_steps'] for x in s), default=None)},
                    'last_view_is_the_trigger_step': [x['scene'] for x in s if x['mem']['last_view_is_the_trigger_step']], 'of_which_acted': [x['scene'] for x in s if x['mem']['last_view_is_the_trigger_step'] and x['acted']],
                    'confirmed_only_with_the_trigger_step': [x['scene'] for x in s if x['mem']['confirmed_only_with_the_trigger_step']],
                    'hand_check_had_no_pixels_to_test': sum(1 for x in s if x['mem']['hand_check'] and x['mem']['hand_check']['expected_pixels_unoccluded'] == 0 and x['mem']['hand_check']['lit_pixels_outside'] == 0)}
    acts = [x for x in rows if x['acted']]; ma = [x for x in acts if x['target_src'] != 'frame']
    out['moves_on_a_remembered_target'] = {'n': len(ma), 'never_saw_target_during_move': sum(1 for x in ma if x['control']['never_saw_target_during_move']), 'switched_to_current_frame': sum(1 for x in ma if not x['control']['never_saw_target_during_move']),
                                           'control_steps_total': sum(x['control']['steps'] for x in ma), 'target_pushed_max_mm': mm(max((x['control']['target_pushed_during_control_m'] or 0.) for x in ma)) if ma else None,
                                           'end_reasons': {k: sum(1 for x in ma if x['control']['end_reason'] == k) for k in sorted({x['control']['end_reason'] for x in ma})}, 'succeeded': sum(x['y_V2'] for x in ma),
                                           'offset_error': q(x['offset_error_m'] for x in ma), 'offset_error_over_5mm': [x['scene'] for x in ma if x['offset_error_m'] > 0.005]}
    out['moves_on_the_current_frame'] = {'n': len(acts) - len(ma), 'offset_error': q(x['offset_error_m'] for x in acts if x['target_src'] == 'frame'), 'hold_from_memory': [x['scene'] for x in acts if x['hold_src'] == 'memory']}
    lc = [x for x in rows if x['decision'] == 'LOW_CARRY']
    out['low_carry_abstentions_what_if'] = {'n': len(lc), 'without_both_height_rules': {k: sum(1 for x in lc if x['what_if']['without_carry_height+memory_gap'] == k) for k in sorted({x['what_if']['without_carry_height+memory_gap'] for x in lc})},
                                            'would_act_scenes': [{'scene': x['scene'], 'target_src': x['target_src'], 'gap_mm': mm(x['gap_m']), 'est_offset_mm': mm(x['est_offset_m']), 'true_offset_mm': mm(x['true_offset_m']), 'offset_error_mm': mm(x['offset_error_m']), 'y_C0': x['y_C0']}
                                                                 for x in lc if x['what_if']['without_carry_height+memory_gap'] == 'ACT'],
                                            'note': 'decision level only: which rung the logged estimate reaches with both height rules removed. No rollout, no outcome.'}
    return out


def main():
    res = {'work_id': 'vision_center_v2_001', 'phase': 'C0 post-pass', 'status': 'offline audit of stored records; exploratory; ground truth is a label only', 'constants': {'memory': V2.MEMORY, 'decision': V2.DECISION, 'memory_error_m': V2.MEMORY_ERROR_M}, 'stages': {}, 'cases': []}
    for stage, folder, lo, hi in STAGES:
        recs = [json.loads((EXE / folder / 'scenes' / f'{s}.json').read_text(encoding='utf-8')) for s in range(lo, hi + 1)]; rows = [scene_row(stage, folder, r) for r in recs if r['triggered']]; byscene = {x['scene']: x for x in rows}
        res['stages'][stage] = {'scenes': len(recs), 'triggered': len(rows), 'ladder_reproduces_every_logged_decision': True, **coverage(rows)}
        for s in CASES[stage]: res['cases'].append(case(stage, folder, s, byscene[s]))
        res['stages'][stage]['rows'] = rows
    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')
    for k, v in res['stages'].items():
        print('==', k, v['scenes'], v['triggered'])
        for s in ('hand', 'base', 'moves_on_a_remembered_target', 'moves_on_the_current_frame', 'low_carry_abstentions_what_if'): print(' ', s, json.dumps(v[s]))
    for c in res['cases']:
        s = c['summary']; print('-- case', c['scene'], s['decision'], s['target_src'], s['hold_src'], 'offset err', mm(s['offset_error_m']), 'target err', mm(s['target_error_m']), 'bound', mm(s['bound_offset_m']), c['outcomes'])
        print('   mem', json.dumps(s.get('mem'))); print('   what_if', json.dumps(s['what_if'])); print('   ctl', json.dumps(s.get('control')))
        tl = c.get('timeline_from_last_view') or []; print('   timeline', json.dumps(tl if len(tl) <= 16 else tl[:3] + tl[-12:]))


if __name__ == '__main__': main()
