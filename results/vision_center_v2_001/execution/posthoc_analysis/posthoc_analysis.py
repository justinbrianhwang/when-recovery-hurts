"""Post hoc description of the Vision-CENTER v2 execution records (written after the outcomes were known; descriptive only).
Ground truth is used here as a label. Nothing in this file was used to choose a constant, and nothing is re-run.
Stage B1 = 300 USED scenes (development data). Stage B2 = 200 new scenes (present only if B1 passed).
Usage, from the repository root: python -X utf8 results/vision_center_v2_001/execution/posthoc_analysis/posthoc_analysis.py"""
import json, csv, gzip
from pathlib import Path
import numpy as np
R = Path('results/vision_center_v2_001/execution'); OUT = R / 'posthoc_analysis'; OLD = Path('results/vision_center_001/run2')
ARM = {'C0': 'C0', 'VISION_CENTER_V1': 'V1', 'VISION_CENTER_V2': 'V2', 'CENTER_ORACLE': 'OR'}
rd = lambda p: json.load(open(p, encoding='utf-8'))


def moved_before_release(rows, release_step):
    """Target cube displacement (m) between the first row of an arm's trace and the step before its release. Label only."""
    if not rows: return None
    last = [x for x in rows if release_step is None or x['t'] < release_step]
    if not last: return None
    a, b = rows[0]['cubeB'], last[-1]['cubeB']; return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def rows(stage, seeds):
    out = []
    for s in seeds:
        r = rd(R / stage / 'scenes' / f'{s}.json'); cells = {ARM[c['arm']]: c for c in r['cells']}
        row = {'scene': s, 'triggered': r['triggered'], **{f'y_{a}': cells[a]['y'] for a in cells}, **{f'steps_{a}': cells[a]['steps'] for a in cells}, 'hand_obs': r['memory']['log']['hand_obs'], 'base_obs': r['memory']['log']['base_obs'],
               'memory_conflicts': r['memory']['log']['conflicts']}
        if r['triggered']:
            g = r['trigger']['gt_for_diagnosis_only']; T = r['trigger']['step']; v1 = cells['V1']; v2 = cells['V2']; c2 = v2['control']; f = c2['readings'][0]; e = v2['gt_reading_error'] or {}; mt = f.get('memory_target') or {}
            tr = json.load(gzip.open(R / stage / r['trace'], 'rt', encoding='utf-8'))['traces']
            src = f['target_src']
            cls = None if src is None else ('current_frame' if src == 'frame' else ('hand_memory' if src == 'memory_hand' else 'base_memory') + ('_last_seen_at_trigger' if mt.get('t_last') == T else '_past_only'))
            row.update(trigger_step=T, true_offset_m=g['offset_xy_m'], true_gap_m=g['height_gap_m'], false_trigger=g['false_trigger'], v1_decision=v1['control']['decision'], v2_decision=c2['decision'], v1_acted=v1['acted'], v2_acted=v2['acted'],
                       v2_target_src=src, v2_hold_src=f['hold_src'], v2_source_class=cls, v2_bound_m=f['u'], v2_est_offset_m=f['offset_norm_m'], v2_est_gap_m=f['gap_m'], v2_offset_error_m=e.get('offset_error_m'), v2_target_error_m=e.get('target_error_m'),
                       v2_error_within_bound=e.get('error_within_bound'), memory_reason=f.get('memory_reason'), mem_src=mt.get('src'), mem_t_last=mt.get('t_last'), mem_age_steps=None if mt.get('t_last') is None else T - mt['t_last'],
                       mem_exposed_steps=mt.get('exposed_steps'), mem_confirmations=mt.get('confirmations'), mem_target_bound_m=mt.get('u'), v1_hold_valid=bool(v1['control']['readings'][0]['hold_fit'] and v1['control']['readings'][0]['hold_fit']['valid']),
                       v2_end_reason=c2['end_reason'], v2_control_steps=v2['control_steps'], v1_end_reason=v1['control']['end_reason'])
            for a in cells:
                x = cells[a]; eff = x['gt_effect_of_control'] or {}; rel = x['gt_release_after_trigger'] or {}
                row[f'pushed_in_control_m_{a}'] = eff.get('cubeB_xy_displacement_m'); row[f'release_offset_m_{a}'] = rel.get('offset_xy_m'); row[f'release_gap_m_{a}'] = rel.get('height_gap_m')
                row[f'target_moved_before_release_m_{a}'] = moved_before_release(tr.get(x['arm']), rel.get('release_step')); row[f'reproduces_trunk_{a}'] = None if x['reproduces_trunk'] is None else x['reproduces_trunk']['match']
        out.append(row)
    return out


def paired(L, ref, arm):
    c = {'both_succeed': 0, 'rescue': 0, 'harm': 0, 'both_fail': 0}
    for x in L: c[('both_succeed' if x[f'y_{arm}'] else 'harm') if x[f'y_{ref}'] else ('rescue' if x[f'y_{arm}'] else 'both_fail')] += 1
    return {**c, 'net': c['rescue'] - c['harm'], 'rescue_scenes': [x['scene'] for x in L if not x[f'y_{ref}'] and x[f'y_{arm}']], 'harm_scenes': [x['scene'] for x in L if x[f'y_{ref}'] and not x[f'y_{arm}']]}


def stat(a):
    a = [float(v) for v in a if v is not None]
    return {'n': len(a), 'median_mm': round(float(np.median(a)) * 1000, 2), 'p90_mm': round(float(np.percentile(a, 90)) * 1000, 2), 'max_mm': round(float(np.max(a)) * 1000, 2)} if a else {'n': 0}


def describe(L, oracle):
    n = len(L); T = [x for x in L if x['triggered']]; arms = ['C0', 'V1', 'V2'] + (['OR'] if oracle else []); acts = [x for x in T if x['v2_acted']]; out = {'scenes': n, 'triggered': len(T)}
    out['success'] = {a: sum(x[f'y_{a}'] for x in L) for a in arms}
    out['paired'] = {'V2_vs_C0': paired(L, 'C0', 'V2'), 'V1_vs_C0': paired(L, 'C0', 'V1'), 'V2_vs_V1': paired(L, 'V1', 'V2'), **({'OR_vs_C0': paired(L, 'C0', 'OR'), 'V2_vs_OR': paired(L, 'OR', 'V2')} if oracle else {})}
    r2 = set(out['paired']['V2_vs_C0']['rescue_scenes']); r1 = set(out['paired']['V1_vs_C0']['rescue_scenes'])
    out['v2_rescues_not_in_v1'] = sorted(r2 - r1); out['v1_rescues_not_in_v2'] = sorted(r1 - r2)
    # --- by source of the target position at the trigger
    by = {}
    for k in ('current_frame', 'hand_memory_past_only', 'hand_memory_last_seen_at_trigger', 'base_memory_past_only', 'base_memory_last_seen_at_trigger'):
        A = [x for x in acts if x['v2_source_class'] == k]
        by[k] = {'acts': len(A), 'success': sum(x['y_V2'] for x in A), 'c0_failure': sum(1 - x['y_C0'] for x in A), 'rescue': sum(1 for x in A if not x['y_C0'] and x['y_V2']), 'harm': sum(1 for x in A if x['y_C0'] and not x['y_V2']),
                 'offset_error': stat([x['v2_offset_error_m'] for x in A]), 'target_error': stat([x['v2_target_error_m'] for x in A]), 'offset_error_outside_bound': sum(1 for x in A if x['v2_error_within_bound'] is False),
                 'offset_error_over_5mm': sum(1 for x in A if (x['v2_offset_error_m'] or 0) > 0.005), 'memory_age_steps': stat([None if x['mem_age_steps'] is None else x['mem_age_steps'] / 1000 for x in A]),
                 'exposed': sum(1 for x in A if (x['mem_exposed_steps'] or 0) > 0), 'scenes': [x['scene'] for x in A], **({'oracle_rescue_among_them': sum(1 for x in A if not x['y_C0'] and x['y_OR'])} if oracle else {})}
    by['note_memory_age_steps'] = 'the memory_age_steps block is in steps, not mm (the helper divides and multiplies by 1000)'
    out['v2_acts_by_target_source'] = by; out['v2_acts_hold_from_memory'] = sum(1 for x in acts if x['v2_hold_src'] == 'memory')
    # --- every remembered target that reached an estimate (acted or not): error against the logged truth and against its own bound
    M = [x for x in T if x['v2_target_src'] not in (None, 'frame') and x['v2_target_error_m'] is not None]
    out['remembered_target_all_estimates'] = {'n': len(M), 'target_error': stat([x['v2_target_error_m'] for x in M]), 'target_error_outside_own_bound': [x['scene'] for x in M if x['v2_target_error_m'] > x['mem_target_bound_m']],
                                              'offset_error_outside_bound': [x['scene'] for x in M if x['v2_error_within_bound'] is False], 'offset_error_over_5mm': [x['scene'] for x in M if (x['v2_offset_error_m'] or 0) > 0.005],
                                              'hand': stat([x['v2_target_error_m'] for x in M if x['mem_src'] == 'hand']), 'base': stat([x['v2_target_error_m'] for x in M if x['mem_src'] == 'base'])}
    F = [x for x in T if x['v2_target_src'] == 'frame' and x['v2_offset_error_m'] is not None]
    out['current_frame_all_estimates'] = {'n': len(F), 'offset_error': stat([x['v2_offset_error_m'] for x in F]), 'offset_error_outside_bound': [x['scene'] for x in F if x['v2_error_within_bound'] is False]}
    # --- abstentions and what they cost
    reasons = sorted({x['v2_decision'] for x in T}); ab = {}
    for k in reasons:
        A = [x for x in T if x['v2_decision'] == k]
        ab[k] = {'n': len(A), 'c0_failure': sum(1 - x['y_C0'] for x in A), 'c0_failure_scenes': [x['scene'] for x in A if not x['y_C0']], 'true_offset_at_least_20mm': sum(1 for x in A if x['true_offset_m'] >= 0.02), 'v1_acted': sum(1 for x in A if x['v1_acted']),
                 **({'oracle_rescue': sum(1 for x in A if not x['y_C0'] and x['y_OR']), 'oracle_rescue_scenes': [x['scene'] for x in A if not x['y_C0'] and x['y_OR']]} if oracle else {})}
    out['v2_decisions'] = ab; out['v1_decisions'] = {k: sum(1 for x in T if x['v1_decision'] == k) for k in sorted({x['v1_decision'] for x in T})}
    out['c0_failures'] = {'all': sum(1 - x['y_C0'] for x in L), 'not_triggered': sum(1 - x['y_C0'] for x in L if not x['triggered']), 'triggered': sum(1 - x['y_C0'] for x in T), 'v2_acted': sum(1 - x['y_C0'] for x in acts),
                          'v2_acted_and_still_failed': [x['scene'] for x in acts if not x['y_C0'] and not x['y_V2']]}
    if oracle:
        orr = [x for x in L if not x['y_C0'] and x['y_OR']]
        out['oracle_rescues'] = {'n': len(orr), 'also_v2': sum(x['y_V2'] for x in orr), 'also_v1': sum(x['y_V1'] for x in orr),
                                 'missed_by_v2': [{'scene': x['scene'], 'v2_decision': x['v2_decision'], 'v1_decision': x['v1_decision'], 'target_source': x['v2_source_class'], 'true_offset_mm': round(x['true_offset_m'] * 1000, 1),
                                                   'true_gap_mm': round(x['true_gap_m'] * 1000, 1), 'memory_reason': x['memory_reason'], 'v2_end_reason': x['v2_end_reason'], 'release_offset_V2_mm': None if x['release_offset_m_V2'] is None else round(x['release_offset_m_V2'] * 1000, 1)}
                                                  for x in orr if not x['y_V2']],
                                 'v2_rescues_the_oracle_did_not': [x['scene'] for x in L if not x['y_C0'] and x['y_V2'] and not x['y_OR']], 'oracle_harm_scenes': [x['scene'] for x in L if x['y_C0'] and not x['y_OR']]}
    H = [x for x in T if x['v2_decision'] == 'NO_HOLD_ESTIMATE']
    out['held_cube_fit_not_accepted'] = {'n': len(H), 'c0_failure': sum(1 - x['y_C0'] for x in H), 'true_offset_at_least_20mm': sum(1 for x in H if x['true_offset_m'] >= 0.02), **({'oracle_rescue': sum(1 for x in H if not x['y_C0'] and x['y_OR'])} if oracle else {}),
                                         'v1_fit_valid_at_trigger': sum(1 for x in H if x['v1_hold_valid'])}
    # --- target pushed, contact
    out['target_moved'] = {a: {'acts': sum(1 for x in T if x.get(f'pushed_in_control_m_{a}') is not None), 'pushed_over_1mm_during_control': [x['scene'] for x in T if (x.get(f'pushed_in_control_m_{a}') or 0) > 0.001],
                               'max_during_control_mm': round(max([x.get(f'pushed_in_control_m_{a}') or 0 for x in T] or [0]) * 1000, 2),
                               'moved_over_1mm_between_trigger_and_release': [x['scene'] for x in T if (x.get(f'target_moved_before_release_m_{a}') or 0) > 0.001]} for a in arms}
    out['v2_end_reasons'] = {k: sum(1 for x in acts if x['v2_end_reason'] == k) for k in sorted({x['v2_end_reason'] for x in acts})}
    out['v2_acts_outcome_by_end_reason'] = {k: {'n': sum(1 for x in acts if x['v2_end_reason'] == k), 'success': sum(x['y_V2'] for x in acts if x['v2_end_reason'] == k)} for k in sorted({x['v2_end_reason'] for x in acts})}
    # --- reproduction inside this host: C0 and every abstaining vision arm must equal the uninterrupted trunk (the harness stops otherwise)
    out['same_host_reproduction'] = {a: {'checked': sum(1 for x in T if x.get(f'reproduces_trunk_{a}') is not None), 'match': sum(1 for x in T if x.get(f'reproduces_trunk_{a}') is True)} for a in ('C0', 'V1', 'V2')}
    return out


def host_difference(L):
    """Stage B1 scenes were also run on another host in Vision-CENTER run2 (50600-50799 as its stage B1, 50800-50899 as its stage B2)."""
    out = {'compared': 0, 'c0_outcome_differs': [], 'trigger_differs': [], 'v1_decision_differs': [], 'v1_outcome_differs': []}
    for x in L:
        s = x['scene']; p = OLD / ('shadow' if s < 50800 else 'pilot') / 'scenes' / f'{s}.json'
        if not p.exists(): continue
        o = rd(p); oc = {c['arm']: c for c in o['cells']}; out['compared'] += 1
        if oc['C0']['y'] != x['y_C0']: out['c0_outcome_differs'].append(s)
        if o['triggered'] != x['triggered'] or (o['triggered'] and o['trigger']['step'] != x['trigger_step']): out['trigger_differs'].append(s)
        if o['triggered'] and x['triggered']:
            if oc['VISION_CENTER']['control']['decision'] != x['v1_decision']: out['v1_decision_differs'].append(s)
        if oc['VISION_CENTER']['y'] != x['y_V1']: out['v1_outcome_differs'].append(s)
    out['old_host'] = 'Vision-CENTER run2: South Korea, machine 151205, RTX 4090, Intel Core i7-13700K, driver 595.91.07'
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True); res = {'status': 'post hoc, descriptive; ground truth is a label only; B1 scenes are used development data'}
    for stage, seeds, oracle, name in (('shadow', range(50600, 50900), False, 'B1_shadow_used_scenes'), ('pilot', range(50900, 51100), True, 'B2_pilot_new_scenes')):
        if not (R / stage / 'scenes').exists(): res[name] = 'NOT_RUN'; continue
        L = rows(stage, seeds); res[name] = describe(L, oracle)
        if stage == 'shadow': res[name]['host_difference_against_run2'] = host_difference(L)
        keys = []
        for x in L:
            for k in x:
                if k not in keys: keys.append(k)
        with open(OUT / f'per_scene_{name}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(L)
    json.dump(res, open(OUT / 'posthoc_analysis.json', 'w', encoding='utf-8'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
