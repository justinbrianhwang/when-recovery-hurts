"""Recovery-V1 oracle XY pilot, run 1: descriptive tables from the stored scene records (POST HOC, written after the outcomes were known).
Nothing here is part of the frozen analysis; the frozen tables are in ../result.json. All 100 scenes are in every table; no scene is dropped.
Usage: python posthoc_analysis.py <dir with the 100 scene records> <out.json> <out.csv>
"""
import csv, json, statistics as st, sys
from pathlib import Path

ARMS = ('C0', 'R5', 'HOLD_MATCHED', 'CENTER_ORACLE'); TOL = 0.010


def q(v):
    v = sorted(x for x in v if x is not None)
    return None if not v else {'n': len(v), 'min': v[0], 'median': st.median(v), 'max': v[-1]}


def main(src, out_json, out_csv):
    rows = []
    for p in sorted(Path(src).glob('5*.json')):
        r = json.loads(p.read_text(encoding='utf-8')); c = {x['arm']: x for x in r['cells']}; ref = r['reference'] or {}; tr = r['trigger'] or {}
        gt = tr.get('gt_for_diagnosis_only') or {}; ce = c['CENTER_ORACLE'].get('gt_effect_of_control') or {}
        row = {'scene': r['scene'], 'triggered': bool(r['triggered']), 'trigger_step': tr.get('step'), 'on_stage2_grid': tr.get('on_stage2_eligibility_grid'), 't_center': ref.get('t_center'),
               'center_end_reason': ref.get('end_reason'), 'offset_at_trigger_m': gt.get('offset_xy_m'), 'offset_after_center_control_m': ce.get('offset_xy_after_m'),
               'center_tcp_xy_displacement_m': ce.get('tcp_xy_displacement_m'), 'center_tcp_dz_m': ce.get('tcp_dz_m'), 'center_commanded_path_m': ref.get('commanded_path_m'),
               'center_grasp_kept': ce.get('grasped_after'), 'trunk_steps': r['trunk']['steps'], 'c0_reproduces_trunk': r['checks']['c0_reproduces_trunk']['match'] if r.get('checks') else None,
               'handoff_pass': (r.get('handoff') or {}).get('pass')}
        for a in ARMS:
            rel = c[a].get('gt_release_after_trigger') or {}
            row[f'y_{a}'] = int(c[a]['y']); row[f'steps_{a}'] = c[a]['steps']; row[f'release_found_{a}'] = rel.get('found'); row[f'release_step_{a}'] = rel.get('release_step'); row[f'release_offset_m_{a}'] = rel.get('offset_xy_m')
            row[f'oracle_reads_{a}'] = c[a].get('oracle_reads', 0); row[f'ended_during_control_{a}'] = bool(c[a].get('ended_during_control'))
        rows.append(row)
    assert len(rows) == 100, len(rows)
    with open(out_csv, 'x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def cnt(sel): return {a: sum(x[f'y_{a}'] for x in sel) for a in ARMS} | {'n': len(sel)}
    trig = [x for x in rows if x['triggered']]; moved = [x for x in trig if x['t_center'] > 0]; aligned = [x for x in trig if x['t_center'] == 0]; notrig = [x for x in rows if not x['triggered']]
    c0f = [x for x in rows if not x['y_C0']]; c0s = [x for x in rows if x['y_C0']]
    strata = {'all': cnt(rows), 'no_trigger': cnt(notrig), 'triggered': cnt(trig), 'moved_T_gt_0': cnt(moved), 'already_aligned_T_eq_0': cnt(aligned), 'c0_success': cnt(c0s), 'c0_failure': cnt(c0f),
              'c0_failure_no_trigger': cnt([x for x in c0f if not x['triggered']]), 'c0_failure_moved': cnt([x for x in c0f if x['triggered'] and x['t_center'] > 0]), 'c0_failure_aligned': cnt([x for x in c0f if x['triggered'] and x['t_center'] == 0]),
              'c0_success_moved': cnt([x for x in c0s if x['triggered'] and x['t_center'] > 0]), 'c0_success_aligned': cnt([x for x in c0s if x['triggered'] and x['t_center'] == 0])}
    # behaviour after the hand-back, CENTER_ORACLE, moved scenes: offset when the control part ended against offset at the policy's own release
    after = []
    for x in moved:
        a, b, o = x['offset_after_center_control_m'], x['release_offset_m_CENTER_ORACLE'], x['offset_at_trigger_m']
        after.append({'scene': x['scene'], 't_center': x['t_center'], 'end_reason': x['center_end_reason'], 'offset_at_trigger_m': o, 'offset_after_control_m': a, 'release_found': x['release_found_CENTER_ORACLE'], 'offset_at_release_m': b,
                      'change_after_handback_m': None if b is None else b - a, 'release_delay_steps': None if b is None else x['release_step_CENTER_ORACLE'] - (x['trigger_step'] + x['t_center']),
                      'y_C0': x['y_C0'], 'y_HOLD_MATCHED': x['y_HOLD_MATCHED'], 'y_CENTER_ORACLE': x['y_CENTER_ORACLE']})
    rel = [x for x in after if x['offset_at_release_m'] is not None]
    undo = {'definition': 'post hoc, descriptive: CENTER_ORACLE scenes with T > 0; offset between the cube centres when the control part ended, against the offset at the first release the policy made after it got control back',
            'moved_scenes': len(after), 'release_found': len(rel), 'no_release_found_scenes': [x['scene'] for x in after if x['offset_at_release_m'] is None],
            'change_after_handback_m': q([x['change_after_handback_m'] for x in rel]), 'release_delay_steps': q([x['release_delay_steps'] for x in rel]),
            'released_farther_than_at_end_of_control_by_more_than_2mm': sorted(x['scene'] for x in rel if x['change_after_handback_m'] > 0.002),
            'released_closer_than_at_end_of_control_by_more_than_2mm': sorted(x['scene'] for x in rel if x['change_after_handback_m'] < -0.002),
            'control_ended_within_tolerance_but_released_above_tolerance': sorted(x['scene'] for x in rel if x['offset_after_control_m'] <= TOL and x['offset_at_release_m'] > TOL),
            'released_at_or_beyond_the_offset_at_trigger': sorted(x['scene'] for x in rel if x['offset_at_release_m'] >= x['offset_at_trigger_m']),
            'released_within_tolerance': sum(1 for x in rel if x['offset_at_release_m'] <= TOL), 'success_by_release_within_tolerance': {
                'within': [sum(x['y_CENTER_ORACLE'] for x in rel if x['offset_at_release_m'] <= TOL), sum(1 for x in rel if x['offset_at_release_m'] <= TOL)],
                'above': [sum(x['y_CENTER_ORACLE'] for x in rel if x['offset_at_release_m'] > TOL), sum(1 for x in rel if x['offset_at_release_m'] > TOL)]}}
    hold_undo = q([x['release_offset_m_HOLD_MATCHED'] - x['offset_at_trigger_m'] for x in moved if x['release_offset_m_HOLD_MATCHED'] is not None])
    pick = lambda sel: [{k: x[k] for k in ('scene', 'triggered', 'trigger_step', 't_center', 'center_end_reason', 'offset_at_trigger_m', 'offset_after_center_control_m', 'center_tcp_xy_displacement_m', 'release_offset_m_C0', 'release_offset_m_HOLD_MATCHED',
                                              'release_offset_m_CENTER_ORACLE', 'y_C0', 'y_R5', 'y_HOLD_MATCHED', 'y_CENTER_ORACLE')} for x in sel]
    out = {'status': 'POST HOC, descriptive; the frozen tables are in result.json', 'scenes': len(rows), 'success_by_stratum': strata,
           'c0_release_offset_by_c0_outcome_m': {'c0_success': q([x['release_offset_m_C0'] for x in c0s if x['triggered']]), 'c0_failure_triggered': q([x['release_offset_m_C0'] for x in c0f if x['triggered']])},
           'center_control_moved': {'tcp_xy_displacement_m': q([x['center_tcp_xy_displacement_m'] for x in moved]), 'commanded_path_m': q([x['center_commanded_path_m'] for x in moved]), 'steps': q([x['t_center'] for x in moved]),
                                    'abs_tcp_dz_m_max': max(abs(x['center_tcp_dz_m']) for x in moved), 'end_reasons': {k: sum(1 for x in moved if x['center_end_reason'] == k) for k in sorted({x['center_end_reason'] for x in moved})},
                                    'grasp_kept': sum(1 for x in moved if x['center_grasp_kept']), 'offset_at_trigger_m': q([x['offset_at_trigger_m'] for x in moved]), 'offset_after_control_m': q([x['offset_after_center_control_m'] for x in moved])},
           'after_handback_CENTER_ORACLE': undo, 'HOLD_MATCHED_release_offset_minus_offset_at_trigger_m_moved': hold_undo,
           'aligned_scenes_CENTER_equals_HOLD': all(x['y_CENTER_ORACLE'] == x['y_HOLD_MATCHED'] and x['steps_CENTER_ORACLE'] == x['steps_HOLD_MATCHED'] for x in aligned),
           'aligned_scenes_differ_from_C0': sorted(x['scene'] for x in aligned if x['y_CENTER_ORACLE'] != x['y_C0']), 'aligned_scenes_step_count_differs_from_C0': sum(1 for x in aligned if x['steps_CENTER_ORACLE'] != x['steps_C0']),
           'trigger': {'steps': q([x['trigger_step'] for x in trig]), 'on_stage2_grid': sum(1 for x in trig if x['on_stage2_grid']), 'no_trigger_scenes': [x['scene'] for x in notrig]},
           'technical': {'c0_reproduces_trunk': sum(1 for x in trig if x['c0_reproduces_trunk']), 'handoff_pass': sum(1 for x in trig if x['handoff_pass']), 'triggered': len(trig),
                         'oracle_reads_by_arm': {a: sum(x[f'oracle_reads_{a}'] for x in rows) for a in ARMS}, 'ended_during_control': {a: sorted(x['scene'] for x in rows if x[f'ended_during_control_{a}']) for a in ARMS}},
           'scenes_c0_failure': pick(c0f), 'scenes_center_step_budget_or_episode_end': pick([x for x in moved if x['center_end_reason'] != 'within_tolerance']), 'scenes_HOLD_differs_from_C0': pick([x for x in rows if x['y_HOLD_MATCHED'] != x['y_C0']]),
           'after_handback_per_scene': after}
    with open(out_json, 'x', encoding='utf-8') as f: json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('after_handback_per_scene',)}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
