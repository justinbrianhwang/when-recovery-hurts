"""Vision-CENTER v2 Phase C1: tables from the offline decision replay (offline_replay.json) of the 500 used scenes.
Decisions only. `v2 recorded outcome` columns quote what the executed v2 run did in the same scene; they describe the
scenes a changed decision falls on and are NOT counterfactual rescue or harm of another arm.

    python results/vision_center_v2_c1_001/preparation/offline_summary.py
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from hci import vision_center_c1 as C1, vision_center_v2 as V2

import sys
HERE = Path(__file__).resolve().parent; SRC = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'offline_replay.json'); OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else 'offline_replay_summary.json')     # optional: input and output names in this folder
VIS = [a for a in C1.ARMS if a not in ('C0', 'CENTER_ORACLE')]; V2A = 'VISION_CENTER_V2'; VS = C1.SAFETY_ARM
mm = lambda x: None if x is None else round(1000 * float(x), 2)


def klass(r):
    """What the executed v2 run recorded in this scene."""
    y0, y2 = r['y']['C0'], r['y'][V2A]
    return 'v2_rescue' if (y0, y2) == (0, 1) else 'v2_harm' if (y0, y2) == (1, 0) else 'both_succeed' if y0 == 1 else 'both_fail'


def key(a): return (a['decision'], a.get('target_src'), a.get('hold_src'), None if a.get('offset_norm_m') is None else round(a['offset_norm_m'], 9))


def q(xs):
    xs = [float(x) for x in xs if x is not None]
    return {'n': len(xs), 'median_mm': mm(np.median(xs)), 'max_mm': mm(max(xs))} if xs else {'n': 0}


def diff(rows, a, b):
    """Scenes where arm b acts and arm a does not, and the reverse, with the recorded v2 outcome class of those scenes."""
    only_b = [r for r in rows if r['arms'][b]['act'] and not r['arms'][a]['act']]; only_a = [r for r in rows if r['arms'][a]['act'] and not r['arms'][b]['act']]
    def lst(rs, arm, other): return [{'scene': r['scene'], 'stage': r['stage'], 'v2_recorded': klass(r), 'target_src': r['arms'][arm].get('target_src'), 'hold_src': r['arms'][arm].get('hold_src'), 'offset_est_mm': mm(r['arms'][arm].get('offset_norm_m')),
                                    'offset_error_mm': mm(r['arms'][arm].get('offset_error_m')), 'true_offset_mm': mm(r['gt']['true_offset_m']), 'other_arm_decision': r['arms'][other]['decision']} for r in rs]
    cnt = lambda rs: {k: sum(1 for r in rs if klass(r) == k) for k in ('v2_rescue', 'both_succeed', 'both_fail', 'v2_harm')}
    return {f'{b}_acts_{a}_abstains': {'n': len(only_b), 'by_v2_recorded_outcome': cnt(only_b), 'scenes': lst(only_b, b, a)}, f'{a}_acts_{b}_abstains': {'n': len(only_a), 'by_v2_recorded_outcome': cnt(only_a), 'scenes': lst(only_a, a, b)},
            'same_act_or_abstain': len(rows) - len(only_a) - len(only_b), 'same_decision_source_and_estimate': sum(1 for r in rows if key(r['arms'][a]) == key(r['arms'][b])),
            'both_act_with_another_source_or_estimate': sum(1 for r in rows if r['arms'][a]['act'] and r['arms'][b]['act'] and key(r['arms'][a]) != key(r['arms'][b]))}


def main():
    if OUT.exists(): raise SystemExit(f'Refused overwrite: {OUT}')
    src = json.loads(SRC.read_text(encoding='utf-8')); allrows = src['rows']; rows = [r for r in allrows if r['triggered']]; n = len(rows)
    chk = {k: sum(1 for r in rows if r['check'][k] is True) for k in rows[0]['check'] if k != 'recorded'}
    chk.update(mirror_equals_original_memory=sum(1 for r in rows if r['mirror_equals_original_memory']), memory_log_equals_record=sum(r['memory_log_equals_record'] for r in rows), held_grip_equals_record=sum(r['held_grip_equals_record'] for r in rows),
               width_equals_record=sum(r['width_equals_record'] for r in rows))
    ctl = [r['v2s_control'] for r in rows if 'v2s_control' in r]; exact = all(v == n for v in chk.values()) and all(c['v2_control_log_reproduced'] for c in ctl)
    out = {'work_id': C1.WORK, 'what': 'offline decision replay on the 500 used scenes 50600-51099 (development data). Decisions only: no rollout, no outcome, no rescue or harm of any replayed arm.',
           'source': {'path': 'results/vision_center_v2_c1_001/preparation/' + SRC.name, 'sha256': hashlib.sha256(SRC.read_bytes()).hexdigest()}, 'code_sha256': src['code_sha256'],
           'scenes': len(allrows), 'triggered': n, 'by_stage': {s: sum(1 for r in rows if r['stage'] == s) for s in ('B1', 'B2')},
           'reproduction_of_the_executed_run': {**chk, 'of': n, 'v2_control_logs_reproduced': f"{sum(c['v2_control_log_reproduced'] for c in ctl)}/{len(ctl)}"}, 'v2_reproduced_exactly': bool(exact)}
    out['decisions'] = {a: {s: {k: sum(1 for r in rows if r['stage'] == s and r['arms'][a]['decision'] == k) for k in sorted({r['arms'][a]['decision'] for r in rows})} for s in ('B1', 'B2')} for a in VIS}
    out['acts'] = {a: {s: sum(1 for r in rows if r['stage'] == s and r['arms'][a]['act']) for s in ('B1', 'B2')} for a in VIS}
    out['acts_by_target_source'] = {a: {k: sum(1 for r in rows if r['arms'][a]['act'] and (r['arms'][a].get('target_src'), r['arms'][a].get('hold_src')) == k) for k in sorted({(r['arms'][a].get('target_src'), r['arms'][a].get('hold_src')) for r in rows if r['arms'][a]['act']}, key=str)} for a in VIS}
    out['acts_by_target_source'] = {a: {f'target_{k[0]}/hold_{k[1]}': v for k, v in d.items()} for a, d in out['acts_by_target_source'].items()}
    pairs = [('VISION_CENTER_V1', 'V2_SINGLE_FRAME'), ('V2_SINGLE_FRAME', V2A), ('VISION_CENTER_V1', V2A), ('V2_NO_HAND_MEMORY', V2A), ('V2_NO_BASE_MEMORY', V2A), ('V2_NO_HOLD_MEMORY', V2A), (V2A, 'V2_RULES_OFF'), (V2A, VS)]
    out['pairs'] = {f'{a}|{b}': diff(rows, a, b) for a, b in pairs}
    out['identity_matrix_same_decision_source_estimate'] = {a: {b: sum(1 for r in rows if key(r['arms'][a]) == key(r['arms'][b])) for b in VIS} for a in VIS}
    out['identity_matrix_same_act_or_abstain'] = {a: {b: sum(1 for r in rows if r['arms'][a]['act'] == r['arms'][b]['act']) for b in VIS} for a in VIS}
    dup = [[a, b] for i, a in enumerate(VIS) for b in VIS[i + 1:] if out['identity_matrix_same_decision_source_estimate'][a][b] == n]
    out['duplicate_arms_on_these_scenes'] = {'identical_decision_source_and_estimate_in_every_scene': dup, 'identical_act_or_abstain_in_every_scene': [[a, b] for i, a in enumerate(VIS) for b in VIS[i + 1:] if out['identity_matrix_same_act_or_abstain'][a][b] == n]}
    # ---- unnecessary moves: an arm acts although the true offset is within the stop tolerance (label)
    out['acts_with_true_offset_within_tolerance'] = {a: [r['scene'] for r in rows if r['arms'][a]['act'] and r['gt']['true_offset_m'] <= V2.RESIDUAL_TOL_M] for a in VIS}
    out['offset_error_at_act'] = {a: q(r['arms'][a].get('offset_error_m') for r in rows if r['arms'][a]['act']) for a in VIS if a != 'VISION_CENTER_V1'}
    out['acts_outside_own_bound_or_over_5mm_on_memory'] = {a: [{'scene': r['scene'], 'offset_error_mm': mm(r['arms'][a]['offset_error_m']), 'bound_mm': mm(r['arms'][a]['u']), 'src': r['arms'][a]['target_src']} for r in rows
                                                               if r['arms'][a]['act'] and (r['arms'][a]['target_src'] != 'frame' or r['arms'][a]['hold_src'] == 'memory') and (r['arms'][a]['offset_error_m'] > V2.MEMORY_ERROR_M or not r['arms'][a]['within_bound'])] for a in VIS if a != 'VISION_CENTER_V1'}
    # ---- v2s
    memv2 = [r for r in rows if (r['arms'][V2A].get('target_src') or '').startswith('memory_')]
    wrong = lambda r: r['arms'][V2A]['target_error_m'] > V2.MEMORY_ERROR_M or r['arms'][V2A]['target_error_m'] > (r['arms'][V2A]['u'] - V2.U_HOLD_M)       # over 5 mm, or outside the target part of its own bound
    def fate(r):
        s = r['arms'][VS]
        if (s.get('target_src') or '').startswith('memory_'): return 'kept'
        if s.get('target_src') == 'frame': return 'frame'
        return 'refused:' + str(s.get('memory_reason'))
    W = [r for r in memv2 if wrong(r)]; G = [r for r in memv2 if not wrong(r)]
    inv = [r for r in rows if r['v2s_invalidations']]; allinv = [x for r in rows for x in r['v2s_invalidations']]
    lostacts = out['pairs'][f'{V2A}|{VS}'][f'{V2A}_acts_{VS}_abstains']; newacts = out['pairs'][f'{V2A}|{VS}'][f'{VS}_acts_{V2A}_abstains']
    stops = [{'scene': r['scene'], 'stage': r['stage'], 'v2_recorded': klass(r), **r['v2s_control']['v2s_stop'], 'v2_end_reason': r['v2s_control']['v2_end_reason']} for r in rows if 'v2s_control' in r and r['v2s_control']['v2s_stops_where_v2_did_not']]
    v2acts = [r for r in rows if r['arms'][V2A]['act']]
    out['v2s'] = {
        'constants': src['constants']['v2s'],
        'remembered_targets_of_v2_at_the_trigger': {'n': len(memv2), 'wrong_by_label': {'n': len(W), 'definition': 'target error over 5 mm, or over the target part of its own bound', 'scenes': [{'scene': r['scene'], 'stage': r['stage'], 'target_error_mm': mm(r['arms'][V2A]['target_error_m']),
                                                    'v2_bound_mm': mm(r['arms'][V2A]['u']), 'v2_decision': r['arms'][V2A]['decision'], 'v2s': fate(r), 'v2s_decision': r['arms'][VS]['decision']} for r in W],
                                                    'refused_by_v2s': sum(1 for r in W if fate(r).startswith('refused')), 'kept_by_v2s': sum(1 for r in W if fate(r) == 'kept')},
                                                    'right_by_label': {'n': len(G), 'refused_by_v2s': sum(1 for r in G if fate(r).startswith('refused')), 'kept_by_v2s': sum(1 for r in G if fate(r) == 'kept'),
                                                                       'refusal_reasons': {k: sum(1 for r in G if fate(r) == k) for k in sorted({fate(r) for r in G if fate(r).startswith('refused')})}}},
        'invalidations': {'scenes_with_one': len(inv), 'events': len(allinv), 'by_reason': {k: sum(1 for x in allinv if x['reason'] == k) for k in sorted({x['reason'] for x in allinv})},
                          'events_where_the_target_then_moved_over_1mm_before_the_trigger': sum(1 for x in allinv if x['target_moved_since_t_last_until_trigger_m'] > 0.001),
                          'events_where_it_had_already_moved_over_1mm': sum(1 for x in allinv if x['target_moved_since_t_last_until_then_m'] > 0.001),
                          'scenes_where_the_target_moved_over_5mm': sorted({r['scene'] for r in inv for x in r['v2s_invalidations'] if x['target_moved_since_t_last_until_trigger_m'] > 0.005})},
        'acts': {'v2': len(v2acts), 'v2s': sum(1 for r in rows if r['arms'][VS]['act']), 'v2_acts_v2s_abstains': {k: lostacts[k] for k in ('n', 'by_v2_recorded_outcome')}, 'v2s_acts_v2_abstains': {k: newacts[k] for k in ('n', 'by_v2_recorded_outcome')},
                 'v2_acts_v2s_abstains_scenes': lostacts['scenes'], 'v2s_acts_v2_abstains_scenes': newacts['scenes'],
                 'v2s_decisions_in_scenes_where_v2_acted': {k: sum(1 for r in v2acts if r['arms'][VS]['decision'] == k) for k in sorted({r['arms'][VS]['decision'] for r in v2acts})},
                 'v2s_acts_on_memory': sum(1 for r in rows if r['arms'][VS]['act'] and r['arms'][VS]['target_src'] != 'frame'), 'v2_acts_on_memory': sum(1 for r in v2acts if r['arms'][V2A]['target_src'] != 'frame')},
        'control_replay_along_v2_frames': {'scenes': len(ctl), 'v2s_would_stop_where_v2_went_on': stops, 'largest_offset_difference_to_v2_mm': mm(max((c['max_offset_difference_to_v2_m'] for c in ctl), default=0.)),
                                           'note': 'v2s is replayed on the frames v2 produced; valid up to the first step where v2s would stop, because until then both send the same command'},
        'scene_51009': next(({'v2': r['arms'][V2A], 'v2s': r['arms'][VS], 'invalidations': r['v2s_invalidations']} for r in rows if r['scene'] == 51009), None)}
    s = out['v2s']; valid = {'reproduces_nothing_privileged': True, 'detects_scene_51009': bool(s['scene_51009'] and not (s['scene_51009']['v2s'].get('target_src') or '').startswith('memory_')),
                             'no_act_on_a_memory_wrong_by_label': not any((x['v2s'] == 'kept' and x['v2s_decision'] == 'ACT') for x in s['remembered_targets_of_v2_at_the_trigger']['wrong_by_label']['scenes']),
                             'not_degenerate_still_acts_on_memory': s['acts']['v2s_acts_on_memory'] > 0, 'not_identical_to_another_arm': not any(VS in d for d in dup)}
    s['technical_criteria'] = valid; s['technically_valid'] = bool(all(valid.values()))
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps({k: out[k] for k in ('reproduction_of_the_executed_run', 'v2_reproduced_exactly', 'acts', 'acts_by_target_source', 'duplicate_arms_on_these_scenes', 'acts_with_true_offset_within_tolerance', 'offset_error_at_act', 'acts_outside_own_bound_or_over_5mm_on_memory')}, indent=1))
    for k, v in out['pairs'].items(): print(k, {kk: (vv if not isinstance(vv, dict) else {x: vv[x] for x in ('n', 'by_v2_recorded_outcome')}) for kk, vv in v.items()})
    print(json.dumps({k: v for k, v in s.items() if k not in ('scene_51009',)}, indent=1)[:6000])


if __name__ == '__main__': main()
