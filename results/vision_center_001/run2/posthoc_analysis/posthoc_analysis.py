"""Post hoc description of the Vision-CENTER run2 records (written after the outcomes were known; descriptive only).
Ground truth is used here as a label. Nothing in this file was used to choose a constant, and nothing is re-run.
Usage, from the repository root: python -X utf8 results/vision_center_001/run2/posthoc_analysis/posthoc_analysis.py"""
import json, csv
from pathlib import Path
import numpy as np
R = Path('results/vision_center_001/run2'); OUT = R / 'posthoc_analysis'


def rd(p):
    return json.load(open(p, encoding='utf-8'))


def rows(stage, seeds):
    out = []
    for s in seeds:
        r = rd(R / stage / 'scenes' / f'{s}.json'); cells = {c['arm']: c for c in r['cells']}
        row = {'scene': s, 'triggered': r['triggered'], **{f'y_{a}': cells[a]['y'] for a in cells}, **{f'reason_{a}': cells[a]['reason'] for a in cells}}
        if r['triggered']:
            g = r['trigger']['gt_for_diagnosis_only']; v = cells['VISION_CENTER']; c = v['control']; e = v['gt_reading_error'] or {}; first = c['readings'][0]; tf = first['target_fit']; hf = first['hold_fit']
            row.update(trigger_step=r['trigger']['step'], steps_left=200 - r['trigger']['step'], true_offset_m=g['offset_xy_m'], cubeA_z=g['cubeA'][2], decision=c['decision'], est_offset_m=e.get('estimated_offset_m'),
                       offset_error_m=e.get('offset_error_m'), target_error_m=e.get('target_error_m'), target_pixels=None if tf is None else tf['pixels'], target_score=None if tf is None else tf['score'],
                       target_margin=None if tf is None else tf['margin'], hold_valid=None if hf is None else hf['valid'], est_held_z=None if first['held_xyz'] is None else first['held_xyz'][2], acted=v['acted'],
                       control_steps=v['control_steps'], end_reason=c['end_reason'], release_offset_C0=(cells['C0']['gt_release_after_trigger'] or {}).get('offset_xy_m'),
                       release_offset_VC=(v['gt_release_after_trigger'] or {}).get('offset_xy_m'))
            if v['acted']:
                ef = v['gt_effect_of_control']
                row.update(offset_after_control=ef['offset_xy_after_m'], target_pushed_m=ef['cubeB_xy_displacement_m'], tcp_moved_m=ef['tcp_xy_displacement_m'], grasped_after=ef['grasped_after'],
                           first_inference_step=v['first_inference_step'], steps_VC=v['steps'])
            if 'CENTER_ORACLE' in cells:
                o = cells['CENTER_ORACLE']
                row.update(oracle_steps=o['control_steps'], oracle_end=o['control']['end_reason'], release_offset_ORACLE=(o['gt_release_after_trigger'] or {}).get('offset_xy_m'))
        out.append(row)
    return out


def table(rs, key):
    d = {}
    for r in rs: d.setdefault(r.get(key), []).append(r)
    return d


def describe(rs, pilot):
    t = [r for r in rs if r['triggered']]
    res = {'scenes': len(rs), 'triggered': len(t), 'no_trigger': [r['scene'] for r in rs if not r['triggered']], 'no_trigger_C0_failures': [r['scene'] for r in rs if not r['triggered'] and not r['y_C0']], 'by_decision': {}}
    for k, v in table(t, 'decision').items():
        off = [x['true_offset_m'] for x in v]
        d = {'n': len(v), 'C0_success': sum(x['y_C0'] for x in v), 'C0_failure': sum(1 - x['y_C0'] for x in v), 'true_offset_median_m': float(np.median(off)), 'true_offset_max_m': float(max(off)),
             'true_offset_ge_15mm': sum(o >= .015 for o in off), 'true_offset_ge_20mm': sum(o >= .02 for o in off), 'C0_failure_scenes': [x['scene'] for x in v if not x['y_C0']]}
        if pilot: d['oracle_rescues_here'] = [x['scene'] for x in v if not x['y_C0'] and x['y_CENTER_ORACLE']]
        res['by_decision'][k] = d
    keys = ('scene', 'triggered', 'decision', 'true_offset_m', 'est_offset_m', 'cubeA_z', 'est_held_z', 'steps_left', 'target_pixels', 'target_score', 'target_margin', 'hold_valid', 'y_VISION_CENTER', 'y_CENTER_ORACLE',
            'reason_C0', 'reason_VISION_CENTER', 'release_offset_C0', 'release_offset_VC')
    res['C0_failures'] = [{k: r.get(k) for k in keys} for r in rs if not r['y_C0']]
    acts = [r for r in t if r['acted']]; rel = [r['release_offset_VC'] for r in acts if r['release_offset_VC'] is not None]
    res['acts'] = {'n': len(acts), 'C0_success_in_these_scenes': sum(r['y_C0'] for r in acts), 'C0_failure_in_these_scenes': sum(1 - r['y_C0'] for r in acts),
                   'rescued': [r['scene'] for r in acts if not r['y_C0'] and r['y_VISION_CENTER']], 'harmed': [r['scene'] for r in acts if r['y_C0'] and not r['y_VISION_CENTER']],
                   'acted_but_still_failed': [r['scene'] for r in acts if not r['y_C0'] and not r['y_VISION_CENTER']], 'true_offset_at_act_m': sorted(round(r['true_offset_m'], 4) for r in acts),
                   'acts_with_true_offset_below_15mm': sum(r['true_offset_m'] < .015 for r in acts), 'acts_with_true_offset_at_most_10mm': sum(r['true_offset_m'] <= .010 for r in acts),
                   'offset_error_at_act_max_m': max(r['offset_error_m'] for r in acts), 'min_true_height_of_held_cube_at_act_m': min(r['cubeA_z'] for r in acts),
                   'acts_with_true_height_below_0.058': [(r['scene'], round(r['cubeA_z'], 4), r['y_C0'], r['y_VISION_CENTER']) for r in acts if r['cubeA_z'] < .058],
                   'target_pushed_more_than_1mm': [(r['scene'], round(r['target_pushed_m'], 4), r['y_C0'], r['y_VISION_CENTER']) for r in acts if r['target_pushed_m'] > .001],
                   'end_reasons': {k: len(v) for k, v in table(acts, 'end_reason').items()}, 'lost_grasp_during_control': [r['scene'] for r in acts if not r['grasped_after']],
                   'release_offset_after_act_m': {'median': float(np.median(rel)), 'max': float(max(rel)), 'no_release_found': [r['scene'] for r in acts if r['release_offset_VC'] is None]},
                   'policy_after_control': {'released_more_than_5mm_further_off_than_at_the_end_of_control': [r['scene'] for r in acts if r['release_offset_VC'] is not None and r['release_offset_VC'] > r['offset_after_control'] + .005],
                                            'steps_from_end_of_control_to_end_of_episode_median': float(np.median([r['steps_VC'] - r['first_inference_step'] for r in acts]))}}
    est = [r for r in t if r.get('offset_error_m') is not None]
    res['estimation'] = {'n_with_estimate': len(est), 'offset_error_median_m': float(np.median([r['offset_error_m'] for r in est])), 'offset_error_max_m': float(max(r['offset_error_m'] for r in est)),
                         'abstained_ALIGNED_with_true_offset_ge_15mm': [(r['scene'], round(r['true_offset_m'], 4), round(r['est_offset_m'], 4), r['y_C0']) for r in t if r['decision'] == 'ALIGNED' and r['true_offset_m'] >= .015]}
    return res


man = rd('configs/recovery_v1_splits.json')['splits']; diag_seeds = list(man['recovery_v1_diag']); orc_seeds = list(man['recovery_v1_pilot_reserved']); pi_seeds = list(range(50800, 50900))
sh = rows('shadow', diag_seeds + orc_seeds); pi = rows('pilot', pi_seeds)
res = {'note': 'Post hoc and descriptive. Ground truth is used as a label only.', 'B1_shadow_used_scenes': describe(sh, False), 'B2_pilot_new_scenes': describe(pi, True)}
orc = [r for r in pi if not r['y_C0'] and r['y_CENTER_ORACLE']]
keys = ('scene', 'decision', 'true_offset_m', 'est_offset_m', 'target_pixels', 'target_score', 'target_margin', 'hold_valid', 'cubeA_z', 'est_held_z', 'steps_left', 'y_VISION_CENTER', 'oracle_steps', 'release_offset_C0', 'release_offset_ORACLE')
res['B2_oracle_rescues'] = [{k: r.get(k) for k in keys} for r in orc]
res['B2_oracle_rescues_missed_by_vision_by_reason'] = {k: [x['scene'] for x in v] for k, v in table([r for r in orc if not r['y_VISION_CENTER']], 'decision').items()}

# the same scenes on earlier hosts
diag = {}
for line in open('results/recovery_v1_diagnostic/run1/episodes.jsonl', encoding='utf-8'):
    e = json.loads(line); diag[int(e['seed'])] = int(e['success_once'])
orc_c0 = {}; orc_y = {}; orc_trig = {}
for s in orc_seeds:
    o = rd(f'results/recovery_v1_oracle_xy_pilot_001/run1/scenes/{s}.json'); cc = {c['arm']: c['y'] for c in o['cells']}; orc_c0[s] = cc['C0']; orc_y[s] = cc['CENTER_ORACLE']; orc_trig[s] = o['trigger']['step'] if o['triggered'] else None
shd = {r['scene']: r for r in sh}
old_resc = [s for s in orc_seeds if not orc_c0[s] and orc_y[s]]
res['reproducibility'] = {
    'C0_now_vs_diagnostic_run_50600_50699': {'earlier_success': sum(diag[s] for s in diag_seeds), 'now_success': sum(shd[s]['y_C0'] for s in diag_seeds), 'scenes_with_different_outcome': [s for s in diag_seeds if diag[s] != shd[s]['y_C0']]},
    'C0_now_vs_oracle_pilot_50700_50799': {'earlier_success': sum(orc_c0.values()), 'now_success': sum(shd[s]['y_C0'] for s in orc_seeds), 'scenes_with_different_outcome': [s for s in orc_seeds if orc_c0[s] != shd[s]['y_C0']],
                                           'scenes_with_different_trigger_step': sum(1 for s in orc_seeds if orc_trig[s] != shd[s].get('trigger_step'))},
    'within_run': 'in every scene the C0 arm, and the Vision-CENTER arm whenever it abstains, reproduce the uninterrupted run exactly (checked by the harness; a mismatch stops the run)'}
res['B1_against_the_oracle_pilot_on_50700_50799 (different hosts; descriptive)'] = {
    'oracle_rescues_in_the_earlier_pilot': old_resc, 'what_vision_did_in_those_scenes_now': [(s, shd[s].get('decision'), shd[s]['y_C0'], shd[s]['y_VISION_CENTER']) for s in old_resc]}
a1 = {s: rd(f'results/vision_center_001/run1_budget_blocked/audit/scenes/{s}.json') for s in (50400, 50401, 50402)}; a2 = {s: rd(R / 'audit' / 'scenes' / f'{s}.json') for s in a1}


def vc(r):
    return [c for c in r['cells'] if c['arm'] == 'VISION_CENTER'][0]


res['reproducibility']['audit_run1_host_vs_run2_host'] = [
    {'scene': s, 'trigger_step': [a1[s]['trigger']['step'], a2[s]['trigger']['step']], 'decision': [vc(a1[s])['control']['decision'], vc(a2[s])['control']['decision']],
     'estimated_offset_m': [vc(a1[s])['gt_reading_error']['estimated_offset_m'], vc(a2[s])['gt_reading_error']['estimated_offset_m']],
     'true_offset_m': [vc(a1[s])['gt_reading_error']['true_offset_m'], vc(a2[s])['gt_reading_error']['true_offset_m']]} for s in a1]
json.dump(res, open(OUT / 'posthoc_analysis.json', 'w', encoding='utf-8', newline='\n'), indent=1)
for name, rs in (('per_scene_B1_shadow.csv', sh), ('per_scene_B2_pilot.csv', pi)):
    ks = []
    for r in rs: ks += [k for k in r if k not in ks]
    with open(OUT / name, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ks); w.writeheader(); w.writerows(rs)
print(json.dumps(res, indent=1))
