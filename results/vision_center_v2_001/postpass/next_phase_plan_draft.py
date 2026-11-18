"""Vision-CENTER v2, Phase C0 (post-pass): (1) what the stored records already say about each v2 component, and
(2) a DRAFT plan for the next phase (C1: generalisation over a second policy seed and component ablation) with its
volume and cost projection. Nothing is run, no scene is consumed, no approval is implied. Refuses to overwrite.

    python results/vision_center_v2_001/postpass/next_phase_plan_draft.py
"""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent; EXE = ROOT / 'results/vision_center_v2_001/execution'
OUT_A = HERE / 'component_attribution_from_logs.json'; OUT_P = HERE / 'V2_C1_PLAN_DRAFT.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cell(r, a): return next((x for x in r['cells'] if x['arm'] == a), None)


def attribution(stage, lo, hi, oracle):
    recs = [json.loads((EXE / stage / 'scenes' / f'{s}.json').read_text(encoding='utf-8')) for s in range(lo, hi + 1)]; rows = []
    for r in recs:
        if not r['triggered']: continue
        v2 = cell(r, 'VISION_CENTER_V2'); v1 = cell(r, 'VISION_CENTER_V1'); c0 = cell(r, 'C0'); rd = v2['control']['readings'][0]; tm = rd.get('memory_target')
        rows.append({'scene': r['scene'], 'y0': c0['y'], 'y1': v1['y'], 'y2': v2['y'], 'yo': cell(r, 'CENTER_ORACLE')['y'] if oracle else None, 'v1_acted': v1['acted'], 'v2_acted': v2['acted'], 'v1_decision': v1['control']['decision'],
                     'v2_decision': v2['control']['decision'], 'target_src': rd['target_src'], 'hold_src': rd['hold_src'], 'age': None if tm is None else r['trigger']['step'] - tm['t_last']})
    acts = [x for x in rows if x['v2_acted']]

    def comp(x):
        if x['target_src'] == 'memory_hand': return 'hand_camera_target_memory'
        if x['target_src'] == 'memory_base': return 'base_camera_target_memory' if x['age'] > 0 else 'base_camera_at_the_trigger_step'
        return 'held_cube_memory' if x['hold_src'] == 'memory' else 'current_frame_only'
    names = ('current_frame_only', 'held_cube_memory', 'hand_camera_target_memory', 'base_camera_target_memory', 'base_camera_at_the_trigger_step'); by = {}
    for n in names:
        a = [x for x in acts if comp(x) == n]
        by[n] = {'v2_acts': len(a), 'v1_also_acted': sum(x['v1_acted'] for x in a), 'v2_rescues': sum(1 for x in a if x['y0'] == 0 and x['y2'] == 1), 'of_which_v1_also_rescued': sum(1 for x in a if x['y0'] == 0 and x['y2'] == 1 and x['y1'] == 1),
                 'v2_harm': sum(1 for x in a if x['y0'] == 1 and x['y2'] == 0), 'scenes_rescued': [x['scene'] for x in a if x['y0'] == 0 and x['y2'] == 1]}
    ab = {}
    for d in sorted({x['v2_decision'] for x in rows if not x['v2_acted']}):
        a = [x for x in rows if x['v2_decision'] == d]
        ab[d] = {'n': len(a), 'v1_acted_here': sum(x['v1_acted'] for x in a), 'v1_rescued_here': sum(1 for x in a if x['y0'] == 0 and x['y1'] == 1), 'v1_harmed_here': sum(1 for x in a if x['y0'] == 1 and x['y1'] == 0),
                 'c0_failures': sum(1 for x in a if x['y0'] == 0), 'oracle_rescues': sum(1 for x in a if x['y0'] == 0 and x['yo'] == 1) if oracle else None}
    v1_only = [x['scene'] for x in rows if x['v1_acted'] and not x['v2_acted']]
    return {'triggered': len(rows), 'v2_acts': len(acts), 'v1_acts': sum(x['v1_acted'] for x in rows), 'v2_acts_by_component': by, 'v2_abstentions': ab, 'v1_acted_where_v2_abstained': {'scenes': v1_only, 'v2_reasons': {x['scene']: x['v2_decision'] for x in rows if x['scene'] in v1_only}},
            'v1_rescues': sum(1 for x in rows if x['y0'] == 0 and x['y1'] == 1), 'v2_rescues': sum(1 for x in rows if x['y0'] == 0 and x['y2'] == 1),
            'reading': 'counts of what the full v2 did, by the source it used. A scene credited to a memory is one the full v2 solved using it; whether an ablated v2 would also have solved it is not known from these records'}


def projection(scene_runs, arms, vision_arms, t, usd_per_hour, spent=0.25, fixed_seconds=600, reserve=0.20, margin=1.25):
    steps = 200 + arms * 200; per = steps * t['env'] + 200 * t['mem'] + vision_arms * 13 * t['read']; hours = (scene_runs * per + fixed_seconds) * margin / 3600
    return {'scene_runs': scene_runs, 'arms_incl_C0': arms, 'vision_arms': vision_arms, 'env_steps_bound_per_scene': steps, 'env_steps_bound_total': scene_runs * steps, 'seconds_per_scene_bound': per, 'hours_with_margin': hours, 'worst_case_usd': spent + hours * usd_per_hour + reserve}


def main():
    for p in (OUT_A, OUT_P):
        if p.exists(): raise SystemExit(f'Refused overwrite: {p}')
    att = {'work_id': 'vision_center_v2_001', 'phase': 'C0 post-pass', 'status': 'from stored records only; development data; not an ablation', 'stage_B2_new_scenes': attribution('pilot', 50900, 51099, True), 'stage_B1_used_scenes_design_data': attribution('shadow', 50600, 50899, False)}
    OUT_A.write_text(json.dumps(att, indent=1), encoding='utf-8', newline='\n')
    audit = json.loads((EXE / 'audit/audit.json').read_text(encoding='utf-8')); res = json.loads((EXE / 'pilot/result.json').read_text(encoding='utf-8')); shd = json.loads((EXE / 'shadow/shadow_result.json').read_text(encoding='utf-8'))
    t = {'env': audit['seconds_per_env_step'], 'read': audit['seconds_per_reading'], 'mem': audit['seconds_per_memory_step']}; price = 0.4604
    measured = {'stage_B2_seconds_per_scene_4_arms': res['wall_seconds'] / 200, 'stage_B1_seconds_per_scene_3_arms': shd['wall_seconds'] / 300, 'host': 'Vast machine 146456, RTX 4090, Ryzen 7 7800X3D, US$0.4604/h, thread limit 1'}
    per_arm = (measured['stage_B2_seconds_per_scene_4_arms'] - measured['stage_B1_seconds_per_scene_3_arms'])                    # one more continuation (the oracle arm) on other scenes: rough
    core = ['C0', 'VISION_CENTER_V1', 'V2_SINGLE_FRAME', 'VISION_CENTER_V2', 'CENTER_ORACLE']; loo = ['V2_NO_HAND_MEMORY', 'V2_NO_BASE_MEMORY', 'V2_NO_HOLD_MEMORY', 'V2_RULES_OFF']

    def expected(arms, runs): return runs * (measured['stage_B2_seconds_per_scene_4_arms'] + (arms - 4) * max(per_arm, 2.0)) / 3600
    tiers = {'tier_1_core': {'arms': core, 'projection_worst_case': projection(400, len(core), 3, t, price), 'expected_rollout_hours': expected(len(core), 400)},
             'tier_1_plus_2': {'arms': core + loo, 'projection_worst_case': projection(400, len(core) + len(loo), 7, t, price), 'expected_rollout_hours': expected(len(core) + len(loo), 400)}}
    for v in tiers.values(): v['expected_total_usd'] = [round(0.30 + v['expected_rollout_hours'] * price + 0.15, 2), round(0.45 + v['expected_rollout_hours'] * price * 1.3 + 0.35, 2)]
    plan = {
        'document': 'V2_C1_PLAN_DRAFT', 'status': 'DRAFT for discussion. NOT approved, NOT executable: the ablation arms and the two-policy driver are not written. Becomes a plan only after the code exists, tests pass, the plan is regenerated by its driver and the user approves its SHA256 with a USD cap.',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'work_id_proposed': 'vision_center_v2_c1_001', 'builds_on': {'work_id': 'vision_center_v2_001', 'plan_sha256': res['plan_sha256'], 'execution_commit': 'fd23aa2', 'freeze_manifest_draft': 'results/vision_center_v2_001/postpass/V2_FREEZE_MANIFEST_DRAFT.json'},
        'questions': ['Q1 generalisation: does the unchanged v2 keep its paired gain and zero harm with a second base policy (Vision-V1 train seed 2) and on new scenes with the first?',
                      'Q2 components: how much of the gain over v1 comes from temporal memory, as opposed to the base-camera estimator used on a single frame and to the changed abstention rules?'],
        'data_role': 'development. Not calibration, not Test. Pre-registered tests make it a planned development replication, not a confirmatory result under the protocol.',
        'policies': {'train_seed_1': {'path': 'artifacts/policies/stack_v1_s1_d500_clean_resnet18_i080000', 'weights.pt': sha(ROOT / 'artifacts/policies/stack_v1_s1_d500_clean_resnet18_i080000/weights.pt'), 'status': 'selected and qualified in Vision-V1; the policy v2 was designed on'},
                     'train_seed_2': {'path': 'artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000', 'weights.pt': sha(ROOT / 'artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000/weights.pt'),
                                      'status': 'trained in Vision-V1 run2 (same data, architecture, 80k EMA); 83/100 on vision_v1_select; not selected, never qualified, never used with any intervention'},
                     'rules': ['no retraining, no fine-tuning, no other checkpoint', 'v2 constants, memory rules and abstention rules are those of the freeze manifest draft for both policies; nothing is adapted to train seed 2',
                               'a third policy seed does not exist; training one is outside this plan']},
        'scenes': {'candidate_range': [51100, 51299], 'n': 200, 'use': 'the same 200 scenes for both policies (paired over arms within a policy; policies compared on equal scenes)', 'seed_level_check': 'PASS in Phase C0 (seed_and_split_audit.json): no overlap with any declared or used set, none named in a stored record',
                   'placement_check': 'first server step; identical initial cube placement against every placement on record fails the step; the set is then not used and not patched silently', 'final': 'the registry file is written and hashed only after the user confirms the range',
                   'untouched': ['calibration 30000-30399', 'test 90000-90399']},
        'arms': {'C0': 'uninterrupted; must reproduce the trunk', 'VISION_CENTER_V1': 'unchanged single-frame baseline (v1 rules, hand camera at the trigger)', 'VISION_CENTER_V2': 'unchanged full v2', 'CENTER_ORACLE': 'unchanged non-deployable reference',
                 'V2_SINGLE_FRAME': 'v2 decision rules and both v2 estimators, but only the observation of the trigger step: hand-camera fit, else the base-camera estimate of that one frame (no confirmation count possible, so confirm_min = 1 for this arm only), held-cube fit of that frame only. No history. '
                                    'This is "v2 without temporal memory": v2 minus this arm = the effect of memory; this arm minus v1 = the effect of the base-camera estimator on one frame plus the changed rules',
                 'V2_NO_HAND_MEMORY': 'tier 2, leave one out: full v2 with remembered hand-camera target fits disabled', 'V2_NO_BASE_MEMORY': 'tier 2, leave one out: full v2 with base-camera observations disabled',
                 'V2_NO_HOLD_MEMORY': 'tier 2, leave one out: full v2 with the remembered held-cube fit disabled',
                 'V2_RULES_OFF': 'tier 2: full v2 memory with the v2-only abstention rules removed (uncertainty margin, side-contact rule, memory height rule; the v1 rules stay). Measures what the rules cost and what they prevent. May harm; development simulation only',
                 'implementation': 'new module and driver that import hci.vision_center_v2 and switch components by arguments; hci.vision_center_v2 itself is not edited (its hash stays ab548ccc...). A test must show that the arm with nothing switched off reproduces VISION_CENTER_V2 decisions on the stored frames of all 500 used scenes'},
        'tier_0_free_local': 'before any paid step: replay every arm\'s DECISION (act / abstain, source, estimate) on the stored frames of the 500 used scenes 50600-51099, CPU only, no simulator. Development only; it shows which arms ever decide differently. An arm that never differs from another on 500 scenes can be dropped from the paid run before the plan is frozen',
        'analysis_pre_registered': {'unit': 'scene, within policy', 'per_policy': ['success of every arm with Wilson interval', 'paired v2 against C0: rescue, harm, net, Tango score interval, exact one-sided paired test', 'paired v2 against v1 and v2 against V2_SINGLE_FRAME: same',
                                                                                    'harm: count, n+, one-sided 95 % Clopper-Pearson upper limit; also among scenes where the arm acted', 'memory errors, target pushed, unnecessary acts, abstentions by reason, oracle recovery'],
                                    'primary_tests': ['H_gen: v2 beats C0 with train seed 2 (exact one-sided paired test, alpha 0.025)', 'H_mem: v2 beats V2_SINGLE_FRAME, both policies pooled by scene pair, stratified by policy (exact one-sided test on the pooled discordant pairs, alpha 0.025)'],
                                    'multiplicity': 'Holm over the two primary tests; everything else descriptive', 'between_policies': ['difference of net gain between the two policies on the same scenes with a scene-bootstrap interval', 'homogeneity of the rescue odds (exact test on the two 2x2 discordant tables)',
                                                                                                                                      'with two policy seeds the between-seed variance has one degree of freedom: reported as a range, not as a variance estimate'],
                                    'safety_stop_criteria_fixed_in_advance': {'harm_max_per_policy': 2, 'memory_errors_max_per_policy': 2, 'target_pushed_over_1mm_max_per_policy': 1, 'note': 'proposed; reported as met / not met, no re-run and no tuning either way'}},
        'volume': {'policies': 2, 'scenes': 200, 'scene_runs': 400, 'audit_scenes_used': 3, **{k: {'arms': v['arms'], **v['projection_worst_case']} for k, v in tiers.items()}},
        'timing_basis': {**t, **measured, 'seconds_per_extra_arm_rough': per_arm},
        'cost': {'usd_per_hour_basis': price, 'tier_1_core': {'expected_rollout_hours': tiers['tier_1_core']['expected_rollout_hours'], 'expected_total_usd': tiers['tier_1_core']['expected_total_usd'], 'worst_case_gate_usd': tiers['tier_1_core']['projection_worst_case']['worst_case_usd']},
                 'tier_1_plus_2': {'expected_rollout_hours': tiers['tier_1_plus_2']['expected_rollout_hours'], 'expected_total_usd': tiers['tier_1_plus_2']['expected_total_usd'], 'worst_case_gate_usd': tiers['tier_1_plus_2']['projection_worst_case']['worst_case_usd']},
                 'proposed_cap_usd': 5.0, 'proposed_cap_is': 'a proposal, not an approval', 'storage_gb': [3.0, 5.0], 'storage_note': 'frames of both cameras up to the trigger for 400 scene runs plus control frames of up to 7 vision arms; the download adds the per-GB transfer price',
                 'not_included': 'setup, install and download are in the expected total (about 25 + 15 minutes); a discarded host as in Phase B would add about US$0.05'},
        'technical_stop_conditions': ['plan hash, code hashes, both policy weight hashes or the seed registry differ from the plan', 'tests fail on the server', 'CUDA / Vulkan check fails (host with driver below the CUDA 12.4 requirement: discard the host, no scene run)',
                                      'an identical initial placement is found for a new seed', 'the restored runtime differs from the snapshot, C0 or an abstaining arm does not reproduce the trunk', 'ground truth reaches a vision arm (oracle read counter)',
                                      'the full-v2 arm of the new driver differs from hci.vision_center_v2 on the audit scenes', 'the budget projection exceeds the cap before a stage or after its first five scenes: the stage does not start or stops; scenes and arms are never reduced',
                                      'a ledger for the set already exists (no resume, no second pass)', 'any error in a scene: the pass stays INCOMPLETE, no scene dropped or replaced, no summary of a partial run'],
        'not_in_this_plan': ['policy training or fine-tuning', 'any change of v2 constants or rules', 'the proposed safety variant v2s (a separate candidate with its own plan)', 'calibration', 'final Test', 'peg insertion or any other task'],
        'needed_from_the_user': ['confirm scenes 51100-51299', 'tier 1 only, or tier 1 + 2', 'confirm the two primary tests and the safety criteria', 'a USD cap and approval of the SHA256 of the plan the driver will generate (not of this draft)']}
    OUT_P.write_text(json.dumps(plan, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(att['stage_B2_new_scenes'], indent=None)[:3000]); print(json.dumps(att['stage_B1_used_scenes_design_data']['v2_acts_by_component'])[:1500]); print(json.dumps(plan['timing_basis'])); print(json.dumps(plan['cost'], indent=1)); print(json.dumps(plan['volume'])[:900])


if __name__ == '__main__': main()
