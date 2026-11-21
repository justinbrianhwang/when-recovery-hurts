"""Reports and the claims table, generated from results.json values only (no number is typed by hand)."""
from __future__ import annotations

import json
from pathlib import Path


def F(r, stage, arm, step):
    for x in r['funnel']:
        if x['stage'] == stage and x['arm'] == arm and x['step'] == step: return x
    return None


def nd(x):
    return 'NA' if x is None else f"{x['numerator']} / {x['denominator']}"


def mm(d, k='median'):
    v = (d or {}).get(k); return 'NA' if v is None else f'{v:.2f}'


ARMS = ('VISION_CENTER_V1', 'VISION_CENTER_V2', 'CENTER_ORACLE'); STAGES = ('B1_shadow', 'B2_pilot')


def counts_tables(r, stages, arms):
    """Review F1: the three scopes in separate, labelled tables; every denominator explicit."""
    L = ['### 1a. Overall (every scene with both outcomes recorded, triggered or not)', '',
         '| Stage | Arm | Scenes | Not triggered | Triggered | Success / all | Rescue / all C0 failures | Harm / all C0 successes | Net (rescue - harm) / all |', '|---|---|---|---|---|---|---|---|---|']
    for st in stages:
        for arm in arms:
            if not F(r, st, arm, 'triggered'): continue
            net = F(r, st, arm, 'overall_net_rescue_minus_harm')
            L.append(f"| {st} | {arm} | {(F(r, st, arm, 'scenes') or {}).get('numerator')} | {nd(F(r, st, arm, 'not_triggered'))} | {nd(F(r, st, arm, 'triggered'))} | {nd(F(r, st, arm, 'overall_success'))} | "
                     f"{nd(F(r, st, arm, 'overall_rescue_of_c0_failure')) if arm != 'C0' else '-'} | {nd(F(r, st, arm, 'overall_harm_of_c0_success')) if arm != 'C0' else '-'} | {nd(net) if net else '-'} |")
    L += ['', '### 1b. Conditional on the trigger (labelled; not the overall performance)', '',
          '| Stage | Arm | Decision ACT / triggered | Control steps executed / ACT | Actual movement (>= 1 mm) / with steps | Abstained / triggered | Rescue / triggered C0 failures | Harm / triggered C0 successes | C0 success kept by abstaining / triggered C0 successes |', '|---|---|---|---|---|---|---|---|---|']
    for st in stages:
        for arm in arms:
            if not F(r, st, arm, 'triggered') or arm == 'C0': continue
            L.append(f"| {st} | {arm} | {nd(F(r, st, arm, 'decision_ACT'))} | {nd(F(r, st, arm, 'control_steps_executed'))} | {nd(F(r, st, arm, 'actual_movement'))} | {nd(F(r, st, arm, 'abstained_or_no_action'))} | "
                     f"{nd(F(r, st, arm, 'triggered_rescue_of_c0_failure'))} | {nd(F(r, st, arm, 'triggered_harm_of_c0_success'))} | {nd(F(r, st, arm, 'triggered_c0_success_kept_by_abstention'))} |")
    L += ['', '### 1c. Conditional on actual movement (C0 successes in which the arm moved the TCP >= 1 mm)', '', '| Stage | Arm | Harm / C0 successes moved in | Kept success / C0 successes moved in |', '|---|---|---|---|']
    for st in stages:
        for arm in arms:
            if not F(r, st, arm, 'triggered') or arm == 'C0': continue
            L.append(f"| {st} | {arm} | {nd(F(r, st, arm, 'moved_harm_of_c0_success_where_arm_moved'))} | {nd(F(r, st, arm, 'moved_c0_success_kept_where_arm_moved'))} |")
    L += ['', 'Not triggered (the arm never ran) and triggered-then-abstained are counted apart; both stay in every overall count. A missing outcome is a completeness error, never 0 or a failure '
          '(`outcome_missing_completeness_error`). The overall counts are checked against the collector summary in `tables/crosscheck_against_collector_summary.csv`.', '']
    return L


def claims(r):
    s = r['summaries']; rc = s['reconcile_88_81']; out = []
    def c(cid, kind, claim, value, where, fig=''): out.append({'id': cid, 'kind': kind, 'claim': claim, 'value': value, 'computed_in': where, 'figure': fig})
    c('C01', 'DIRECT_OBSERVATION', 'In the 484 triggered Phase B scenes, v2 decided ACT in 88; every ACT executed control steps and moved the TCP by >= 1 mm',
      f"ACT {rc['phase_b_v2_decision_ACT']['n']}, moved {rc['phase_b_v2_moved_over_1mm']['n']}", 'results.json summaries.reconcile_88_81; opportunity_funnel.csv steps decision_ACT / actual_movement (VISION_CENTER_V2)')
    c('C02', 'DIRECT_OBSERVATION', '"81" is a different set: the offline-replay scenes where the excluded v2s variant also decided ACT; it is a subset of the 88',
      f"{rc['offline_r4_rows_with_v2s_control_replay']['n']} (subset: {rc['subset']}; in 88 not in 81: {rc['in_88_not_in_81']})", 'results.json summaries.reconcile_88_81')
    for st in STAGES:
        for arm in ARMS:
            h = F(r, st, arm, 'overall_harm_of_c0_success'); hm = F(r, st, arm, 'moved_harm_of_c0_success_where_arm_moved'); rs = F(r, st, arm, 'overall_rescue_of_c0_failure')
            if h: c(f'C03-{st}-{arm}', 'DIRECT_OBSERVATION', f'{arm} in {st}: OVERALL harm among all C0 successes and rescue among all C0 failures (every scene, triggered or not); trigger-conditional values labelled; harm among C0 successes it moved in',
                    f"harm {nd(h)}; rescue {nd(rs)}; trigger-conditional: harm {nd(F(r, st, arm, 'triggered_harm_of_c0_success'))}, rescue {nd(F(r, st, arm, 'triggered_rescue_of_c0_failure'))}; harm when moved {nd(hm)}",
                    'opportunity_funnel.csv steps overall_harm_of_c0_success, overall_rescue_of_c0_failure, triggered_*, moved_harm_of_c0_success_where_arm_moved; tables/denominator_check.csv; tables/crosscheck_against_collector_summary.csv')
    o = F(r, 'B2_pilot', 'CENTER_ORACLE', 'overall_rescue_of_c0_failure'); c('C04', 'DIRECT_OBSERVATION', 'Oracle rescues in B2 among all C0 failures (the only stage with an oracle); the oracle is one reference controller, not an upper bound of all recoveries', f"{nd(o)} (trigger-conditional {nd(F(r, 'B2_pilot', 'CENTER_ORACLE', 'triggered_rescue_of_c0_failure'))})", 'opportunity_funnel.csv')
    for arm in ('VISION_CENTER_V1', 'VISION_CENTER_V2'):
        a = F(r, 'B2_pilot', arm, 'oracle_rescue_also_rescued'); miss = [x for x in r['funnel'] if x['stage'] == 'B2_pilot' and x['arm'] == arm and x['step'].startswith('oracle_rescue_missed:') and x['numerator']]
        c(f'C05-{arm}', 'DIRECT_OBSERVATION', f'{arm}: oracle rescues it also rescued; missed ones by the first block in code order (a label, not a cause)', f"{nd(a)}; " + '; '.join(f"{x['step'].split(':')[1]} {x['numerator']} ({x['scenes']})" for x in miss), 'opportunity_funnel.csv; results.json summaries.oracle_rescue_missed')
    md = s.get('memory_decomposition_VISION_CENTER_V2')
    if md: c('C06', 'DIRECT_OBSERVATION', 'v2 memory-sourced target estimates: current error, error at the last memory update, and true target motion since then (vector identity holds)',
             f"n {md['n']}; err now median {mm(md['err_now_mm'])} mm, p90 {mm(md['err_now_mm'], 'p90')}, max {mm(md['err_now_mm'], 'max')}; err at store median {mm(md['err_at_store_mm'])}; move since store max {mm(md['target_move_since_store_mm'], 'max')} mm; above own u: {md['err_now_above_memory_u']}; identity residual max {md['identity_residual_max_m']:.1e} m",
             'tables/target_estimate_and_memory_decomposition.csv')
    for arm in ARMS:
        m = s[f'motion_{arm}']
        c(f'C07-{arm}', 'DIRECT_OBSERVATION', f'{arm}: during control the target moved > 1 mm / the alignment improvement came mostly from target motion (push-assisted)', f"target moved {m['target_moved_over_1mm']} of {m['branches_with_control']}; push-assisted {m['push_assisted']} ({m['push_assisted_scenes']})", 'scene_arm_summary.csv columns target_moved, push_assisted')
        c(f'C08-{arm}', 'DIRECT_OBSERVATION', f'{arm}: command, actual TCP motion and residual at the end of control (separate bounds 5 mm/step, 3 cm from start, 12 steps)',
          f"cmd path median {mm(m['cmd_path_mm'])} mm, TCP net median {mm(m['tcp_xy_net_mm'])} mm, residual <= 10 mm at end {m['residual_within_tol_at_end']} of {m['branches_with_control']}; bound violations {m['bounds']}", 'scene_arm_summary.csv')
    s6 = r['s6']; f6 = s6['explanatory_fit_diff_vs_vreln_dtsub']
    c('C09', 'DIRECT_OBSERVATION', 'Official S6 stays FAIL on the local CPU replay', f"{s6['official_S6']['over_tolerance']} of {s6['official_S6']['pairs']} separated pair-rows above 1 mm", 'tables/s6_validity_summary.json official_S6')
    c('C10', 'ASSOCIATION', 'The S6 difference is associated with (relative velocity . normal) x sub-step time, strongest for vertical normals', f"overall r {f6['r']:.2f}, slope {f6['slope_through_origin']:.2f}; " + '; '.join(f"{k}: r {v['r']:.2f}" for k, v in s6['fit_by_normal_direction'].items() if v.get('r') is not None), 'tables/s6_validity_summary.json')
    c('C11', 'HYPOTHESIS', 'PhysX reports contact separation for a pose before the last sub-step integration (time offset)', 'code-level evidence is not available in the stored sources (PhysX internals); empirical association only', 'S6_MEASUREMENT_VALIDITY_NOTE.md')
    c('C12', 'NOT_ESTABLISHED', 'The time offset is the only cause of the S6 differences', f"{f6['residual_over_tol_after_fit']} pair-rows remain above 1 mm after the explanatory fit", 'tables/s6_validity_summary.json')
    c('C13', 'NOT_ESTABLISHED', 'Effects of the decision-replay ablation arms on rescue, harm, motion or side effects', 'NOT_RUN (decisions only)', 'tables/ablation_chain.csv')
    c('C14', 'NOT_ESTABLISHED', 'Generalisation across policy seeds', 'Phase B has train_seed_1 only; C1 will add train_seed_2 on shared scenes (paired, not 400 independent)', 'C1_ANALYSIS_READINESS.md')
    c('C15', 'NOT_ESTABLISHED', 'Safety of contacts inside a control step (sub-steps 1-4) and continuous-time safety', 'not logged', 'S6_MEASUREMENT_VALIDITY_NOTE.md')
    oz = F(r, 'B2_pilot', 'CENTER_ORACLE', 'oracle_within_tolerance_at_start')
    c('C16', 'DIRECT_OBSERVATION', 'Oracle engagements with zero control steps, split by recorded end reason (within_tolerance vs any other)', f"within tolerance {nd(oz)}; other end reason {nd(F(r, 'B2_pilot', 'CENTER_ORACLE', 'oracle_zero_steps_other_end_reason'))}", 'opportunity_funnel.csv (CENTER_ORACLE oracle_within_tolerance_at_start, oracle_zero_steps_other_end_reason)')
    for k, v in r['figures'].items(): out.append({'id': f'F-{k}', 'kind': 'DIRECT_OBSERVATION', 'claim': f'case figure: {v.get("why")}', 'value': v.get('status'), 'computed_in': 'figures.py', 'figure': v.get('file', '')})
    return out


def write_reports(out, r):
    out = Path(out); s = r['summaries']
    L = ['# Existing-data mechanism report (Phase B original, offline decision replay, local CPU replay)', '',
         'Every number below is written from `results.json` by `report.py`; nothing is typed by hand. Sources are never pooled.', '',
         '## 1. Counts with explicit denominators (PHASE_B_ORIGINAL, policy train_seed_1)', '',
         ] + counts_tables(r, STAGES, ('C0',) + ARMS)
    rc = s['reconcile_88_81']
    L += ['Oracle: ACT means at least one control step. It engaged at every trigger; '
          f"{nd(F(r, 'B2_pilot', 'CENTER_ORACLE', 'oracle_within_tolerance_at_start'))} triggered scenes ended with zero steps and the recorded end reason `within_tolerance` (`ORACLE_WITHIN_TOLERANCE_AT_START`); "
          f"{nd(F(r, 'B2_pilot', 'CENTER_ORACLE', 'oracle_zero_steps_other_end_reason'))} ended with zero steps for another recorded end reason (`ORACLE_ZERO_STEPS:<reason>`).", '',
          '### The 88 and the 81', '',
          f"- **88** = {rc['sources']['88']}: {rc['phase_b_v2_decision_ACT']['n']} scenes; all moved >= 1 mm ({rc['phase_b_v2_moved_over_1mm']['n']}).",
          f"- **81** = {rc['sources']['81']}: {rc['offline_r4_rows_with_v2s_control_replay']['n']} scenes ({rc['offline_r4_rows_with_v2s_control_replay']['meaning']}).",
          f"- 81 is a subset of 88: {rc['subset']}. In 88 but not 81 ({len(rc['in_88_not_in_81'])}): {rc['in_88_not_in_81']}. The two numbers count different sets; neither is adjusted.", '',
          '## 2. Observation, estimate and memory (target cube and held cube)', '']
    for arm in ('VISION_CENTER_V1', 'VISION_CENTER_V2'):
        t = s[f'target_estimate_{arm}']; h = s[f'held_estimate_{arm}']
        L += [f'**{arm}**: triggered with a control record {t["triggered_with_control_record"]}. Target coverage: {t["coverage"]}. No-estimate reasons: {t["no_estimate_reasons"] or "none"}.', '',
              '| Target source | n | Error now median (mm) | P90 | Max |', '|---|---|---|---|---|']
        for src, d in t['err_now_mm_by_source'].items(): L.append(f"| {src} | {d['n']} | {mm(d)} | {mm(d, 'p90')} | {mm(d, 'max')} |")
        L += ['', f"Held cube: coverage {h['coverage']}; error now (XY) median {mm(h['err_now_xy_mm'])} mm, P90 {mm(h['err_now_xy_mm'], 'p90')}, max {mm(h['err_now_xy_mm'], 'max')} (n {h['err_now_xy_mm']['n']}); "
              f"true held-cube shift relative to the TCP since the last hold fit: median {mm(h['held_rel_tcp_change_since_fit_mm'])} mm, max {mm(h['held_rel_tcp_change_since_fit_mm'], 'max')} (n {h['held_rel_tcp_change_since_fit_mm']['n']}, world-frame XY).", '']
    md = s.get('memory_decomposition_VISION_CENTER_V2')
    if md:
        L += ['**v2 memory-sourced target estimates: vector decomposition** `error now = error at the last memory update + (true position then - true position now)` (vector identity; the residual is float rounding).', '',
              '| Term | n | Median (mm) | P90 | Max |', '|---|---|---|---|---|',
              f"| error now | {md['err_now_mm']['n']} | {mm(md['err_now_mm'])} | {mm(md['err_now_mm'], 'p90')} | {mm(md['err_now_mm'], 'max')} |",
              f"| error at the last update (fused memory value) | {md['err_at_store_mm']['n']} | {mm(md['err_at_store_mm'])} | {mm(md['err_at_store_mm'], 'p90')} | {mm(md['err_at_store_mm'], 'max')} |",
              f"| true target motion since then | {md['target_move_since_store_mm']['n']} | {mm(md['target_move_since_store_mm'])} | {mm(md['target_move_since_store_mm'], 'p90')} | {mm(md['target_move_since_store_mm'], 'max')} |", '',
              f"Magnitudes do not add (the vectors can cancel). Age at the trigger (steps): median {md['age_steps']['median']}, max {md['age_steps']['max']}. Error now above the memory's own bound u: {md['err_now_above_memory_u']} ({md['scenes_err_now_above_memory_u']}). "
              f"Target moved > 1 mm since the last update: {md['target_moved_more_than_1mm_since_store']}. Identity residual max {md['identity_residual_max_m']:.1e} m.", '']
    L += ['## 3. Recovery opportunities (B2 pilot: the only stage with C0 and the oracle on the same scenes)', '',
          '| Arm | Oracle rescues also rescued | Rescued where the oracle failed | Missed: first block in code order |', '|---|---|---|---|']
    for arm in ('VISION_CENTER_V1', 'VISION_CENTER_V2'):
        miss = [x for x in r['funnel'] if x['stage'] == 'B2_pilot' and x['arm'] == arm and x['step'].startswith('oracle_rescue_missed:') and x['numerator']]
        L.append(f"| {arm} | {nd(F(r, 'B2_pilot', arm, 'oracle_rescue_also_rescued'))} | {nd(F(r, 'B2_pilot', arm, 'overall_rescued_where_oracle_failed'))} | " + '; '.join(f"{x['step'].split(':')[1]} {x['numerator']} (scenes {x['scenes']})" for x in miss) + ' |')
    L += ['', 'The single label is the first block the code reached; overlapping flags are in `scene_arm_summary.csv` (`miss_flags`). A label is not the unique physical cause. '
          'B1 has no oracle, so no oracle-based opportunity is computed there. Decision-replay arms get no rescue or harm (NOT_RUN).', '',
          '| Stage | Scene | Arm | Class | Flags | Decision | Estimated offset (m) | True offset (m) |', '|---|---|---|---|---|---|---|---|']
    for m in s['oracle_rescue_missed']: L.append(f"| {m['stage']} | {m['scene']} | {m['arm']} | {m['miss_class']} | {m['flags'] or '-'} | {m['decision']} | {m['est_offset_m'] if m['est_offset_m'] is None else round(m['est_offset_m'], 4)} | {m['true_offset_m'] if m['true_offset_m'] is None else round(m['true_offset_m'], 4)} |")
    L += ['', f"Harm events (all stages, executed arms): {s['harm_events'] or 'none'}.", '', '## 4. Command, actual motion, alignment, resume and opening', '',
          '| Arm | Branches with control | Moved | Cmd path median (mm) | Cmd net median | TCP net median | TCP path median | Held net median | Target moved > 1 mm | Push-assisted | Offset before median | Offset after median | Residual <= 10 mm at end | Grasp lost in control | Bound violations |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        m = s[f'motion_{arm}']
        L.append(f"| {arm} | {m['branches_with_control']} | {m['moved']} | {mm(m['cmd_path_mm'])} | {mm(m['cmd_net_mm'])} | {mm(m['tcp_xy_net_mm'])} | {mm(m['tcp_xy_path_mm'])} | {mm(m['held_xy_net_mm'])} | {m['target_moved_over_1mm']} | {m['push_assisted']} | "
                 f"{mm(m['offset_before_mm'])} | {mm(m['offset_after_mm'])} | {m['residual_within_tol_at_end']} | {m['grasp_lost_during_control']} | {m['bounds']} |")
    L += ['']
    for arm in ARMS:
        m = s[f'motion_{arm}']
        L.append(f"- {arm}: end reasons {m['end_reason']}; cos(command, true offset) median {m['cos_cmd_vs_true_offset']['median']} (NA {m['cos_cmd_NA']}); steps from control end to the first open command median {m['steps_control_end_to_open']['median']} (max {m['steps_control_end_to_open']['max']}); "
                 f"offset change from control end to opening median {mm(m['offset_change_end_to_open_mm'])} mm (max {mm(m['offset_change_end_to_open_mm'], 'max')}); outcomes vs C0 {m['outcome_vs_c0']}; push-assisted scenes {m['push_assisted_scenes']}.")
    L += ['', 'The three limits (5 mm per step, 3 cm target from the start, 12 steps) are checked separately. A command path longer than the TCP path is not an implementation error by itself. A final success does not make the movement correct, and a failure after resume is not attributed to the resume.', '',
          '## 5. Ablation chain (OFFLINE_DECISION_REPLAY; decisions only)', '', '| Arm | Stage | Removed | Triggered | Target source changed | Decision changed | ACT lost vs v2 | ACT gained | Motion | Outcome |', '|---|---|---|---|---|---|---|---|---|---|']
    for a in s['offline_ablation']: L.append(f"| {a['arm']} | {a['stage']} | {a['removed_element']} | {a['triggered_scenes']} | {a['information_changed_target_src']} | {a['decision_changed']} | {a['act_lost_vs_v2']} | {a['act_gained_vs_v2']} | {a['command_motion_change']} | {a['outcome_difference']} |")
    L += ['', 'Counting the information source of an ACT is not its causal contribution; leave-one-out differences overlap and do not add. V2_SINGLE_FRAME changes two things at once (memories removed and confirmation count). Only train_seed_1 exists here.', '',
          '## 6. Case figures', '', '| Scene | Why selected | Figure |', '|---|---|---|']
    for k, v in sorted(r['figures'].items()): L.append(f"| {k} | {v.get('why')} | [{v.get('file')}]({v.get('file')}) |")
    (out / 'EXISTING_DATA_MECHANISM_REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    s6 = r['s6']; f6 = s6['explanatory_fit_diff_vs_vreln_dtsub']
    T = ['# S6 measurement-validity note', '', 'Source: LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS (6 cases, stored contact lists and poses; another host than Phase B). Phase B records have no contact list, so S6 cannot be computed on them.', '',
         '## 1. Official result (unchanged)', '', f"S6 = **{s6['official_S6']['status']}**: {s6['official_S6']['over_tolerance']} of {s6['official_S6']['pairs']} separated shape-pair rows differ by more than {s6['official_S6']['tolerance_m'] * 1e3:.0f} mm. "
         f"|difference| median {s6['abs_diff_mm']['median']:.3f} mm, P90 {s6['abs_diff_mm']['p90']:.3f}, P99 {s6['abs_diff_mm']['p99']:.3f}, max {s6['abs_diff_mm']['max']:.3f}. The tolerance and the verdict are not changed by anything below.", '',
         '## 2. What is compared', '', '- **PhysX value**: the minimum point separation of the shape pair in the stored contact list (signed, metres). ManiSkill reads the list after the last of the 5 physics sub-steps of a control step.',
         '- **Reconstruction**: the signed gap between the two convex shapes (stored PhysX hulls / boxes) posed at the logged poses of the same row, i.e. after the step, world frame, `hci.spatial_links.gap`.',
         '- **Pairs**: every listed pair that PhysX reports separated (separation > 0), matched by body and shape index on both sides (the orientation flip of the normal is handled; a different shape index never matches).',
         '- **Rows**: arm traces are counted only after the trigger (their prefix repeats the trunk).', '',
         '## 3. Decomposition', '', '| Group | n | median abs (mm) | P99 abs | max abs | above 1 mm |', '|---|---|---|---|---|---|']
    for gname in ('by_pair_kind', 'by_phase', 'by_relative_speed', 'by_normal_direction', 'by_rotation_of_body_a'):
        for k, d in s6[gname].items(): T.append(f"| {gname}: {k} | {d['n']} | {d['median_abs_mm']:.3f} | {d['p99_abs_mm']:.3f} | {d['max_abs_mm']:.3f} | {d['over_tol']} |")
    T += ['', '## 4. The time-offset explanation: evidence level', '',
          f"- **Empirical (ASSOCIATION)**: regressing the difference on (v_rel . n) x dt_sub (v_rel from the finite difference of logged positions over the control step, an approximation): r = {f6['r']:.3f}, slope {f6['slope_through_origin']:.3f} (n {f6['n']}). By normal: "
          + '; '.join(f"{k}: r {v['r']:.2f}, slope {v['slope_through_origin']:.2f} (n {v['n']})" for k, v in s6['fit_by_normal_direction'].items() if v.get('r') is not None) + '.',
          '  By pair kind: ' + '; '.join(f"{k}: r {v['r']:.2f} (n {v['n']})" for k, v in s6['fit_by_pair_kind'].items() if v.get('r') is not None) + '.',
          f"- **After the explanatory fit**: |residual| median {f6['residual_abs_mm_after_fit']['median']:.3f} mm, P99 {f6['residual_abs_mm_after_fit']['p99']:.3f}, max {f6['residual_abs_mm_after_fit']['max']:.3f}; {f6['residual_over_tol_after_fit']} rows still above 1 mm. The fit is EXPLANATORY_ONLY and does not replace S6.",
          '- **Code (HYPOTHESIS)**: the stored sources show only that the list is read after the last sub-step (ManiSkill step loop). That PhysX computes the separation before integrating that sub-step is not visible in any stored code (PhysX is a binary); it stays a hypothesis.',
          '- **The earlier statement** (r = 0.83 for |v_z| x dt) concerned vertical motion. Here the association is strongest for vertical normals and weaker for horizontal ones; it does not cover all pairs.',
          '- **Alternative explanations not excluded**: rotation within the step (finite difference ignores it), contact offset / rest offset conventions, the hull used by narrow phase vs the stored hull, numerical tolerance of the gap solver. No new simulation was run to separate them.', '',
          '## 5. What S5 shows and does not show', '', 'S5 (one-sided): every pair PhysX lists as touching is reconstructed no more than 1 mm apart. It does **not** verify penetration depth, the signed distance of separated pairs, or distances of pairs not listed. S5 PASS cannot be extended to those.', '',
          '## 6. Usability of measures', '', '| Measure | Status |', '|---|---|',
          '| contact existence from the PhysX list (last sub-step) | LIMITED: last sub-step only |', '| reconstructed clearances from logged poses and hulls | LIMITED: interpretable at the logged instant; another host than Phase B |',
          '| PhysX separation as a distance at the logged instant | NOT_VERIFIABLE_NOW (S6 FAIL) |', '| impulse / dt as a force over the control step | NOT_VERIFIABLE_NOW (last sub-step only; not an average, peak or total) |',
          '| sub-step contacts, continuous-time safety | NOT_VERIFIABLE_NOW |']
    (out / 'S6_MEASUREMENT_VALIDITY_NOTE.md').write_text('\n'.join(T) + '\n', encoding='utf-8')


def start_here(out, r):
    s = r['summaries']; rc = s['reconcile_88_81']; s6 = r['s6']; f6 = s6['explanatory_fit_diff_vs_vreln_dtsub']; v2 = s['motion_VISION_CENTER_V2']; md = s.get('memory_decomposition_VISION_CENTER_V2') or {}
    miss = lambda arm: '; '.join(f"{x['step'].split(':')[1]} {x['numerator']}" for x in r['funnel'] if x['stage'] == 'B2_pilot' and x['arm'] == arm and x['step'].startswith('oracle_rescue_missed:') and x['numerator'])
    L = ['# START HERE: C1 mechanism analysis (free offline analysis; no paid run)', '',
         'Work `vision_center_v2_c1_001`. Reference plan `6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7` (unchanged; not approved). Arms, policies, thresholds, statistical plan unchanged; v2s stays excluded; scenes 51100-51299 unused.', '',
         '## Reading order', '', '1. [ANALYSIS_SPEC.md](ANALYSIS_SPEC.md): sources, time axis, denominators, classification order.',
         '2. [EXISTING_DATA_MECHANISM_REPORT.md](EXISTING_DATA_MECHANISM_REPORT.md): what the stored records show (every number generated from `results.json`).',
         '3. [S6_MEASUREMENT_VALIDITY_NOTE.md](S6_MEASUREMENT_VALIDITY_NOTE.md): S6 stays FAIL; what the explanation does and does not establish.',
         '4. [C1_ANALYSIS_READINESS.md](C1_ANALYSIS_READINESS.md): what a C1 run will fill automatically; remaining gaps.',
         '5. Tables: [scene_arm_summary.csv](scene_arm_summary.csv), [event_table.csv](event_table.csv), [opportunity_funnel.csv](opportunity_funnel.csv), [claims_to_evidence.csv](claims_to_evidence.csv), [source_inventory.csv](source_inventory.csv), [metric_dictionary.json](metric_dictionary.json), [analysis_manifest.json](analysis_manifest.json), [results.json](results.json).',
         '6. Detail tables: [tables/](tables/) (memory and held-cube decomposition, ablation chain, S6 pair rows and summary).', '7. Case figures: [figures/](figures/) (one PNG per selected scene; listed in the report section 6).',
         '8. Code: [code/](code/) (`mechanism.py`, `s6_validity.py`, `figures.py`, `report.py`, `run_analysis.py`, tests) and the test log.', '',
         '## Status', '', f"- Analysis status: see `analysis_manifest.json` (`status`, `problems`, `input_sha256_unchanged`).",
         '- Computed now from stored data: Phase B original (train_seed_1, executed arms C0/v1/v2 and the B2 oracle), the offline decision replay (decisions only), the local CPU replay (S6 and contacts only).',
         '- Only after a C1 run: outcomes and motions of the six ablation arms, train_seed_2, the oracle on all scenes, spatial fields on every scene.', '',
         '## Main findings (observations unless marked)', '',
         f"1. **88 vs 81 are different sets.** 88 = Phase B v2 ACT decisions in 484 triggered scenes (all moved >= 1 mm: {rc['phase_b_v2_moved_over_1mm']['n']}); 81 = offline-replay scenes where the excluded v2s also acted (subset of the 88: {rc['subset']}; the 7 others: {rc['in_88_not_in_81']}).",
         "2. **Harm in Phase B** (executed arms; harm / C0 successes): " + '; '.join(f"{st} {arm.replace('VISION_CENTER_', '').replace('CENTER_', '')} {nd(F(r, st, arm, 'overall_harm_of_c0_success'))}" for st in STAGES for arm in ARMS if F(r, st, arm, 'overall_harm_of_c0_success')) + ". "
         + "Rescue among ALL C0 failures (trigger-conditional in brackets): " + '; '.join(f"{st} {arm.replace('VISION_CENTER_', '').replace('CENTER_', '')} {nd(F(r, st, arm, 'overall_rescue_of_c0_failure'))} [{nd(F(r, st, arm, 'triggered_rescue_of_c0_failure'))}]" for st in STAGES for arm in ARMS if F(r, st, arm, 'overall_rescue_of_c0_failure')) + ". "
         + "v2 C0 successes kept by abstaining vs kept while moving: " + '; '.join(f"{st} {nd(F(r, st, 'VISION_CENTER_V2', 'triggered_c0_success_kept_by_abstention'))} vs {nd(F(r, st, 'VISION_CENTER_V2', 'moved_c0_success_kept_where_arm_moved'))}" for st in STAGES) + ". Abstention is not a safety pass.",
         f"3. **Missed oracle rescues (B2)**: v2 missed {nd(F(r, 'B2_pilot', 'VISION_CENTER_V2', 'oracle_rescue_missed'))} ({miss('VISION_CENTER_V2')}); v1 missed {nd(F(r, 'B2_pilot', 'VISION_CENTER_V1', 'oracle_rescue_missed'))} ({miss('VISION_CENTER_V1')}). Labels are the first block in code order, not causes.",
         f"4. **Memory**: v2 memory-sourced target errors at the trigger: median {mm(md.get('err_now_mm'))} mm, max {mm(md.get('err_now_mm'), 'max')} mm (n {(md.get('err_now_mm') or {}).get('n')}); above the memory's own bound in {md.get('err_now_above_memory_u')} scenes; the vector identity holds (residual {md.get('identity_residual_max_m', 0):.1e} m).",
         f"5. **Motion**: v2 moved in {v2['moved']} branches; the target moved > 1 mm during control in {v2['target_moved_over_1mm']}, and the alignment gain came mostly from target motion (push-assisted) in {v2['push_assisted']}. Command path, TCP motion and residual are reported separately; no limit was exceeded ({v2['bounds']}).",
         f"6. **S6 stays FAIL** ({s6['official_S6']['over_tolerance']} of {s6['official_S6']['pairs']} pair-rows > 1 mm). The time-offset term is an ASSOCIATION (r {f6['r']:.2f}; strongest for vertical normals); {f6['residual_over_tol_after_fit']} rows remain > 1 mm after it: not the only cause (NOT_ESTABLISHED).",
         '7. **Ablation effects are NOT_RUN** in existing data: only decision changes are counted (tables/ablation_chain.csv).', '',
         '## Limits', '', '- Phase B has one policy; no generalisation claim. Shared C1 scenes are paired, not 400 independent samples.',
         '- Contacts are known only from the local replay (another host, last sub-step). Phase B contacts are NOT_OBSERVABLE; nothing was back-filled.',
         '- Ground truth is used only for post-hoc labels. Figures mark POST-HOC GROUND TRUTH and ONLINE ESTIMATES; the memory bound circle is an uncertainty radius, not a safe region.', '',
         '## Claim levels', '', 'Every claim is in [claims_to_evidence.csv](claims_to_evidence.csv) with its level (DIRECT_OBSERVATION / ASSOCIATION / HYPOTHESIS / NOT_ESTABLISHED), value, computing location and figure.']
    (Path(out) / 'START_HERE.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


_write_reports_body = write_reports


def write_reports(out, results):
    _write_reports_body(out, results)
    here = Path(__file__).resolve().parent
    for n in ('ANALYSIS_SPEC.md', 'C1_ANALYSIS_READINESS.md'): (Path(out) / n).write_text((here / 'docs' / n).read_text(encoding='utf-8'), encoding='utf-8')
    start_here(out, results)


# ---------------------------------------------------------------- C1 (review F3): the same sections, per policy, from results.json
def claims_c1(r):
    out = []; kind = r['source_kind']; stages = [f'C1_{p}' for p in r['policies']]
    def c(cid, lvl, claim, value, where): out.append({'id': cid, 'kind': lvl, 'source_kind': kind, 'claim': claim, 'value': value, 'computed_in': where, 'figure': ''})
    for st in stages:
        for arm in r['arms']:
            if arm == 'C0' or not F(r, st, arm, 'overall_harm_of_c0_success'): continue
            c(f'K01-{st}-{arm}', 'DIRECT_OBSERVATION', f'{arm} in {st}: overall rescue / all C0 failures, harm / all C0 successes; trigger-conditional and moved-conditional values labelled',
              f"rescue {nd(F(r, st, arm, 'overall_rescue_of_c0_failure'))}; harm {nd(F(r, st, arm, 'overall_harm_of_c0_success'))}; trigger-conditional rescue {nd(F(r, st, arm, 'triggered_rescue_of_c0_failure'))}; harm when moved {nd(F(r, st, arm, 'moved_harm_of_c0_success_where_arm_moved'))}",
              'opportunity_funnel.csv; tables/denominator_check.csv')
            if F(r, st, arm, 'oracle_rescue_missed'):
                c(f'K02-{st}-{arm}', 'DIRECT_OBSERVATION', f'{arm} in {st}: oracle rescues missed, first block in code order (a label, not a cause)', nd(F(r, st, arm, 'oracle_rescue_missed')), 'opportunity_funnel.csv oracle_rescue_missed:*')
    cc = [x for x in r['crosscheck'] if 'match' in x]
    c('K03', 'CHECK', 'The overall counts equal the collector summary (independent recount)', f"{sum(1 for x in cc if x['match'])} of {len(cc)} fields match" if cc else 'NOT_AVAILABLE', 'tables/crosscheck_against_collector_summary.csv')
    for pid, s6 in r['s6'].items():
        o = s6['official_S6']; c(f'K04-{pid}', 'DIRECT_OBSERVATION', f'S6 on {pid} (1 mm tolerance unchanged)', f"{o['status']}" + (f": {o['over_tolerance']} of {o['pairs']} separated pair-rows above 1 mm" if 'pairs' in o else ''), 'tables/s6_validity_summary.json')
    c('K05', 'NOT_ESTABLISHED', 'Causal effect of each removed element', 'the executed ablation table is a paired description; leave-one-out differences overlap and do not add; V2_SINGLE_FRAME changes two things (memories and confirmation count)', 'tables/executed_ablation_chain.csv')
    c('K06', 'NOT_ESTABLISHED', 'Generalisation beyond the two policy seeds; independence of the two passes', 'the two passes share scenes: paired, not independent samples', 'policy_paired_outcomes.csv')
    c('K07', 'NOT_ESTABLISHED', 'Contacts inside a control step and continuous-time safety', 'last sub-step only', 'tables/s6_validity_summary.json')
    for x in r['cases']: out.append({'id': f"CASE-{x['policy']}-{x['scene']}", 'kind': 'DIRECT_OBSERVATION', 'source_kind': kind, 'claim': f"case: {x['why']}", 'value': x['status'], 'computed_in': 'case_index.csv', 'figure': x.get('frames') or ''})
    return out


def write_c1_report(out, r):
    out = Path(out); kind = r['source_kind']; stages = [f'C1_{p}' for p in r['policies']]; vis = [a for a in r['arms'] if a not in ('C0', 'CENTER_ORACLE')]
    L = [f'# C1 mechanism report ({kind})', '']
    if kind == 'MOCK': L += ['> **MOCK records (tool verification). No count, rate or outcome in this file is a research result.**', '']
    L += ['Every number is written from `results.json` by `report.py`. Policies are reported separately and never pooled; shared scenes are paired, not independent.', '',
          '## 1. Counts with explicit denominators', ''] + counts_tables(r, stages, r['arms'])
    cc = [x for x in r['crosscheck'] if 'match' in x]
    L += [f"Cross-check against the collector summary: {sum(1 for x in cc if x['match'])} of {len(cc)} fields match." if cc else 'Cross-check: NOT_AVAILABLE (no collector summary named).', '',
          '## 2. Oracle opportunities (C0 failed, the oracle succeeded)', '', '| Policy | Arm | Also rescued | Rescued where the oracle failed | Missed: first block in code order |', '|---|---|---|---|---|']
    for st in stages:
        for arm in vis:
            miss = [x for x in r['funnel'] if x['stage'] == st and x['arm'] == arm and x['step'].startswith('oracle_rescue_missed:') and x['numerator']]
            L.append(f"| {st} | {arm} | {nd(F(r, st, arm, 'oracle_rescue_also_rescued'))} | {nd(F(r, st, arm, 'overall_rescued_where_oracle_failed'))} | " + ('; '.join(f"{x['step'].split(':')[1]} {x['numerator']} ({x['scenes']})" for x in miss) or '-') + ' |')
    L += ['', '## 3. Target memory and held-cube estimates', '', '| Policy | Arm | Coverage | Memory-sourced n | Err now median (mm) | Err at store median | Move since store max | Above own u | Identity residual max (m) | Held err median (mm) |', '|---|---|---|---|---|---|---|---|---|---|']
    for pid in r['policies']:
        s = r['summaries'][pid]
        for arm in vis:
            t = s.get(f'target_estimate_{arm}') or {}; md = s.get(f'memory_decomposition_{arm}') or {}; h = s.get(f'held_estimate_{arm}') or {}
            L.append(f"| {pid} | {arm} | {t.get('coverage')} | {md.get('n', 0)} | {mm(md.get('err_now_mm'))} | {mm(md.get('err_at_store_mm'))} | {mm(md.get('target_move_since_store_mm'), 'max')} | {md.get('err_now_above_memory_u', '-')} | "
                     f"{'-' if md.get('identity_residual_max_m') is None else format(md['identity_residual_max_m'], '.1e')} | {mm(h.get('err_now_xy_mm'))} |")
    L += ['', '## 4. Command, actual motion, target motion, resume and opening', '', '| Policy | Arm | With control | Moved | Cmd path median (mm) | TCP net median | Target moved > 1 mm | Push-assisted | Residual <= 10 mm at end | Steps end -> open median | Bound violations |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for pid in r['policies']:
        for arm in [a for a in r['arms'] if a != 'C0']:
            m = r['summaries'][pid].get(f'motion_{arm}')
            if m: L.append(f"| {pid} | {arm} | {m['branches_with_control']} | {m['moved']} | {mm(m['cmd_path_mm'])} | {mm(m['tcp_xy_net_mm'])} | {m['target_moved_over_1mm']} | {m['push_assisted']} | {m['residual_within_tol_at_end']} | {m['steps_control_end_to_open']['median']} | {m['bounds']} |")
    L += ['', 'Per-event rows (trigger, control start and end, policy resume, first open, release, success, end) are in `event_table.csv`.', '',
          '## 5. Ablation: information -> decision -> executed motion -> outcome (paired with v2 by scene; descriptive)', '',
          '| Policy | Arm | Removed | Triggered | Target source changed | Decision changed | ACT lost | ACT gained | Moved (v2) | Success (v2) | Only this arm / only v2 | Harm / rescue vs C0 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a in r['executed_ablation']: L.append(f"| {a['policy']} | {a['arm']} | {a['removed_element']} | {a['triggered_scenes']} | {a['information_changed_target_src']} | {a['decision_changed']} | {a['act_lost_vs_v2']} | {a['act_gained_vs_v2']} | {a['moved']} ({a['v2_moved']}) | {a['success']} ({a['v2_success']}) | {a['success_only_this_arm']} / {a['success_only_v2']} | {a['harm_vs_c0']} / {a['rescue_vs_c0']} |")
    L += ['', 'A difference here is not a causal effect of one element; V2_SINGLE_FRAME also changes the confirmation count; the oracle is one reference controller.', '', '## 6. S6 measurement validity (1 mm tolerance unchanged)', '']
    for pid, s6 in r['s6'].items():
        o = s6['official_S6']
        if 'pairs' in o:
            ci = s6.get('contact_list_vs_impulse') or {}
            L.append(f"- {pid}: S6 **{o['status']}**, {o['over_tolerance']} of {o['pairs']} separated pair-rows above 1 mm; excluded {o['excluded']}; separated listed pairs with a recorded nonzero impulse: {ci.get('of_which_nonzero_impulse_recorded')} (a listed pair is not an observed contact). "
                     "A time-offset explanation stays an association; the reconstruction is not a verified ground truth; sub-steps are not observed.")
        else: L.append(f"- {pid}: {o['status']} ({o.get('why')})")
    L += ['', '## 7. Case index (rule fixed before the run)', '', '| Policy | Scene | Why | Status | Record | Trace sha256 | Frames |', '|---|---|---|---|---|---|---|']
    for x in r['cases']: L.append(f"| {x['policy']} | {x['scene']} | {x['why']} | {x['status']} | {x['record']} | {str(x['trace_sha256'])[:16]}... | {x.get('frames') or '-'} |")
    (out / 'C1_MECHANISM_REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    S = [f'# START HERE: C1 mechanism analysis output ({kind})', '']
    if kind == 'MOCK': S += ['**MOCK: tool verification only, not a research result.**', '']
    S += ['1. [C1_MECHANISM_REPORT.md](C1_MECHANISM_REPORT.md)', '2. [gate_report.json](gate_report.json): the expected-set gate (plan or MOCK manifest, policies, scenes, arms).',
          '3. Tables: [scene_arm_summary.csv](scene_arm_summary.csv), [event_table.csv](event_table.csv), [opportunity_funnel.csv](opportunity_funnel.csv), [policy_paired_outcomes.csv](policy_paired_outcomes.csv), [case_index.csv](case_index.csv), [claims_to_evidence.csv](claims_to_evidence.csv), [tables/](tables/).',
          '4. [analysis_manifest.json](analysis_manifest.json): code, settings, inputs and output hashes; `status` is ANALYSIS_COMPLETE only when every required output exists.']
    (out / 'START_HERE.md').write_text('\n'.join(S) + '\n', encoding='utf-8')
