"""Mechanism-analysis runner. Reads stored records only; writes a new output folder (refuses an existing one).

  existing : Phase B original + stored offline decision replay + stored local CPU replay (S6 validity)
  c1       : a future, separately approved C1 run folder (pass_train_seed_1 / pass_train_seed_2), same tables
"""
from __future__ import annotations

import argparse, gzip, json, platform, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE), str(HERE.parent / 'src')]
import numpy as np
import mechanism as M

CODE_FILES = ['mechanism_analysis/mechanism.py', 'mechanism_analysis/completeness.py', 'mechanism_analysis/s6_validity.py', 'mechanism_analysis/run_analysis.py', 'mechanism_analysis/figures.py', 'mechanism_analysis/report.py', 'mechanism_analysis/metric_dictionary.json', 'mechanism_analysis/docs/ANALYSIS_SPEC.md', 'mechanism_analysis/docs/C1_ANALYSIS_READINESS.md', 'mechanism_analysis/tests/test_mechanism.py', 'mechanism_analysis/tests/test_targeted_fix.py']
import completeness as _G
FROZEN = _G.FROZEN_PLAN_SHA256
CASES = (50632, 50892, 51004, 51009, 50959, 50702)
PHASE_B = {'B1_shadow': 'shadow', 'B2_pilot': 'pilot'}


def rel(p, root):
    try: return Path(p).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError: return Path(p).name


def inventory(run, kind, run_id, host, code, repo, extra=()):
    rows = []
    for stage, sub in run.stages.items():
        for f in sorted((run.root / sub).rglob('*')):
            if f.is_file(): rows.append({'path': rel(f, repo), 'source_kind': kind, 'run': run_id, 'stage': stage, 'host': host, 'code': code, 'sha256': M.sha256(f), 'bytes': f.stat().st_size})
    for f in extra:
        if Path(f).exists(): rows.append({'path': rel(f, repo), 'source_kind': kind, 'run': run_id, 'stage': '', 'host': host, 'code': code, 'sha256': M.sha256(f), 'bytes': Path(f).stat().st_size})
    return rows


def hashes(paths): return {str(p): M.sha256(p) for p in paths}


def summaries(res, off, arms=('VISION_CENTER_V1', 'VISION_CENTER_V2', 'CENTER_ORACLE')):
    rows = [r for r in res['rows']]; s = {}
    # memory decomposition (target), per arm and source, with coverage
    for arm in [x for x in arms if x in M.VISION_ARMS]:
        mm = [m for m in res['memory'] if m['arm'] == arm]; cov = Counter(m['status'] if m['status'] != 'ESTIMATED' else f"ESTIMATED:{m['target_src']}" for m in mm)
        s[f'target_estimate_{arm}'] = {'triggered_with_control_record': len(mm), 'coverage': dict(cov),
                                        'no_estimate_reasons': dict(Counter(f"{m.get('why')}:{m.get('memory_reason')}" for m in mm if m['status'] == 'NO_TARGET_ESTIMATE')),
                                        'err_now_mm_by_source': {src: {k: (None if x is None else x * 1e3) if k != 'n' else x for k, x in M.dist([m['err_now_m'] for m in mm if m.get('target_src') == src]).items()}
                                                                 for src in sorted({m['target_src'] for m in mm if m.get('target_src')})}}
        mem = [m for m in mm if str(m.get('target_src', '')).startswith('memory_') and m['status'] == 'ESTIMATED']
        if mem:
            s[f'memory_decomposition_{arm}'] = {'n': len(mem), 'identity_residual_max_m': max(m['identity_residual_m'] for m in mem),
                                                'err_at_store_mm': {k: (x * 1e3 if k != 'n' and x is not None else x) for k, x in M.dist([m['err_at_store_m'] for m in mem]).items()},
                                                'target_move_since_store_mm': {k: (x * 1e3 if k != 'n' and x is not None else x) for k, x in M.dist([m['target_move_since_store_m'] for m in mem]).items()},
                                                'err_now_mm': {k: (x * 1e3 if k != 'n' and x is not None else x) for k, x in M.dist([m['err_now_m'] for m in mem]).items()},
                                                'age_steps': M.dist([m['age_steps'] for m in mem]),
                                                'err_now_above_memory_u': sum(1 for m in mem if m.get('err_now_within_memory_u') is False),
                                                'target_moved_more_than_1mm_since_store': sum(1 for m in mem if m['target_move_since_store_m'] > 0.001),
                                                'scenes_err_now_above_memory_u': sorted(m['scene'] for m in mem if m.get('err_now_within_memory_u') is False)}
        hh = [h for h in res['hold'] if h['arm'] == arm]
        s[f'held_estimate_{arm}'] = {'n': len(hh), 'coverage': dict(Counter(f"{h['status']}:{h.get('hold_src')}" for h in hh)),
                                      'err_now_xy_mm': {k: (x * 1e3 if k != 'n' and x is not None else x) for k, x in M.dist([h.get('err_now_xy_m') for h in hh if h['status'] == 'ESTIMATED']).items()},
                                      'held_rel_tcp_change_since_fit_mm': {k: (x * 1e3 if k != 'n' and x is not None else x) for k, x in M.dist([h.get('held_rel_tcp_change_since_fit_m') for h in hh]).items()}}
    # motion during control
    for arm in arms:
        a = [r for r in rows if r['arm'] == arm and r['control_steps'] > 0]
        mm = lambda k: {kk: (x * 1e3 if kk != 'n' and x is not None else x) for kk, x in M.dist([r.get(k) for r in a]).items()}
        s[f'motion_{arm}'] = {'branches_with_control': len(a), 'moved': sum(1 for r in a if r['moved']), 'cmd_path_mm': mm('cmd_path_m'), 'cmd_net_mm': mm('cmd_net_m'), 'tcp_xy_net_mm': mm('tcp_xy_net_m'),
                              'tcp_xy_path_mm': mm('tcp_xy_path_m'), 'held_xy_net_mm': mm('held_xy_net_m'), 'held_minus_tcp_xy_change_mm': mm('held_minus_tcp_xy_change_m'), 'target_xy_net_mm': mm('target_xy_net_m'),
                              'target_moved_over_1mm': sum(1 for r in a if r.get('target_moved')), 'push_assisted': sum(1 for r in a if r.get('push_assisted')), 'push_assisted_scenes': sorted(r['scene'] for r in a if r.get('push_assisted')),
                              'offset_before_mm': mm('offset_before_m'), 'offset_after_mm': mm('offset_after_m'), 'residual_within_tol_at_end': sum(1 for r in a if r.get('residual_within_tol_at_end')),
                              'cos_cmd_vs_true_offset': M.dist([r.get('cos_cmd_vs_true_offset') for r in a]), 'cos_cmd_NA': sum(1 for r in a if r.get('cos_cmd_vs_true_offset') is None),
                              'end_reason': dict(Counter(r['end_reason'] for r in a)), 'grasp_lost_during_control': sum(1 for r in a if r.get('grasped_before') is True and r.get('grasped_after') is False),
                              'bounds': {'step_cmd_violations': sum(1 for r in a if r.get('within_cmd_step_bound') is False), 'target_cap_violations': sum(1 for r in a if r.get('within_target_cap') is False), 'max_steps_violations': sum(1 for r in a if r.get('within_max_steps') is False)},
                              'steps_control_end_to_open': M.dist([r.get('steps_control_end_to_open') for r in a]),
                              'offset_change_end_to_open_mm': mm('offset_change_end_to_open_m'),
                              'outcome_vs_c0': dict(Counter(r['outcome_vs_c0'] for r in a))}
    s['harm_events'] = [{'stage': r['stage'], 'scene': r['scene'], 'arm': r['arm']} for r in rows if r.get('outcome_vs_c0') == 'HARM']
    s['oracle_rescue_missed'] = [{'stage': r['stage'], 'scene': r['scene'], 'arm': r['arm'], 'miss_class': r['miss_class'], 'flags': r['miss_flags'], 'decision': r['decision'], 'est_offset_m': r['est_offset_m'], 'true_offset_m': r['true_offset_m']}
                                 for r in rows if r.get('miss_class')]
    s['offline_ablation'] = M.ablation_table(off)
    return s


def select_cases(rows, fixed=CASES):
    """Fixed rule: the six existing cases; then, per type, the lowest seed: v2 rescue with push assistance; v2 move whose
    estimate error was outside its own bound; each oracle-rescue-missed class for v2 and v1; every harm event."""
    sel = {c: 'existing explanatory case' for c in fixed}
    def first(pred, why):
        c = sorted(r['scene'] for r in rows if pred(r))
        for s in c:
            if s not in sel: sel[s] = why; return
    first(lambda r: r['arm'] == 'VISION_CENTER_V2' and r.get('outcome_vs_c0') == 'RESCUE' and r.get('push_assisted'), 'v2 rescue whose alignment reduction came mostly from target motion')
    first(lambda r: r['arm'] == 'VISION_CENTER_V2' and r['acted'] and r.get('error_within_bound') is False, 'v2 move with estimate error outside its own bound')
    for arm in ('VISION_CENTER_V2', 'VISION_CENTER_V1'):
        for m in M.MISS_ORDER: first(lambda r, m=m, arm=arm: r['arm'] == arm and r.get('miss_class') == m, f'{arm} missed an oracle rescue: {m}')
    for r in rows:
        if r.get('outcome_vs_c0') == 'HARM' and r['scene'] not in sel: sel[r['scene']] = f'harm event ({r["arm"]})'
    return sel


def existing(a):
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    repo = HERE.parent; run = M.Run(a.phase_b, 'PHASE_B_ORIGINAL', 'vision_center_v2_001', PHASE_B, 'train_seed_1', host='Phase B server (RTX 4090, plan host record)')
    inputs = [p for st in PHASE_B.values() for p in sorted((Path(a.phase_b) / st).rglob('*')) if p.is_file()] + [Path(a.offline)] + sorted(Path(a.replay).rglob('*.gz')) + [Path(a.replay) / 'replay_summary.json'] + [repo / f for f in CODE_FILES if (repo / f).is_file()]
    before = hashes(inputs)
    res = M.analyse_run(run); off = M.offline_rows(a.offline)
    problems = M.check_keys(res['rows']) + M.check_keys(off) + [f'{r["branch"]}: {p}' for r in res['rows'] for p in M.check_time_order(r) + M.check_integrity(r)]
    problems += [f"{e['branch']} {e['event']} at row {e['t']}: pose missing" for e in res['events'] if e['t'] is not None and e.get('pose_missing')]
    fun = M.funnel(res['rows'], ['C0', 'VISION_CENTER_V1', 'VISION_CENTER_V2', 'CENTER_ORACLE']); problems += M.check_funnel(fun)
    off_fun = []
    for stage in sorted({r['stage'] for r in off}):
        for arm in sorted({r['arm'] for r in off}):
            s = [r for r in off if r['stage'] == stage and r['arm'] == arm]; act = [r for r in s if r['act']]
            off_fun.append({'run': 'offline_replay_r4', 'stage': stage, 'policy': 'train_seed_1', 'arm': arm, 'step': 'decision_ACT (decision only, NOT a movement)', 'numerator': len(act), 'denominator': len(s),
                            'fraction': M.ratio(len(act), len(s)), 'parent_step': 'triggered', 'scenes': ' '.join(str(r['scene']) for r in sorted(act, key=lambda r: r['scene'])), 'note': 'OFFLINE_DECISION_REPLAY: no movement, no outcome'})
    out.mkdir(parents=True); (out / 'tables').mkdir(); (out / 'figures').mkdir()
    M.write_csv(out / 'scene_arm_summary.csv', res['rows'] + off)
    M.write_csv(out / 'event_table.csv', res['events'])
    M.write_csv(out / 'opportunity_funnel.csv', [{**r, 'source_kind': 'PHASE_B_ORIGINAL'} for r in fun] + [{**r, 'source_kind': 'OFFLINE_DECISION_REPLAY'} for r in off_fun])
    M.write_csv(out / 'tables' / 'target_estimate_and_memory_decomposition.csv', res['memory']); M.write_csv(out / 'tables' / 'held_estimate_decomposition.csv', res['hold'])
    s = summaries(res, off); M.write_csv(out / 'tables' / 'ablation_chain.csv', s['offline_ablation'])
    o = M.read_json(a.offline); v2_act = sorted(r['scene'] for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2' and r['decision'] == 'ACT')
    v2_moved = sorted(r['scene'] for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2' and r['moved']); v2s_ctl = sorted(r['scene'] for r in o['rows'] if 'v2s_control' in r)
    v2s_act = sorted(r['scene'] for r in o['rows'] if ((r.get('arms') or {}).get('VISION_CENTER_V2S') or {}).get('act'))
    s['reconcile_88_81'] = {'phase_b_v2_decision_ACT': {'n': len(v2_act), 'scenes': v2_act}, 'phase_b_v2_moved_over_1mm': {'n': len(v2_moved)},
                            'offline_r4_rows_with_v2s_control_replay': {'n': len(v2s_ctl), 'scenes': v2s_ctl, 'meaning': 'scenes where the excluded v2s variant also decided ACT, so the v2 control log was replayed against it'},
                            'offline_r4_v2s_decision_ACT': {'n': len(v2s_act), 'equals_rows_with_v2s_control': v2s_act == v2s_ctl}, 'subset': set(v2s_ctl) <= set(v2_act), 'in_88_not_in_81': sorted(set(v2_act) - set(v2s_ctl)), 'in_81_not_in_88': sorted(set(v2s_ctl) - set(v2_act)),
                            'sources': {'88': 'PHASE_B_ORIGINAL scene records, cell VISION_CENTER_V2 control.decision == ACT (B1 + B2 triggered scenes, 484)', '81': 'OFFLINE_DECISION_REPLAY offline_replay_r4.json rows with v2s_control'}}
    # S6 validity on the stored local replay
    import s6_validity as S6
    recs = {}
    for _, f, rec in run.scenes():
        if isinstance(rec, dict) and M.UNREADABLE not in rec and rec.get('scene') is not None: recs[rec['scene']] = rec      # malformed records are already recorded errors (review m7)
    s6rows = S6.collect(a.replay, recs); s6 = S6.summarise(s6rows, excluded=S6.collect.excluded)
    with gzip.open(out / 'tables' / 's6_pair_rows.csv.gz', 'wt', encoding='utf-8', newline='') as g:
        import csv
        cols = sorted({k for r in s6rows for k in r}); w = csv.DictWriter(g, fieldnames=cols); w.writeheader(); [w.writerow({k: ('' if r.get(k) is None else r[k]) for k in cols}) for r in s6rows]
    (out / 'tables' / 's6_validity_summary.json').write_text(json.dumps({**s6, 'source_kind': 'LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS'}, indent=1), encoding='utf-8')
    # review F1/F3: independent recount against the collector's own Phase B summaries (O.paired layout, all scenes)
    import completeness as G
    pb = Path(a.phase_b); ref = {}
    for stage, f, keys in (('B1_shadow', pb / 'shadow' / 'shadow_result.json', {'VISION_CENTER_V2': 'paired_V2_vs_C0', 'VISION_CENTER_V1': 'paired_V1_vs_C0'}),
                           ('B2_pilot', pb / 'pilot' / 'result.json', {'VISION_CENTER_V2': 'primary_V2_vs_C0', 'VISION_CENTER_V1': 'baseline_V1_vs_C0', 'CENTER_ORACLE': 'reference_CENTER_ORACLE_vs_C0'})):
        if not f.is_file(): problems.append(f'COLLECTOR_RESULT_MISSING: {f.name}'); continue
        cr = M.read_json(f)
        for arm, k in keys.items(): ref[(stage, 'train_seed_1', arm)] = cr[k]
    cc_rows, cc_p = G.crosscheck(fun, ref); problems += cc_p
    M.write_csv(out / 'tables' / 'crosscheck_against_collector_summary.csv', cc_rows); M.write_csv(out / 'tables' / 'denominator_check.csv', denominator_check(fun))
    sel = select_cases(res['rows'])
    import figures as F
    figs = F.case_figures(run, res, sel, out / 'figures')
    import report as R
    results = {'funnel': fun, 'offline_funnel': off_fun, 'denominator_check': denominator_check(fun), 'crosscheck': cc_rows, 'summaries': s, 's6': s6, 'cases': {str(k): v for k, v in sel.items()}, 'figures': figs, 'problems': problems, 'errors': res['errors']}
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
    M.write_csv(out / 'source_inventory.csv', inventory(run, 'PHASE_B_ORIGINAL', 'vision_center_v2_001', 'Phase B server', 'plan b7078afb (Phase B)', repo, [Path(a.phase_b) / 'plan.json'])
                + [{'path': rel(a.offline, repo), 'source_kind': 'OFFLINE_DECISION_REPLAY', 'run': 'offline_replay_r4', 'stage': '', 'host': 'local CPU', 'code': 'results/vision_center_v2_c1_001/preparation/offline_replay.py', 'sha256': M.sha256(a.offline), 'bytes': Path(a.offline).stat().st_size}]
                + [{'path': 'LOCAL_REPLAY/' + p.name, 'source_kind': 'LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS', 'run': 'replay_003', 'stage': '', 'host': 'local WSL CPU (pinned ManiSkill a4a4f92, SAPIEN 3.0.3)', 'code': 'scripts/37_c1_probe_sim_check.py replay', 'sha256': M.sha256(p), 'bytes': p.stat().st_size} for p in sorted(Path(a.replay).rglob('*')) if p.is_file()]
                + [{'path': 'C1_ORIGINAL', 'source_kind': 'C1_ORIGINAL', 'run': 'vision_center_v2_c1_001', 'stage': '', 'host': '', 'code': f'plan {FROZEN}', 'sha256': '', 'bytes': 0, 'note': 'does not exist: no C1 run was made'}])
    (out / 'metric_dictionary.json').write_text((HERE / 'metric_dictionary.json').read_text(encoding='utf-8'), encoding='utf-8')
    after = hashes(inputs)
    if after != before: problems.append('an input file changed during the analysis')
    if not problems and not res['errors']: M.write_csv(out / 'claims_to_evidence.csv', R.claims(results)); R.write_reports(out, results)      # review F2: no report from an incomplete analysis
    else: incomplete_marker(out, problems + [str(e) for e in res['errors']])
    man = {'what': 'C1 mechanism decomposition on stored records (no simulation, no rendering, no rollout)', 'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'numpy': np.__version__,
           'c1_plan_sha256': FROZEN, 'c1_plan_file_sha256': M.sha256(a.c1_plan) if a.c1_plan and Path(a.c1_plan).exists() else None,
           'code_sha256': {f: M.sha256(repo / f) for f in CODE_FILES if (repo / f).exists()}, 'inputs': {'phase_b_files': len([p for p in before if 'server_backup' in p or 'shadow' in p or 'pilot' in p]), 'input_sha256_unchanged': after == before,
           'offline_replay': {'path': rel(a.offline, repo), 'sha256': M.sha256(a.offline)}, 'local_replay_files': {Path(p).name: h for p, h in before.items() if Path(a.replay) in Path(p).parents or Path(p).parent == Path(a.replay)}},
           'settings': {k: getattr(M, k) for k in ('MOVE_MIN_M', 'PUSH_MIN_M', 'NA_NORM_M', 'RESIDUAL_TOL_M', 'TARGET_CAP_M', 'STEP_CMD_M', 'MAX_STEPS')},
           'problems': problems, 'status': 'ANALYSIS_COMPLETE' if not problems and not res['errors'] else 'INCOMPLETE'}
    finish(out, man, {'rows': len(res['rows']), 'offline_rows': len(off), 's6_pairs': len(s6rows), 'cases': len(sel)})          # exit 3 on any problem or recorded error (review m6)


C1_REQUIRED = ('gate_report.json', 'scene_arm_summary.csv', 'event_table.csv', 'opportunity_funnel.csv', 'policy_paired_outcomes.csv', 'tables/target_estimate_and_memory_decomposition.csv',
               'tables/held_estimate_decomposition.csv', 'tables/executed_ablation_chain.csv', 'tables/denominator_check.csv', 'tables/crosscheck_against_collector_summary.csv', 'tables/s6_validity_summary.json', 'tables/s6_pair_rows.csv.gz',
               'case_index.csv', 'results.json', 'claims_to_evidence.csv', 'C1_MECHANISM_REPORT.md', 'START_HERE.md', 'metric_dictionary.json')


def denominator_check(fun):
    """OVERALL / TRIGGERED / MOVED denominators side by side (review F1), one row per stage, policy and arm."""
    d = {(x['stage'], x['policy'], x['arm'], x['step']): x for x in fun}; out = []
    for st, pol, arm in sorted({(x['stage'], x['policy'], x['arm']) for x in fun}):
        g = lambda s, k='numerator': (d.get((st, pol, arm, s)) or {}).get(k)
        out.append({'stage': st, 'policy': pol, 'arm': arm, 'all_scenes': g('scenes'), 'triggered': g('triggered'), 'not_triggered': g('not_triggered'), 'complete_outcomes': g('overall_scenes_with_complete_outcomes'),
                    'c0_success_all': g('overall_c0_success'), 'c0_failure_all': g('overall_c0_failure'), 'c0_failure_triggered': g('triggered_rescue_of_c0_failure', 'denominator'), 'c0_success_triggered': g('triggered_harm_of_c0_success', 'denominator'),
                    'success_all': g('overall_success'), 'rescue': g('overall_rescue_of_c0_failure'), 'rescue_over_all_c0_failures': g('overall_rescue_of_c0_failure', 'fraction'),
                    'rescue_over_triggered_c0_failures': g('triggered_rescue_of_c0_failure', 'fraction'), 'harm': g('overall_harm_of_c0_success'), 'harm_over_all_c0_successes': g('overall_harm_of_c0_success', 'fraction'),
                    'harm_over_triggered_c0_successes': g('triggered_harm_of_c0_success', 'fraction'), 'c0_successes_where_arm_moved': g('moved_harm_of_c0_success_where_arm_moved', 'denominator'),
                    'harm_where_arm_moved': g('moved_harm_of_c0_success_where_arm_moved'), 'net_rescue_minus_harm': g('overall_net_rescue_minus_harm'), 'outcome_missing': g('outcome_missing_completeness_error')})
    return out


def finish(out, man, extra_print=None):
    man['outputs'] = {p.relative_to(out).as_posix(): M.sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'analysis_manifest.json'}
    (out / 'analysis_manifest.json').write_text(json.dumps(man, indent=1, default=str), encoding='utf-8')
    if not (out / 'analysis_manifest.json').is_file(): raise SystemExit('analysis_manifest.json was not written')
    print(json.dumps({'status': man['status'], 'problems': man['problems'][:12], **(extra_print or {})}, indent=1))
    if man['status'] != 'ANALYSIS_COMPLETE': raise SystemExit(3)


def incomplete_marker(out, problems):
    (out / 'INCOMPLETE_NOT_A_RESULT.txt').write_text('This analysis is INCOMPLETE. No performance, test or mechanism report was written; the tables here (if any) are partial data kept with the problem list.\n\n'
                                                     + '\n'.join(problems) + '\n', encoding='utf-8')


def c1(a):
    """A C1 run folder: an expected-set gate first (review F2), then one Run per policy pass with every existing table and the
    S6, ablation, case and report chain (review F3). Policies are never pooled."""
    import completeness as G, s6_validity as S6, report as R
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    root = Path(a.run); kind = a.source_kind; repo = HERE.parent
    if kind == 'C1_ORIGINAL':
        if a.expected: raise SystemExit('usage: --expected is only for MOCK; a C1_ORIGINAL run is checked against the plan file and its registries')
        exp, gate = G.expected_from_plan(a.plan, repo, FROZEN)
    else:
        if not a.expected: raise SystemExit('usage: MOCK needs --expected <MOCK expected manifest>; the completeness gate is never skipped')
        exp, gate = G.expected_from_mock(a.expected, repo, FROZEN)
    inputs = sorted(p for p in root.rglob('*') if p.is_file()) if root.is_dir() else []
    inputs += [Path(x) for x in ((a.plan, repo / 'configs/vision_center_c1_splits.json', repo / 'configs/vision_center_c1_arms.json') if kind == 'C1_ORIGINAL' else (a.expected, repo / 'configs/vision_center_c1_arms.json', repo / 'configs/vision_center_c1_splits.json', a.plan)) if Path(x).is_file()]
    inputs += [repo / f for f in CODE_FILES if (repo / f).is_file()]      # the analysis code, dictionary and docs copied into the outputs (review f5)
    before = hashes(inputs)      # the run folder and every external file the gate read (review f1 MINOR)
    gate += G.check_run(root, exp, kind)
    out.mkdir(parents=True)
    if kind == 'MOCK': (out / 'MOCK_NOT_A_RESULT.txt').write_text('Every table here comes from MOCK records (tool verification). No count, rate or outcome here is a research result.\n', encoding='utf-8')
    exp_summary = {k: v for k, v in (exp or {}).items() if k not in ('scenes',)}; exp_summary['scenes_n'] = None if not exp or exp.get('scenes') is None else len(exp['scenes'])
    exp_summary['scenes_first_last'] = None if not exp or not exp.get('scenes') else [exp['scenes'][0], exp['scenes'][-1]]
    (out / 'gate_report.json').write_text(json.dumps({'expected': exp_summary, 'problems': gate, 'pass': not gate}, indent=1), encoding='utf-8')
    man = {'what': 'C1 mechanism decomposition on stored records (no simulation, no rendering, no rollout)', 'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'numpy': np.__version__,
           'source_kind': kind, 'is_research_result': kind == 'C1_ORIGINAL', 'run': root.name, 'expected': exp_summary, 'frozen_plan_sha256': FROZEN,
           'code_sha256': {f: M.sha256(repo / f) for f in CODE_FILES if (repo / f).exists()}, 'settings': {k: getattr(M, k) for k in ('MOVE_MIN_M', 'PUSH_MIN_M', 'NA_NORM_M', 'RESIDUAL_TOL_M', 'TARGET_CAP_M', 'STEP_CMD_M', 'MAX_STEPS')}}
    if gate:
        incomplete_marker(out, gate); man.update(stage_failed='expected-set gate', problems=gate, status='INCOMPLETE'); return finish(out, man, {'rows': 0})
    problems = []; allrows, fun, events, mem, hold, s6rows_all, cases = [], [], [], [], [], [], []; res_all = {}; s = {}; s6 = {}
    stage_of = lambda pid: f'C1_{pid}'
    for pid in exp['policies']:
        run = M.Run(root, kind, root.name, {stage_of(pid): f'pass_{pid}'}, pid); res = M.analyse_run(run); res_all[pid] = res
        problems += M.check_keys(res['rows']) + [f'{r["branch"]}: {p}' for r in res['rows'] for p in M.check_time_order(r) + M.check_integrity(r)]
        problems += [f"{e['status']}: {e}" for e in res['errors']]
        problems += [f"{e['branch']} {e['event']} at row {e['t']}: pose missing" for e in res['events'] if e['t'] is not None and e.get('pose_missing')]
        f = M.funnel(res['rows'], list(exp['arms'])); problems += M.check_funnel(f); fun += f; allrows += res['rows']; events += res['events']; mem += res['memory']; hold += res['hold']
        s[pid] = summaries(res, [], arms=[x for x in exp['arms'] if x != M.REF])
        if exp['spatial_fields']:
            items = [(rec, run.trace(st, rec)[0]) for st, _, rec in run.scenes() if isinstance(rec, dict) and M.UNREADABLE not in rec and rec.get('scene') in {r['scene'] for r in res['rows']}]
            rows6 = S6.collect_c1(items); s6[pid] = {**S6.summarise(rows6, excluded=S6.collect_c1.excluded), 'source_kind': kind, 'policy': pid}
            if not rows6 or s6[pid]['official_S6']['status'] == 'NOT_OBSERVABLE': problems.append(f"S6_NOT_OBSERVABLE for {pid}: spatial fields were expected ({s6[pid]['official_S6']['excluded']})")
            ex = S6.collect_c1.excluded
            if ex['rows_without_contacts_or_link_poses'] or ex['scenes_without_collision_geometry']: problems.append(f'SPATIAL_FIELDS_MISSING for {pid}: {ex} (every C1 row is planned to carry them)')
            s6rows_all += [{**x, 'policy': pid} for x in rows6]
        else: s6[pid] = {'official_S6': {'status': 'NOT_EXPECTED', 'why': 'the MOCK manifest declares no spatial fields'}, 'source_kind': kind, 'policy': pid}
        for scene, why in select_cases(res['rows'], fixed=()).items():
            recp = root / f'pass_{pid}' / 'scenes' / f'{scene}.json'; rec = M.read_json(recp); frp = root / f'pass_{pid}' / str(rec.get('frames'))
            st = 'INDEXED' if rec.get('frames') and frp.is_file() and M.sha256(frp) == rec.get('frames_sha256') else 'FRAMES_NOT_RECORDED_OR_CHANGED'
            if st != 'INDEXED': problems.append(f'CASE_FRAMES_MISSING {pid} {scene}')      # both kinds (review f3)
            cases.append({'source_kind': kind, 'policy': pid, 'scene': scene, 'why': why, 'status': st, 'record': rel(recp, root), 'record_sha256': M.sha256(recp), 'trace': f"pass_{pid}/{rec.get('trace')}", 'trace_sha256': rec.get('trace_sha256'),
                          'frames': f"pass_{pid}/{rec.get('frames')}" if rec.get('frames') else None, 'frames_sha256': rec.get('frames_sha256'), 'rule': 'select_cases (fixed before the run): first scene per rule, lowest seed'})
    problems += G.row_coverage(allrows, exp, stage_of)
    fun = [{**x, 'source_kind': kind} for x in fun]
    ref = {}
    if exp.get('collector_result'):
        cr = M.read_json(root / exp['collector_result'])
        if cr.get('plan_sha256') != exp['record_plan_sha256']: problems.append(f"COLLECTOR_RESULT_OTHER_PLAN: {cr.get('plan_sha256')!r}")      # both kinds (review f2)
        for pid in exp['policies']:
            for arm, v in (((cr.get('per_policy') or {}).get(pid) or {}).get('against_C0') or {}).items(): ref[(stage_of(pid), pid, arm)] = v
            problems += [f'CROSSCHECK_MISSING {pid} {arm}: the collector summary has no paired counts for it' for arm in exp['arms'] if arm != M.REF and (stage_of(pid), pid, arm) not in ref]
        cc_rows, cc_p = G.crosscheck(fun, ref); problems += cc_p
    else: cc_rows = [{'note': 'CROSSCHECK_NOT_AVAILABLE: the MOCK manifest names no collector summary'}]
    (out / 'tables').mkdir()
    M.write_csv(out / 'scene_arm_summary.csv', allrows); M.write_csv(out / 'opportunity_funnel.csv', fun); M.write_csv(out / 'event_table.csv', events)
    M.write_csv(out / 'tables' / 'target_estimate_and_memory_decomposition.csv', mem); M.write_csv(out / 'tables' / 'held_estimate_decomposition.csv', hold)
    abl = M.executed_ablation_table(allrows); M.write_csv(out / 'tables' / 'executed_ablation_chain.csv', abl)
    M.write_csv(out / 'tables' / 'denominator_check.csv', denominator_check(fun)); M.write_csv(out / 'tables' / 'crosscheck_against_collector_summary.csv', cc_rows)
    (out / 'tables' / 's6_validity_summary.json').write_text(json.dumps(s6, indent=1, default=str), encoding='utf-8')
    if s6rows_all:
        import csv
        with gzip.open(out / 'tables' / 's6_pair_rows.csv.gz', 'wt', encoding='utf-8', newline='') as g:
            cols = sorted({k for r in s6rows_all for k in r}); w = csv.DictWriter(g, fieldnames=cols); w.writeheader(); [w.writerow({k: ('' if r.get(k) is None else r[k]) for k in cols}) for r in s6rows_all]
    M.write_csv(out / 'case_index.csv', cases)
    paired = {}
    for r in allrows: paired.setdefault((r['scene'], r['arm']), {})[r['policy']] = r.get('outcome_vs_c0')
    M.write_csv(out / 'policy_paired_outcomes.csv', [{'source_kind': kind, 'scene': sc, 'arm': arm, **{pid: v.get(pid) for pid in exp['policies']}, 'note': 'scene-paired; shared scenes are not independent samples across policies'} for (sc, arm), v in sorted(paired.items())])
    (out / 'metric_dictionary.json').write_text((HERE / 'metric_dictionary.json').read_text(encoding='utf-8'), encoding='utf-8')
    results = {'source_kind': kind, 'policies': exp['policies'], 'arms': exp['arms'], 'funnel': fun, 'denominator_check': denominator_check(fun), 'summaries': s, 'executed_ablation': abl, 's6': s6, 'cases': cases,
               'crosscheck': cc_rows, 'problems': problems, 'errors': {pid: r['errors'] for pid, r in res_all.items()}}
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
    after = hashes(inputs)
    if after != before: problems.append('INPUT_CHANGED: an input file changed during the analysis')
    man.update(inputs={'files': len(inputs), 'input_sha256_unchanged': after == before})
    if problems:
        incomplete_marker(out, problems); man.update(stage_failed='analysis', problems=problems, status='INCOMPLETE'); return finish(out, man, {'rows': len(allrows)})
    M.write_csv(out / 'claims_to_evidence.csv', R.claims_c1(results)); R.write_c1_report(out, results)
    missing = [p for p in C1_REQUIRED if not (out / p).is_file()]
    man.update(required_outputs={p: (out / p).is_file() for p in C1_REQUIRED}, problems=[f'REQUIRED_OUTPUT_MISSING: {p}' for p in missing], status='ANALYSIS_COMPLETE' if not missing else 'INCOMPLETE')
    if missing: incomplete_marker(out, man['problems'])
    finish(out, man, {'rows': len(allrows), 'cases': len(cases), 's6_pairs': len(s6rows_all)})


def main():
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('existing'); q.add_argument('--phase-b', default='runs/vision_center_v2_001/server_backup/extracted/runs/vision_center_v2_001')
    q.add_argument('--offline', default='results/vision_center_v2_c1_001/preparation/offline_replay_r4.json'); q.add_argument('--replay', required=True)
    q.add_argument('--c1-plan', default='results/vision_center_v2_c1_001/preparation_v3/plan.json'); q.add_argument('--out', required=True)
    q = s.add_parser('c1'); q.add_argument('--run', required=True); q.add_argument('--out', required=True); q.add_argument('--source-kind', default='C1_ORIGINAL', choices=['C1_ORIGINAL', 'MOCK'])
    q.add_argument('--plan', default=str(HERE.parent / 'results/vision_center_v2_c1_001/preparation_v3/plan.json'), help='C1_ORIGINAL: the frozen plan FILE; its own sha256 must equal the frozen hash in completeness.FROZEN_PLAN_SHA256 (not settable from the CLI)')
    q.add_argument('--expected', help='MOCK only: a MOCK_EXPECTED_MANIFEST json (policies, scenes, arms, record_plan_sha256)')
    a = p.parse_args(); {'existing': existing, 'c1': c1}[a.cmd](a)


if __name__ == '__main__':
    main()
