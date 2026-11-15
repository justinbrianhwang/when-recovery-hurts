"""Stage 3B: one representation comparison for the Continue / R5 selector (run id stage3b_001). See amendments/AMENDMENT_010.

CPU only; no simulator, no base-policy inference. The single change against Stage 3A is the model input: all 568 saved
`policy_feature` dimensions (standardised on the training snapshots of the fold) instead of 8 principal components.
Folds, rows, weights, logistic-regression settings, single-class rules, the score > 0 rule and the two evaluators are
those of scripts/24_stage3a_selector.py, which is imported unchanged. The Stage 3A models are refitted in the same run
and must reproduce the stored Stage 3A numbers before any new candidate is evaluated.

  manifest   direction lock, hashes of inputs, code and configs; written BEFORE anything is fitted
  run        integrity, leave-one-scene-out fits, predictions (hashed before labels are joined), Stage 3A reproduction,
             the two evaluation tables, the entry-criteria decision
  verify     inputs still have the hashes of the manifest
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json
import numpy as np

_spec = importlib.util.spec_from_file_location('stage3a', Path(__file__).with_name('24_stage3a_selector.py'))
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)
read_json, write_json, write_csv, read_csv, sha256 = s3.read_json, s3.write_json, s3.write_csv, s3.read_csv, s3.sha256

BASE = s3.MODELS                                   # M_time, M_Q, M_pair: Stage 3A controls, refitted unchanged
CANDIDATES = ('M_Q_full', 'M_pair_full')
ALL = BASE + CANDIDATES
STAGES = ('2B', '2D')


# ---------------------------------------------------------------- the one change: full-feature input
def fit_full_preprocess(F, cfg):
    """StandardScaler on all saved dimensions, fitted on the unique training snapshots of one fold. No PCA."""
    from sklearn.preprocessing import StandardScaler
    F = np.asarray(F, float); dim = cfg['features']['policy_feature_dim']
    if F.ndim != 2 or F.shape[1] != dim: raise ValueError(f'policy_feature has shape {F.shape}, expected (*, {dim}); it is not cut or padded')
    return {'kind': 'full_plus_time', 'dim': dim, 'scaler': StandardScaler().fit(F), 'checkpoints': list(cfg['checkpoints'])}


def design_full(columns, pre):
    """Model input of the two candidates. Only the two whitelisted columns of Stage 3A are accepted."""
    extra = sorted(set(columns) - set(s3.ALLOWED_INPUTS))
    if extra: raise ValueError(f'refused model input columns: {extra} (allowed: {list(s3.ALLOWED_INPUTS)})')
    cps = np.asarray(columns['checkpoint']).astype(int); known = np.asarray(pre['checkpoints'])
    if not np.isin(cps, known).all(): raise ValueError('a checkpoint outside the fixed list')
    F = np.asarray(columns['policy_feature'], float)
    if F.ndim != 2 or F.shape[1] != pre['dim']: raise ValueError(f'policy_feature has shape {F.shape}, expected (*, {pre["dim"]}); it is not cut or padded')
    return np.hstack([pre['scaler'].transform(F), (cps[:, None] == known[None, :]).astype(float)])


def fit_fold(train_snaps, train_rows, cfg):
    """Stage 3A fold fit (unchanged) plus the two full-feature candidates on the same rows and weights."""
    base, diag = s3.fit_fold(train_snaps, train_rows, cfg); keys = [k for k, _ in train_snaps]; index = {k: i for i, k in enumerate(keys)}
    F = np.stack([f for _, f in train_snaps]); cps = np.array([k[2] for k in keys]); pre = fit_full_preprocess(F, cfg); per_scene = {}
    for k in keys: per_scene[k[:2]] = per_scene.get(k[:2], 0) + 1
    ri = np.array([index[(x['stage'], x['scene'], x['checkpoint'])] for x in train_rows])
    w = s3.row_weights([(x['stage'], x['scene']) for x in train_rows], per_scene, len(cfg['repeats']))
    X = design_full({'policy_feature': F, 'checkpoint': cps}, pre)[ri]
    yc = np.array([x['y_continue'] for x in train_rows]); yr = np.array([x['y_retreat5'] for x in train_rows]); yj = np.array([x['joint_label'] for x in train_rows])
    hc, dc = s3.fit_binary(X, yc, w, cfg); hr, dr = s3.fit_binary(X, yr, w, cfg); hp, dp = s3.fit_multi(X, yj, w, cfg)
    diag['M_Q_full'] = {'continue': dc, 'retreat5': dr}; diag['M_pair_full'] = dp
    diag['full_input'] = {'columns': int(X.shape[1]), 'zero_variance_features_in_training': int((pre['scaler'].var_ == 0).sum())}
    return {'base': base, 'pre_full': pre, 'M_Q_full': {'continue': hc, 'retreat5': hr}, 'M_pair_full': hp}, diag


def predict_fold(model, snaps):
    if not snaps: return []
    out = s3.predict_fold(model['base'], snaps); keys = [k for k, _ in snaps]
    X = design_full({'policy_feature': np.stack([f for _, f in snaps]), 'checkpoint': np.array([k[2] for k in keys])}, model['pre_full'])
    qc = s3.predict_binary(model['M_Q_full']['continue'], X); qr = s3.predict_binary(model['M_Q_full']['retreat5'], X); p = s3.predict_multi(model['M_pair_full'], X)
    for i, k in enumerate(keys):
        s = float(qr[i] - qc[i])
        out.append({'stage': k[0], 'scene': k[1], 'checkpoint': k[2], 'model': 'M_Q_full', 'q_continue': float(qc[i]), 'q_retreat5': float(qr[i]),
                    'p_00': None, 'p_01': None, 'p_10': None, 'p_11': None, 'score': s, 'choose': 'retreat5' if s > 0 else 'continue'})
        s = float(p[i, 1] - p[i, 2])
        out.append({'stage': k[0], 'scene': k[1], 'checkpoint': k[2], 'model': 'M_pair_full', 'q_continue': float(p[i, 2] + p[i, 3]), 'q_retreat5': float(p[i, 1] + p[i, 3]),
                    **{f'p_{n}': float(p[i, j]) for j, n in enumerate(('00', '01', '10', '11'))}, 'score': s, 'choose': 'retreat5' if s > 0 else 'continue'})
    return out


def heads(diag):
    return [diag['M_time']['continue'], diag['M_time']['retreat5'], diag['M_Q']['continue'], diag['M_Q']['retreat5'], diag['M_pair'],
            diag['M_Q_full']['continue'], diag['M_Q_full']['retreat5'], diag['M_pair_full']]


def run_loso(feats, rows, episodes, cfg):
    """The same 36 folds as Stage 3A: sorted scenes, one whole scene held out per fold."""
    scenes = sorted(episodes); preds = []; diags = []; folds = []; timers = {}
    for i, sc in enumerate(scenes):
        tr_s, tr_r, te_s = s3.split(feats, rows, {sc}); model, diag = fit_fold(tr_s, tr_r, cfg)
        timers[sc], counts = s3.select_timer([episodes[k] for k in scenes if k != sc]); ok = all(h['converged'] for h in heads(diag))
        preds += [{'evaluation': 'loso', 'fold': i, 'fit_converged': ok, **p} for p in predict_fold(model, te_s)]
        diags.append({'fold': i, 'held_out': {'stage': sc[0], 'scene': sc[1]}, 'training_selected_timer': timers[sc], 'training_timer_success_scenes': counts, **diag})
        folds.append({'fold': i, 'held_out_stage': sc[0], 'held_out_scene': sc[1], 'held_out_snapshots': len(te_s), 'training_scenes': sorted(f'{a}:{b}' for a, b in scenes if (a, b) != sc)})
    return preds, diags, folds, timers


# ---------------------------------------------------------------- evaluation (same evaluators as Stage 3A, any model list)
def evaluate(pred_rows, rows, episodes, timers, models=ALL):
    by = {m: {(p['stage'], p['scene'], p['checkpoint']): p for p in pred_rows if p['model'] == m} for m in models}; keys = sorted(by[models[0]]); keyset = set(keys)
    if any(set(by[m]) != keyset for m in models): raise ValueError('the models do not cover the same snapshots')
    scenes = sorted(episodes); mine = [x for x in rows if (x['stage'], x['scene'], x['checkpoint']) in keyset]
    if len(mine) != len(rows) or {k[:2] for k in keys} != set(scenes): raise ValueError('predictions do not cover every scene and snapshot')
    choose = {'always_continue': {k: 'continue' for k in keys}, 'always_retreat5': {k: 'retreat5' for k in keys}, **{m: {k: by[m][k]['choose'] for k in keys} for m in models}}
    dec = {name: {sc: s3.episode_decision(episodes[sc], s3.timer_rule(name)) for sc in scenes} for name in s3.TIMER_ORDER}
    dec['training_selected_timer'] = {sc: s3.episode_decision(episodes[sc], s3.timer_rule(timers[sc])) for sc in scenes}
    for m in models: dec[m] = {sc: s3.episode_decision(episodes[sc], lambda t, sc=sc, m=m: by[m][(sc[0], sc[1], t)]['score'] > 0) for sc in scenes}
    out = {'snapshot_level': {}, 'cached_episode': {}, 'probabilities': {}}
    for scope in STAGES + ('pool',):
        st = None if scope == 'pool' else scope
        out['snapshot_level'][scope] = {name: s3.snapshot_eval(c, mine, st) for name, c in choose.items()}
        out['probabilities'][scope] = {m: s3.probability_eval(list(by[m].values()), mine, st) for m in models}
        out['cached_episode'][scope] = {name: s3.episode_summary(d, episodes, st) for name, d in dec.items()}
    snap_rows = [{'stage': k[0], 'scene': k[1], 'checkpoint': k[2], **{f'{m}_score': by[m][k]['score'] for m in models}, **{f'{m}_choose': by[m][k]['choose'] for m in models},
                  'continue_successes_of_4': sum(x['y_continue'] for x in mine if (x['stage'], x['scene'], x['checkpoint']) == k),
                  'retreat5_successes_of_4': sum(x['y_retreat5'] for x in mine if (x['stage'], x['scene'], x['checkpoint']) == k)} for k in keys]
    ep_rows = []
    for sc in scenes:
        ep = episodes[sc]
        for name, d in dec.items():
            x = d[sc]; t = x['decision_checkpoint']
            ep_rows.append({'stage': sc[0], 'scene': sc[1], 'method': name, 'timer_used': timers[sc] if name == 'training_selected_timer' else '', 'decision_checkpoint': '' if t is None else t,
                            'arm': x['arm'], 'score_at_decision': by[name][(sc[0], sc[1], t)]['score'] if (name in models and t is not None) else '', 'y': x['y'], 'steps': x['steps'],
                            'branch_key': x['branch_key'], 'trunk_success': ep['trunk_success'], 'trunk_steps': ep['trunk_steps'],
                            'reached_checkpoints': ' '.join(str(t) for t in ep['snapshots']), 'repeat_used': 0})
    return out, snap_rows, ep_rows


def reproduce_stage3a(preds, res, cfg):
    """The refitted Stage 3A models against the stored Stage 3A predictions and tables. A difference stops the run."""
    old = {(x['stage'], int(x['scene']), int(x['checkpoint']), x['model']): x for x in read_csv(cfg['stage3a']['predictions']) if x['evaluation'] == 'loso'}
    new = {(p['stage'], p['scene'], p['checkpoint'], p['model']): p for p in preds if p['model'] in BASE}; worst = 0.; choice = 0
    if set(old) != set(new): return {'pass': False, 'reason': 'the prediction keys differ from Stage 3A'}
    for k, p in new.items():
        choice += p['choose'] != old[k]['choose']
        for f in ('score', 'q_continue', 'q_retreat5', 'p_00', 'p_01', 'p_10', 'p_11'):
            a, b = p[f], (None if old[k][f] == '' else float(old[k][f]))
            worst = max(worst, 0. if (a is None and b is None) else (float('inf') if (a is None or b is None) else abs(a - b)))
    ref = read_json(cfg['stage3a']['summary']); diff = []

    def same(a, b):
        if isinstance(b, dict): return isinstance(a, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in b)
        return abs(a - b) <= 1e-9 if isinstance(b, float) else a == b
    for scope in STAGES + ('pool',):
        diff += [f'probabilities/{scope}/{name}' for name in BASE if not same(res['probabilities'][scope][name], ref['probabilities'][scope][name])]
    for scope in STAGES + ('pool',):
        for name in ('never', 'timer12', 'timer52', 'timer92', 'timer132', 'training_selected_timer') + BASE:
            if res['cached_episode'][scope][name] != ref['cached_episode'][scope][name]: diff.append(f'cached_episode/{scope}/{name}')
        for name in ('always_continue', 'always_retreat5') + BASE:
            a, b = res['snapshot_level'][scope][name], ref['snapshot_level'][scope][name]
            if any((abs(a[k] - b[k]) > 1e-9 if isinstance(a[k], float) else a[k] != b[k]) for k in b): diff.append(f'snapshot_level/{scope}/{name}')
    return {'pass': not diff and choice == 0 and worst < 1e-9, 'predictions_compared': len(new), 'max_abs_score_or_probability_difference': worst, 'choices_that_differ': choice,
            'tables_that_differ': diff, 'compared_with': {k: cfg['stage3a'][k] for k in ('predictions', 'summary')}}


def gate(cand, res, complete):
    """The five operating entry criteria for one candidate, on the cached single-intervention reading. Fixed before the run."""
    e = res['cached_episode']; n = lambda scope, name: e[scope][name]['success_scenes']; c = {}
    c['1_not_below_never_and_timer_in_each_stage'] = all(n(s, cand) >= n(s, 'never') and n(s, cand) >= n(s, 'training_selected_timer') for s in STAGES)
    c['2_at_least_two_more_scenes_than_never_and_than_timer_over_both_stages'] = n('pool', cand) >= n('pool', 'never') + 2 and n('pool', cand) >= n('pool', 'training_selected_timer') + 2
    c['3_at_most_one_baseline_success_lost_over_both_stages'] = len(e['pool'][cand]['scenes_lost']) <= 1
    c['4_net_gain_of_at_least_one_scene_in_each_stage'] = all(e[s][cand]['difference_from_never'] >= 1 for s in STAGES)
    c['5_all_folds_and_checks_complete'] = bool(complete)
    return {'candidate': cand, 'pass': all(c.values()), 'criteria': c,
            'numbers': {s: {'candidate': n(s, cand), 'never': n(s, 'never'), 'training_selected_timer': n(s, 'training_selected_timer'), 'scenes': e[s][cand]['scenes'],
                            'net_gain_over_never': e[s][cand]['difference_from_never'], 'scenes_gained': e[s][cand]['scenes_gained'], 'scenes_lost': e[s][cand]['scenes_lost'],
                            'scenes_intervened': e[s][cand]['scenes_intervened']} for s in STAGES + ('pool',)}}


def choose(gates):
    """Both pass: more successful scenes, then fewer scenes lost, then fewer interventions, then M_Q_full."""
    ok = [g for g in gates if g['pass']]
    if not ok: return None
    key = lambda g: (-g['numbers']['pool']['candidate'], len(g['numbers']['pool']['scenes_lost']), g['numbers']['pool']['scenes_intervened'], CANDIDATES.index(g['candidate']))
    return sorted(ok, key=key)[0]['candidate']


# ---------------------------------------------------------------- manifest / run / verify
def input_files(cfg, c3a):
    return sorted(set(s3.input_files(c3a)) | {cfg['stage3a'][k] for k in ('manifest', 'summary', 'predictions', 'model_config')})


def outside(out, c3a):
    s3.outside_inputs(out, c3a); o = Path(out).resolve()
    for d in (Path('results/stage3a').resolve(), Path('runs/stage3a_001').resolve()):
        if o == d or d in o.parents: raise SystemExit(f'Refused: the output folder {out} is inside {d}')


def cmd_manifest(a):
    cfg = read_json(a.config); c3a = read_json(cfg['stage3a']['config']); outside(a.out, c3a); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    m3a = read_json(cfg['stage3a']['manifest']); files = input_files(cfg, c3a); missing = [f for f in files if not Path(f).exists()]; problems = list(missing)
    if sha256(cfg['stage3a']['config']) != m3a['config']['sha256']: problems.append('the Stage 3A config differs from the one Stage 3A ran with')
    if sha256(cfg['stage3a']['code']) != m3a['code'][cfg['stage3a']['code']]: problems.append('the Stage 3A code differs from the one Stage 3A ran with')
    if s3.versions() != m3a['versions']: problems.append(f"library versions differ from the Stage 3A record: {s3.versions()} vs {m3a['versions']}")
    changed = [f for f, h in m3a['inputs'].items() if not Path(f).exists() or sha256(f) != h]
    if changed: problems.append(f'Stage 3A inputs changed: {changed[:5]}')
    code = {f: sha256(f) for f in ('scripts/25_stage3b_full_feature.py', cfg['stage3a']['code'])}
    write_json(out / 'direction_lock.json', {'written_utc': datetime.now(timezone.utc).isoformat(), **{k: cfg[k] for k in (
        'question', 'nature', 'fixed', 'only_change', 'candidates', 'controls_from_stage3a', 'decision_rule', 'forbidden', 'entry_criteria', 'if_no_candidate_passes', 'if_a_candidate_passes')}})
    m = {'run_id': cfg['run_id'], 'created_utc': datetime.now(timezone.utc).isoformat(), 'nature': cfg['nature'], 'status': cfg['status'], 'state': 'READY' if not problems else 'NOT_READY',
         'problems': problems, 'config': {'path': a.config, 'sha256': sha256(a.config), 'content': cfg}, 'stage3a_config': {'path': cfg['stage3a']['config'], 'sha256': sha256(cfg['stage3a']['config']), 'content': c3a},
         'code': code, 'source_version': {'stage3a_code_identical_to_the_stage3a_run': not any('code differs' in p for p in problems)},
         'inputs': {f: sha256(f) for f in files if Path(f).exists()}, 'input_count': len(files), 'versions': s3.versions(),
         'direction_lock_sha256': sha256(out / 'direction_lock.json'), 'written_before': 'any preprocessing, fit, prediction or evaluation of this stage'}
    write_json(out / 'analysis_manifest.json', m); print(json.dumps({k: m[k] for k in ('state', 'problems', 'input_count', 'versions')}, indent=1))
    if problems: raise SystemExit(2)


def changed(m):
    bad = [f for f, h in m['inputs'].items() if not Path(f).exists() or sha256(f) != h]
    return bad, sorted(set(input_files(m['config']['content'], m['stage3a_config']['content'])) - set(m['inputs']))


def cmd_verify(a):
    m = read_json(Path(a.out) / 'analysis_manifest.json'); bad, new = changed(m)
    r = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'files': len(m['inputs']), 'changed_or_missing': bad, 'new_files_in_input_locations': new, 'unchanged': not bad and not new}
    write_json(Path(a.out) / f'{a.label}.json', r); print(json.dumps({k: r[k] for k in ('files', 'changed_or_missing', 'new_files_in_input_locations', 'unchanged')}))
    if not r['unchanged']: raise SystemExit(2)


def cmd_run(a):
    out = Path(a.out); m = read_json(out / 'analysis_manifest.json'); cfg = m['config']['content']; c3a = m['stage3a_config']['content']; bad, new = changed(m)
    if (bad or new or m['state'] != 'READY' or sha256(m['config']['path']) != m['config']['sha256'] or sha256(m['stage3a_config']['path']) != m['stage3a_config']['sha256']
            or any(sha256(f) != h for f, h in m['code'].items()) or s3.versions() != m['versions']):
        raise SystemExit(f'Refused: inputs, configs, code or library versions differ from the manifest ({bad[:3]} {new[:3]})')
    feats = {}; rows = []; episodes = {}; problems = []
    for stage in c3a['stages']:
        feats.update(s3.load_features(c3a, stage)); r, e, p = s3.load_labels(c3a, stage); rows += r; episodes.update(e); problems += p
    integ = s3.integrity(c3a, feats, rows, episodes, problems); integ['commonality'] = s3.commonality(c3a); integ['pass'] = not integ['problems'] and integ['commonality']['pass']
    write_json(out / 'input_integrity.json', integ)
    if not integ['pass']: raise SystemExit(f"Stopped: input or commonality problem {integ['problems']}")
    write_json(out / 'model_config.json', {'logistic_regression': c3a['logistic_regression'], 'weights': c3a['weights'], 'decision_rule': cfg['decision_rule'], 'candidates': cfg['candidates'],
                                           'input_of_the_candidates': {'columns': c3a['features']['policy_feature_dim'] + len(c3a['checkpoints']),
                                                                       'order': 'StandardScaler(policy_feature, all 568 dimensions) -> append checkpoint one-hot [12, 52, 92, 132]',
                                                                       'fit_on': 'unique training snapshots of the fold only'},
                                           'changed_against_stage3a': {'item': 'model input representation', 'stage3a': c3a['features']['order'], 'stage3b': cfg['only_change']},
                                           'unchanged_against_stage3a': ['folds', 'rows', 'row weights', 'estimator and its settings', 'single-class rule', 'convergence retry rule', 'score > 0 rule',
                                                                         'snapshot-level and cached-episode evaluators', 'library versions'],
                                           'note': 'standardising 568 dimensions instead of 8 components also changes the geometry of the L2 penalty; a difference in results is not '
                                                   'evidence that the PCA was the cause of the Stage 3A result', 'versions': s3.versions()})
    preds, diags, folds, timers = run_loso(feats, rows, episodes, c3a); write_json(out / 'scene_folds.json', {'scheme': 'leave-one-scene-out, identical to Stage 3A', 'folds': folds})
    write_csv(out / 'oof_predictions.csv', preds); pred_sha = sha256(out / 'oof_predictions.csv')
    write_json(out / 'oof_predictions.sha256.json', {'file': 'oof_predictions.csv', 'sha256': pred_sha, 'rows': len(preds), 'written_before': 'any held-out label was joined to these predictions'})
    failed = [{'fold': d['fold'], 'head': i} for d in diags for i, h in enumerate(heads(d)) if not h['converged']]
    write_json(out / 'fold_fit_diagnostics.json', {'loso': diags, 'fits_not_converged': failed, 'retried_heads': sum(1 for d in diags for h in heads(d) if h['retried']),
                                                   'degenerate_heads': sum(1 for d in diags for h in heads(d) if h['degenerate']),
                                                   'head_order': ['M_time/continue', 'M_time/retreat5', 'M_Q/continue', 'M_Q/retreat5', 'M_pair', 'M_Q_full/continue', 'M_Q_full/retreat5', 'M_pair_full']})
    # held-out labels enter only from here on
    res, snap_rows, ep_rows = evaluate(preds, rows, episodes, timers); rep = reproduce_stage3a(preds, res, cfg); write_json(out / 'stage3a_reproduction.json', rep)
    if not rep['pass']: raise SystemExit(f'Stopped: the Stage 3A controls were not reproduced ({rep}); this has to be explained before any candidate is read')
    write_csv(out / 'snapshot_decisions.csv', snap_rows); write_csv(out / 'cached_episode_decisions.csv', ep_rows)
    complete = integ['pass'] and not failed and len(folds) == len(episodes) == 36 and {(f['held_out_stage'], f['held_out_scene']) for f in folds} == set(episodes) and rep['pass']
    gates = [gate(c, res, complete) for c in CANDIDATES]; chosen = choose(gates)
    status = ({**cfg['status'], 'stage3b': 'COMPLETED_DEV_GATE_PASS', 'selected_candidate': chosen, 'current_configuration_next_rollout': 'REQUIRES_SEPARATE_APPROVAL_AFTER_COST_REPORT'}
              if chosen else {**cfg['status'], **cfg['if_no_candidate_passes']})
    write_json(out / 'gate_decision.json', {'criteria_definition': cfg['entry_criteria'], 'candidates': gates, 'selected_candidate': chosen, 'status': status,
                                            'scope': 'Operating criteria for the next development step, applied to data already looked at. Not a significance test, not a generalisation '
                                                     'claim, not a harm-rate guarantee.'})
    write_json(out / 'stage_separated_summary.json', {'evaluation': 'leave-one-scene-out', 'predictions_sha256': pred_sha, **res, 'fits_not_converged': failed,
                                                      'training_selected_timer_by_fold': {f'{k[0]}:{k[1]}': v for k, v in sorted(timers.items())},
                                                      'scope': ['Development comparison on scenes whose outcomes had been seen several times; not an independent validation.',
                                                                'Independent units: 24 + 12 scenes; 128 snapshots; the 512 rows are not independent.',
                                                                'The cached episode table reads stored repeat-0 paths at four fixed checkpoints; it is not a new closed-loop run.',
                                                                'Every fold has another model. No p-value, interval, calibration or risk bound.', 'The pool mixes the stages 24:12 and is auxiliary.'],
                                                      'status': status})
    write_json(out / 'run_status.json', {'completed': True, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'fits_not_converged': failed, 'predictions_sha256': pred_sha, 'status': status})
    print(json.dumps({'stage3a_reproduced': rep['pass'], 'cached_episode_success': {s: {k: v['success_scenes'] for k, v in res['cached_episode'][s].items()} for s in res['cached_episode']},
                      'gate': {g['candidate']: g['criteria'] for g in gates}, 'selected': chosen, 'not_converged': failed}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('manifest'); q.add_argument('--out', required=True); q.add_argument('--config', default='configs/stage3b_full_feature.json')
    q = s.add_parser('run'); q.add_argument('--out', required=True)
    q = s.add_parser('verify'); q.add_argument('--out', required=True); q.add_argument('--label', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
