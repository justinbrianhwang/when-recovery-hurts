"""Stage 3A: first observation-based selector between Continue and the existing R5 (run id stage3a_001). See amendments/AMENDMENT_009.

CPU only. Reads what Stage 2B and Stage 2D saved; runs no simulator, no base-policy inference, no rendering. A small
supervised selector is fitted on the saved `policy_feature` of each snapshot. Development experiment planned after the
Stage 2B/2C/2D results were seen: the scene-held-out numbers below are not an independent validation.

  manifest   hashes of every input, of this file and of the config; written BEFORE anything is fitted
  run        integrity and commonality checks, leave-one-scene-out fits, stage-transfer fits, the two evaluation tables,
             the development-pool export; refuses if an input, this file or the config differs from the manifest
  verify     inputs still have the hashes of the manifest

Separation kept in the code: `load_features` reads the online-input archives only; `load_labels` reads outcome files
only; a fold is fitted from training rows only; the out-of-fold predictions are written and hashed before any held-out
label is joined to them.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, platform, sys, warnings
import numpy as np

ALLOWED_INPUTS = ('policy_feature', 'checkpoint')          # the only columns a model may be built from
MODELS = ('M_time', 'M_Q', 'M_pair')
JOINT = {0: 'both_fail', 1: 'rescue', 2: 'harm', 3: 'both_succeed'}      # joint_label = 2*y_continue + y_retreat5
TIMER_ORDER = ('never', 'timer12', 'timer52', 'timer92', 'timer132')      # tie order of the training-selected timer
ONLINE_FIELDS = ('rgb_history', 'state_history', 'last_action', 'queue', 't', 'remaining_budget', 'policy_feature')
SHARED_CODE = ('src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/pilot.py', 'src/hci/pilot2b.py', 'src/hci/experiment.py', 'src/hci/io.py',
               'src/hci/schema.py', 'src/hci/trees.py', 'src/hci/diagnostics.py')


# ---------------------------------------------------------------- small io
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f: f.write(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))


def write_csv(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); fields = []
    for x in rows:
        for n in x:
            if n not in fields: fields.append(n)
    with p.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{n: x.get(n, '') for n in fields} for x in rows])


def bit(s):
    """Outcome fields are '0' or '1' (or the ints). Nothing else is converted."""
    if s in (0, 1) and not isinstance(s, bool): return int(s)
    if s in ('0', '1'): return int(s)
    raise ValueError(f'not a 0/1 outcome: {s!r}')


def tree_hash(x):
    """Same definition as hci.pilot.tree_hash for a flat dict of arrays (re-implemented so that no runtime module is imported)."""
    h = hashlib.sha256()
    for k in sorted(x): h.update(str(k).encode()); h.update(np.ascontiguousarray(np.asarray(x[k])).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- loaders (features and labels never share a function)
def load_features(cfg, stage):
    """Online inputs only. Returns {(stage, scene, checkpoint): {'policy_feature', 'inputs_sha256', 't', 'queue_len', 'last_action'}}."""
    out = {}; d = Path(cfg['stages'][stage]['run']) / 'branches' / 'inputs'
    for p in sorted(d.glob('*.npz')):
        scene = int(p.stem)
        with np.load(p) as z:
            cps = sorted({int(k.split('_')[0][1:]) for k in z.files})
            for t in cps:
                x = {k[len(f't{t}_'):]: z[k] for k in z.files if k.startswith(f't{t}_')}
                if set(x) != set(ONLINE_FIELDS): raise ValueError(f'{p}: checkpoint {t} holds {sorted(x)}, not the declared online fields')
                out[(stage, scene, t)] = {'policy_feature': np.asarray(x['policy_feature'], np.float64), 'dtype': str(x['policy_feature'].dtype),
                                          'inputs_sha256': tree_hash(x), 't': int(x['t']), 'queue_len': int(len(x['queue'])),
                                          'last_action': np.asarray(x['last_action'], np.float64), 'remaining_budget': int(x['remaining_budget'])}
    return out


def load_labels(cfg, stage):
    """Outcome files only: the paired CSV, cross-checked with the scene records. Returns (rows, episodes, problems)."""
    s = cfg['stages'][stage]; col = s['retreat_column']; arm = s['retreat_arm']; rows = []; episodes = {}; problems = []
    table = {}
    for x in read_csv(s['pairs']):
        k = (int(x['scene']), int(x['checkpoint']), int(x['block_repeat_id']))
        if k in table: problems.append(f'{stage}: duplicate CSV row {k}')
        table[k] = {'snapshot_id': x['snapshot_id'], 'y_continue': bit(x['y_C0']), 'y_retreat5': bit(x[col])}
    seen = set()
    for p in sorted((Path(s['results']) / 'scenes').glob('*.json')):
        rec = read_json(p); scene = int(rec['scene']); snap = {int(x['checkpoint']): x for x in rec['snapshots']}
        ep = {'stage': stage, 'scene': scene, 'trunk_success': bit(rec['trunk']['success']), 'trunk_steps': int(rec['trunk']['steps']),
              'planned_checkpoints': [int(t) for t in rec['planned_checkpoints']], 'snapshots': sorted(snap),
              'skips': {int(x['checkpoint']): x['reason'] for x in rec['skips']}, 'snapshot_id': {t: v['snapshot_id'] for t, v in snap.items()},
              'inputs_sha256': {t: v['policy_inputs_sha256'] for t, v in snap.items()}, 'queue_len': {t: int(v['queue_len']) for t, v in snap.items()},
              'trunk_duplicate_match': {int(d['checkpoint']): bool(d['match']) for d in rec['no_intervention_duplicates']}, 'r0': {},
              'role': rec['role'], 'policy_sha256': rec['policy_sha256'], 'config_sha256': rec['config_sha256']}
        for b in rec['blocks']:
            t = int(b['checkpoint']); r = int(b['repeat']); k = (scene, t, r); cell = {c['arm']: c for c in b['cells']}; c0 = cell['C0']; rt = cell[arm]
            if k in seen: problems.append(f'{stage}: duplicate block {k}')
            seen.add(k)
            if k not in table: problems.append(f'{stage}: block {k} is not in the CSV'); continue
            row = table[k]
            if (row['y_continue'], row['y_retreat5']) != (bit(c0['y']), bit(rt['y'])) or row['snapshot_id'] != b['snapshot_id']:
                problems.append(f'{stage}: CSV and scene record disagree at {k}')
            if rt['intervention'] != 2 or rt['intervention_step'] != t or c0['intervention'] is not None:
                problems.append(f'{stage}: {k} is not Continue / Runtime.force(2) at the checkpoint')
            if rt['policy_noise_repeat_id'] != r or c0['policy_noise_repeat_id'] != r: problems.append(f'{stage}: {k} left the noise key of its block')
            rows.append({'stage': stage, 'scene': scene, 'checkpoint': t, 'snapshot_id': b['snapshot_id'], 'repeat': r, **{n: row[n] for n in ('y_continue', 'y_retreat5')},
                         'joint_label': 2 * row['y_continue'] + row['y_retreat5']})
            if r == 0:
                ep['r0'][t] = {'continue': {'y': bit(c0['y']), 'steps': int(c0['steps'])},
                               'retreat5': {'y': bit(rt['y']), 'steps': int(rt['steps']), 'success_step': rt['success_step'],
                                            'branch_key': f'{stage}:{scene}:{t}:r0:{arm}'}}
        episodes[(stage, scene)] = ep
    if set(table) != seen: problems.append(f'{stage}: {len(set(table) - seen)} CSV rows have no block in the scene records')
    return rows, episodes, problems


# ---------------------------------------------------------------- integrity and commonality
def integrity(cfg, feats, rows, episodes, problems):
    out = {'problems': list(problems), 'stages': {}}; P = out['problems']; reps = cfg['repeats']; H = cfg['horizon']
    for stage, s in cfg['stages'].items():
        mine = [x for x in rows if x['stage'] == stage]; eps = [e for (st, _), e in episodes.items() if st == stage]; fk = [k for k in feats if k[0] == stage]
        snaps = sorted({(x['scene'], x['checkpoint']) for x in mine}); e = s['expected']
        got = {'scenes': len(eps), 'snapshots': len(snaps), 'blocks': len(mine), 'trunk_success': sum(x['trunk_success'] for x in eps),
               'success_continue': sum(x['y_continue'] for x in mine), 'success_retreat5': sum(x['y_retreat5'] for x in mine)}
        if got != e: P.append(f'{stage}: recount {got} differs from the reports {e}')
        if sorted((k[1], k[2]) for k in fk) != snaps: P.append(f'{stage}: saved inputs and outcome blocks do not cover the same snapshots')
        keys = [(x['scene'], x['checkpoint'], x['repeat']) for x in mine]
        if len(keys) != len(set(keys)) or set(keys) != {(a, b, r) for a, b in snaps for r in reps}: P.append(f'{stage}: blocks are not snapshots x repeats')
        if len({(x['scene'], x['checkpoint'], x['snapshot_id']) for x in mine}) != len(snaps): P.append(f'{stage}: a snapshot has more than one snapshot id')
        listed = {}
        for line in Path(s['backup_list']).read_text().splitlines():
            h, name = line.split(None, 1); listed[name.strip().lstrip('*')] = h
        files = sorted((Path(s['run']) / 'branches' / 'inputs').glob('*.npz')); bad_file = [p.as_posix() for p in files if listed.get(p.as_posix()) != sha256(p)]
        if bad_file or len(files) != len(eps): P.append(f'{stage}: input archives differ from the server backup list or are not one per scene: {bad_file}')
        sem = {'t_over_horizon': 0., 'queue_fraction': 0., 'last_action': 0.}; dims = set(); dtypes = set(); unexplained = []; dup_bad = []; r0_bad = []
        for ep in eps:
            sc = ep['scene']
            for t in ep['planned_checkpoints']:
                if t in ep['snapshots']:
                    f = feats.get((stage, sc, t))
                    if f is None: P.append(f'{stage}: scene {sc} checkpoint {t} has no saved input'); continue
                    if f['inputs_sha256'] != ep['inputs_sha256'][t]: P.append(f'{stage}: scene {sc} checkpoint {t}: saved input differs from the hash taken at the snapshot')
                    if f['t'] != t or f['queue_len'] != ep['queue_len'][t] or f['remaining_budget'] != H - t: P.append(f'{stage}: scene {sc} checkpoint {t}: t / queue of the input differ from the record')
                    v = f['policy_feature']; dims.add(v.shape); dtypes.add(f['dtype'])
                    sem['t_over_horizon'] = max(sem['t_over_horizon'], abs(v[-2] - t / H)); sem['queue_fraction'] = max(sem['queue_fraction'], abs(v[-1] - f['queue_len'] / 8))
                    sem['last_action'] = max(sem['last_action'], float(np.abs(v[-6:-2] - f['last_action']).max()))
                    if not ep['trunk_duplicate_match'].get(t): dup_bad.append((sc, t))
                    c0 = ep['r0'][t]['continue']
                    if (c0['y'], c0['steps']) != (ep['trunk_success'], ep['trunk_steps']): r0_bad.append((sc, t))
                elif ep['skips'].get(t) not in ('trunk_succeeded_before_checkpoint', 'trunk_ended_before_checkpoint', 'not_all_three_choices_eligible'): unexplained.append((sc, t))
        if unexplained: P.append(f'{stage}: checkpoints missing without a planned reason: {unexplained}')
        if dup_bad or r0_bad: P.append(f'{stage}: Continue repeat 0 does not reproduce the trunk at {dup_bad + r0_bad}')
        if dims != {(cfg['features']['policy_feature_dim'],)} or dtypes != {'float32'}: P.append(f'{stage}: policy_feature shape/dtype {dims}/{dtypes}')
        if max(sem.values()) > 1e-6: P.append(f'{stage}: the last entries of policy_feature are not last action, t/horizon, queue fraction ({sem})')
        out['stages'][stage] = {'recount': got, 'matches_reports': got == e, 'input_archives': len(files), 'input_archives_match_backup_list': not bad_file,
                                'saved_inputs_match_the_hash_taken_at_the_snapshot': True, 'policy_feature_shape': [list(d) for d in dims], 'policy_feature_dtype': sorted(dtypes),
                                'policy_feature_tail_max_abs_difference': sem, 'continue_repeat0_equals_trunk_snapshots': len(snaps) - len(set(dup_bad + r0_bad)),
                                'skipped_checkpoints': {r: sum(1 for ep in eps for v in ep['skips'].values() if v == r) for r in sorted({v for ep in eps for v in ep['skips'].values()})}}
    out['pool'] = {'scenes': len(episodes), 'snapshots': len(feats), 'blocks': len(rows)}
    return out


def commonality(cfg):
    """Are the two stages the same experiment for the common arms? Meaning is checked, not only file hashes."""
    a, b = (read_json(Path(cfg['stages'][s]['results']) / 'plan.json') for s in ('2B', '2D')); fa, fb = (read_json(Path(cfg['stages'][s]['results']) / 'observation_fields.json') for s in ('2B', '2D'))
    da, db = (read_json(Path(cfg['stages'][s]['results']) / 'dataset.json') for s in ('2B', '2D')); aud = read_json(Path(cfg['stages']['2D']['results']) / 'restore_audit.json')
    tc = read_json(Path(cfg['stages']['2D']['results']) / 'techcheck.json')['checks']; checks = {}
    checks['same_policy_weights_and_metadata'] = (a['policy']['weights_sha256'], a['policy']['policy_meta_sha256']) == (b['policy']['weights_sha256'], b['policy']['policy_meta_sha256'])
    checks['same_config_incl_horizon_retreat_protocol'] = a['config']['config_sha256'] == b['config']['config_sha256'] and a['config']['env']['horizon'] == cfg['horizon']
    checks['same_checkpoints_and_repeats'] = a['checkpoints'] == b['checkpoints'] == cfg['checkpoints'] and a['repeats'] == b['repeats'] == cfg['repeats']
    checks['same_success_definition'] = da['success_definition'] == db['success_definition'] == 'success_once_absorbing'
    checks['same_runtime_and_policy_code'] = all(a['code'].get(f) == b['code'].get(f) for f in SHARED_CODE)
    checks['same_cameras_and_online_field_definitions'] = fa['cameras'] == fb['cameras'] and fa['online_inputs'] == fb['online_inputs']
    checks['same_noise_key_rule'] = "seed_for(scene, physical_step_of_the_call" in a['noise']['seed_rule'] and "seed_for(scene, physical_step_of_the_call" in b['noise']['seed_rule']
    checks['scenes_disjoint'] = not set(a['scenes']['list']) & set(b['scenes']['list']) and set(a['scenes']['list']) <= set(b['scenes']['stage2b_scenes_excluded'])
    checks['2d_r5_is_runtime_force2'] = 'Runtime.force(2)' in b['arms']['retreat_5cm']['definition'] and bool(tc['every_r5_control_part_equals_its_reference'])
    ref = [x for s in aud['results'] for x in s['checkpoints'] if x['reached'] and x['branches']['reference_r5_control']['executed']]
    checks['2d_reference_run_leaves_state_and_noise_unchanged'] = bool(ref) and all(
        x['checks']['after_reference_r5_control_restored']['pass'] and x['checks']['after_reference_r5_control_trace']['pass'] and x['branches']['reference_r5_control']['inference_calls'] == 0
        for x in ref) and bool(tc['reference_is_control_only_and_state_verified_after_it'])
    checks['inputs_saved_before_outcomes'] = 'before any continuation' in fa['online_inputs_file'] and 'before any continuation' in fb['online_inputs_file']
    checks['both_datasets_completed'] = bool(da['completed'] and db['completed'] and da['reproducibility_ok'] and db['reproducibility_ok'])
    host = {s: {'nvidia_smi': d['runtime'].get('nvidia_smi', {}).get('stdout'), 'torch': d['runtime'].get('torch'), 'platform': d['runtime'].get('platform')} for s, d in (('2B', da), ('2D', db))}
    differ = sorted(f for f in set(a['code']) | set(b['code']) if a['code'].get(f) != b['code'].get(f))
    return {'pass': all(checks.values()), 'checks': checks, 'code_files_that_differ_between_the_stages': differ,
            'note_on_code_differences': 'the listed files are the stage-specific collectors and drivers, or are named in only one of the two plans; simulation.py (Runtime.force/retreat), policies.py (features, seeds) and '
                                        'pilot.py are byte-identical in both plans, and the Stage 2D collector calls Runtime.force(2) for R5',
            'host_and_driver': host, 'host_note': 'the stages ran on different hosts and GPU drivers; kept as stage metadata, never a model input, cause of any stage difference not separated'}


# ---------------------------------------------------------------- model input (whitelist) and fold fits
def design(columns, pre):
    """The single place where model inputs are assembled. Anything but the two whitelisted columns is refused."""
    extra = sorted(set(columns) - set(ALLOWED_INPUTS))
    if extra: raise ValueError(f'refused model input columns: {extra} (allowed: {list(ALLOWED_INPUTS)})')
    cps = np.asarray(columns['checkpoint']).astype(int); known = np.asarray(pre['checkpoints'])
    if not np.isin(cps, known).all(): raise ValueError('a checkpoint outside the fixed list')
    onehot = (cps[:, None] == known[None, :]).astype(float)
    if pre['kind'] == 'time_only': return onehot
    F = np.asarray(columns['policy_feature'], float)
    if F.ndim != 2 or F.shape[1] != pre['dim']: raise ValueError(f'policy_feature has shape {F.shape}, expected (*, {pre["dim"]}); no other input is substituted')
    Z = pre['scaler2'].transform(pre['pca'].transform(pre['scaler1'].transform(F)))
    return np.hstack([Z, onehot])


def fit_preprocess(F, cfg):
    """StandardScaler -> PCA(8) -> StandardScaler, fitted on the unique training snapshots of one fold."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    F = np.asarray(F, float); n = cfg['features']['pca_components']
    if F.shape[1] != cfg['features']['policy_feature_dim'] or F.shape[0] <= n: raise ValueError('too few training snapshots or a wrong feature dimension; nothing is substituted')
    s1 = StandardScaler().fit(F); pca = PCA(n_components=n, svd_solver=cfg['features']['pca']['svd_solver'], whiten=cfg['features']['pca']['whiten']).fit(s1.transform(F))
    s2 = StandardScaler().fit(pca.transform(s1.transform(F)))
    return {'kind': 'pc_plus_time', 'dim': F.shape[1], 'scaler1': s1, 'pca': pca, 'scaler2': s2, 'checkpoints': list(cfg['checkpoints'])}


def fit_lr(X, y, w, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    p = cfg['logistic_regression']; tried = []
    for it in (p['max_iter'], p['retry_max_iter']):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            m = LogisticRegression(C=p['C'], solver=p['solver'], fit_intercept=p['fit_intercept'], class_weight=p['class_weight'], tol=p['tol'], max_iter=it).fit(X, y, sample_weight=w)
        bad = any(issubclass(x.category, ConvergenceWarning) for x in ws); tried.append({'max_iter': it, 'n_iter': int(np.max(m.n_iter_)), 'converged': not bad})
        if not bad: break
    return m, {'attempts': tried, 'converged': tried[-1]['converged'], 'retried': len(tried) > 1}


def fit_binary(X, y, w, cfg):
    y = np.asarray(y).astype(int); cls = np.unique(y)
    if len(cls) == 1: return {'constant': float(cls[0])}, {'degenerate': True, 'only_class': int(cls[0]), 'converged': True, 'retried': False}
    m, d = fit_lr(X, y, w, cfg); return {'lr': m}, {'degenerate': False, **d}


def predict_binary(h, X):
    if 'constant' in h: return np.full(len(X), h['constant'])
    return h['lr'].predict_proba(X)[:, list(h['lr'].classes_).index(1)]


def fit_multi(X, y, w, cfg):
    y = np.asarray(y).astype(int); cls = np.unique(y)
    if len(cls) == 1: return {'constant': int(cls[0])}, {'degenerate': True, 'only_class': int(cls[0]), 'classes': [int(cls[0])], 'converged': True, 'retried': False}
    m, d = fit_lr(X, y, w, cfg); return {'lr': m}, {'degenerate': False, 'classes': [int(c) for c in cls], **d}


def predict_multi(h, X):
    """Probabilities in the fixed order 00, 01, 10, 11. A class absent from the training fold has probability 0."""
    out = np.zeros((len(X), 4))
    if 'constant' in h: out[:, h['constant']] = 1.; return out
    out[:, [int(c) for c in h['lr'].classes_]] = h['lr'].predict_proba(X); return out


def row_weights(scenes_of_rows, snaps_per_scene, repeats=4):
    w = np.array([1. / (snaps_per_scene[s] * repeats) for s in scenes_of_rows], float)
    return w / w.mean()


def fit_fold(train_snaps, train_rows, cfg):
    """train_snaps: [(key, policy_feature)] unique snapshots; train_rows: label rows of those snapshots only."""
    keys = [k for k, _ in train_snaps]; index = {k: i for i, k in enumerate(keys)}
    if len(index) != len(keys): raise ValueError('training snapshots are not unique')
    F = np.stack([f for _, f in train_snaps]); cps = np.array([k[2] for k in keys])
    pre = {'M_time': {'kind': 'time_only', 'checkpoints': list(cfg['checkpoints'])}}; pre['M_Q'] = pre['M_pair'] = fit_preprocess(F, cfg)
    per_scene = {}
    for k in keys: per_scene[k[:2]] = per_scene.get(k[:2], 0) + 1
    ri = np.array([index[(x['stage'], x['scene'], x['checkpoint'])] for x in train_rows]); w = row_weights([(x['stage'], x['scene']) for x in train_rows], per_scene, len(cfg['repeats']))
    yc = np.array([x['y_continue'] for x in train_rows]); yr = np.array([x['y_retreat5'] for x in train_rows]); yj = np.array([x['joint_label'] for x in train_rows])
    heads = {}; diag = {'training_scenes': len(per_scene), 'training_snapshots': len(keys), 'training_rows': len(train_rows), 'weight_mean': float(w.mean()),
                        'class_counts_joint': {str(c): int((yj == c).sum()) for c in range(4)}}
    for name in ('M_time', 'M_Q'):
        X = design({'policy_feature': F, 'checkpoint': cps}, pre[name])[ri]
        hc, dc = fit_binary(X, yc, w, cfg); hr, dr = fit_binary(X, yr, w, cfg); heads[name] = {'continue': hc, 'retreat5': hr}; diag[name] = {'continue': dc, 'retreat5': dr}
    X = design({'policy_feature': F, 'checkpoint': cps}, pre['M_pair'])[ri]; heads['M_pair'], diag['M_pair'] = fit_multi(X, yj, w, cfg)
    diag['pca_explained_variance_ratio'] = [float(v) for v in pre['M_Q']['pca'].explained_variance_ratio_]
    return {'pre': pre, 'heads': heads}, diag


def predict_fold(model, snaps):
    """snaps: [(key, policy_feature)]. Returns prediction rows without any label."""
    if not snaps: return []
    keys = [k for k, _ in snaps]; cols = {'policy_feature': np.stack([f for _, f in snaps]), 'checkpoint': np.array([k[2] for k in keys])}; out = []
    for name in MODELS:
        X = design(cols, model['pre'][name])
        if name == 'M_pair':
            p = predict_multi(model['heads'][name], X); qc = p[:, 2] + p[:, 3]; qr = p[:, 1] + p[:, 3]; score = p[:, 1] - p[:, 2]
        else:
            qc = predict_binary(model['heads'][name]['continue'], X); qr = predict_binary(model['heads'][name]['retreat5'], X); score = qr - qc; p = np.full((len(X), 4), np.nan)
        for i, k in enumerate(keys):
            out.append({'stage': k[0], 'scene': k[1], 'checkpoint': k[2], 'model': name, 'q_continue': float(qc[i]), 'q_retreat5': float(qr[i]),
                        **{f'p_{n}': (None if np.isnan(p[i, j]) else float(p[i, j])) for j, n in enumerate(('00', '01', '10', '11'))},
                        'score': float(score[i]), 'choose': 'retreat5' if score[i] > 0 else 'continue'})
    return out


def converged(diag):
    return all(d['converged'] for d in (diag['M_time']['continue'], diag['M_time']['retreat5'], diag['M_Q']['continue'], diag['M_Q']['retreat5'], diag['M_pair']))


# ---------------------------------------------------------------- cached episode decisions (repeat 0 only)
def branch_r0(ep, checkpoint, repeat=0):
    """Every snapshot was taken on the repeat-0 trunk, so only repeat-0 branches can stand for an episode."""
    if repeat != 0: raise ValueError('episode-level reading uses repeat 0 only: branches of other repeats do not share a past with a later checkpoint')
    return ep['r0'][checkpoint]['retreat5']


def check_episode(ep):
    for t in ep['planned_checkpoints']:
        if t not in ep['snapshots'] and ep['skips'].get(t) not in ('trunk_succeeded_before_checkpoint', 'trunk_ended_before_checkpoint', 'not_all_three_choices_eligible'):
            raise ValueError(f"scene {ep['scene']}: checkpoint {t} is missing without a planned reason; it is not filled in")
        if t in ep['snapshots'] and (ep['r0'][t]['continue']['y'], ep['r0'][t]['continue']['steps']) != (ep['trunk_success'], ep['trunk_steps']):
            raise ValueError(f"scene {ep['scene']}: Continue repeat 0 at checkpoint {t} is not the trunk")


def episode_decision(ep, wants):
    """wants(checkpoint) -> bool, asked in the order 12, 52, 92, 132 at the checkpoints the trunk reached with all choices
    eligible. The first yes takes the stored R5 repeat-0 branch and ends; later checkpoints are never consulted."""
    check_episode(ep)
    for t in sorted(ep['snapshots']):
        if wants(t):
            b = branch_r0(ep, t)
            return {'decision_checkpoint': t, 'arm': 'retreat5', 'y': b['y'], 'steps': b['steps'], 'branch_key': b['branch_key']}
    return {'decision_checkpoint': None, 'arm': 'continue', 'y': ep['trunk_success'], 'steps': ep['trunk_steps'], 'branch_key': f"{ep['stage']}:{ep['scene']}:trunk"}


def timer_rule(name):
    if name == 'never': return lambda t: False
    step = int(name[len('timer'):]); return lambda t: t == step


def select_timer(train_eps):
    """The candidate with the most successful TRAINING scenes; ties in the fixed order Never, 12, 52, 92, 132."""
    n = {name: sum(episode_decision(ep, timer_rule(name))['y'] for ep in train_eps) for name in TIMER_ORDER}
    best = max(n.values()); return next(name for name in TIMER_ORDER if n[name] == best), n


# ---------------------------------------------------------------- evaluation (labels are joined here, after predictions are fixed)
def snapshot_eval(pred, rows, stage=None):
    """pred: {(stage, scene, checkpoint): 'continue' | 'retreat5'} for one method. The chosen arm is applied to all repeats of the snapshot."""
    rows = [x for x in rows if stage is None or x['stage'] == stage]; per = {}; n01 = n10 = d0 = d1 = 0; s01 = set(); s10 = set(); chosen = {}
    for x in rows:
        k = (x['stage'], x['scene'], x['checkpoint']); arm = pred[k]; chosen[k] = arm; y = x['y_retreat5'] if arm == 'retreat5' else x['y_continue']; sc = (x['stage'], x['scene'])
        a = per.setdefault(sc, [0, 0, 0]); a[0] += y; a[1] += x['y_continue']; a[2] += 1
        d0 += x['y_continue'] == 0; d1 += x['y_continue'] == 1
        if x['y_continue'] == 0 and y == 1: n01 += 1; s01.add(sc)
        if x['y_continue'] == 1 and y == 0: n10 += 1; s10.add(sc)
    sel = [a[0] / a[2] for a in per.values()]; c0 = [a[1] / a[2] for a in per.values()]; diff = [a - b for a, b in zip(sel, c0)]
    return {'scenes': len(per), 'snapshots': len(chosen), 'blocks': len(rows), 'snapshots_choosing_retreat5': sum(v == 'retreat5' for v in chosen.values()),
            'fraction_snapshots_choosing_retreat5': sum(v == 'retreat5' for v in chosen.values()) / len(chosen) if chosen else 'NA',
            'scenes_with_any_retreat5_choice': len({k[:2] for k, v in chosen.items() if v == 'retreat5'}),
            'success_blocks_selected': sum(a[0] for a in per.values()), 'success_blocks_continue': sum(a[1] for a in per.values()),
            'scene_mean_selected': float(np.mean(sel)) if sel else 'NA', 'scene_mean_continue': float(np.mean(c0)) if c0 else 'NA',
            'scene_mean_difference': float(np.mean(diff)) if diff else 'NA', 'scenes_better': sum(d > 0 for d in diff), 'scenes_worse': sum(d < 0 for d in diff),
            'scenes_equal': sum(d == 0 for d in diff), 'c0_fail_to_selected_success': n01, 'c0_fail_blocks': int(d0), 'scenes_with_fail_to_success': len(s01),
            'c0_success_to_selected_fail': n10, 'c0_success_blocks': int(d1), 'scenes_with_success_to_fail': len(s10)}


def probability_eval(prows, rows, stage=None):
    """Brier scores of q_C and q_R, scene-equal weighted, and the mean predicted label probabilities (M_pair)."""
    lab = {}
    for x in rows:
        if stage is None or x['stage'] == stage: lab.setdefault((x['stage'], x['scene'], x['checkpoint']), []).append(x)
    per = {}
    for p in prows:
        k = (p['stage'], p['scene'], p['checkpoint'])
        if k not in lab: continue
        a = per.setdefault(k[:2], {'c': [], 'r': [], 'j': []})
        for x in lab[k]:
            a['c'].append((p['q_continue'] - x['y_continue']) ** 2); a['r'].append((p['q_retreat5'] - x['y_retreat5']) ** 2)
            if p['p_00'] is not None: a['j'].append(sum((p[f'p_{n}'] - (x['joint_label'] == j)) ** 2 for j, n in enumerate(('00', '01', '10', '11'))))
    mine = [p for p in prows if (p['stage'], p['scene'], p['checkpoint']) in lab]
    out = {'brier_q_continue_scene_mean': float(np.mean([np.mean(a['c']) for a in per.values()])), 'brier_q_retreat5_scene_mean': float(np.mean([np.mean(a['r']) for a in per.values()])),
           'mean_q_continue': float(np.mean([p['q_continue'] for p in mine])), 'mean_q_retreat5': float(np.mean([p['q_retreat5'] for p in mine])),
           'score_min': float(min(p['score'] for p in mine)), 'score_max': float(max(p['score'] for p in mine)), 'scores_positive': sum(p['score'] > 0 for p in mine),
           'always_continue': all(p['choose'] == 'continue' for p in mine), 'always_retreat5': all(p['choose'] == 'retreat5' for p in mine)}
    if mine and mine[0]['p_00'] is not None:
        out['brier_joint_multiclass_scene_mean'] = float(np.mean([np.mean(a['j']) for a in per.values()]))
        out['mean_predicted_label_probability'] = {JOINT[j]: float(np.mean([p[f'p_{n}'] for p in mine])) for j, n in enumerate(('00', '01', '10', '11'))}
    return out


def episode_summary(dec, episodes, stage=None):
    """dec: {(stage, scene): decision}. Compared with the stored trunk of the same scenes."""
    keys = sorted(k for k in dec if stage is None or k[0] == stage); lost = [k[1] for k in keys if episodes[k]['trunk_success'] == 1 and dec[k]['y'] == 0]
    gained = [k[1] for k in keys if episodes[k]['trunk_success'] == 0 and dec[k]['y'] == 1]; cp = {}
    for k in keys:
        t = dec[k]['decision_checkpoint']; cp[str(t)] = cp.get(str(t), 0) + 1
    return {'scenes': len(keys), 'success_scenes': sum(dec[k]['y'] for k in keys), 'trunk_success_scenes': sum(episodes[k]['trunk_success'] for k in keys),
            'difference_from_never': sum(dec[k]['y'] - episodes[k]['trunk_success'] for k in keys), 'scenes_intervened': sum(dec[k]['arm'] == 'retreat5' for k in keys),
            'decision_checkpoints': cp, 'scenes_gained': gained, 'scenes_lost': lost}


def improved_in_both_stages(by_stage):
    """True only if the difference from Never is positive in EVERY stage; one stage alone is not 'both'."""
    return bool(by_stage) and all(v['difference_from_never'] > 0 for v in by_stage.values())


def evaluate(pred_rows, rows, episodes, timers, stages):
    """pred_rows: label-free predictions for the evaluated snapshots; timers: {(stage, scene): training-selected timer name}."""
    by_model = {m: {(p['stage'], p['scene'], p['checkpoint']): p for p in pred_rows if p['model'] == m} for m in MODELS}; keys = sorted(by_model['M_Q'])
    scenes = sorted({k[:2] for k in episodes if k[0] in stages}); out = {'snapshot_level': {}, 'cached_episode': {}, 'probabilities': {}}; snap_rows = []; ep_rows = []
    choose = {'always_continue': {k: 'continue' for k in keys}, 'always_retreat5': {k: 'retreat5' for k in keys}, **{m: {k: by_model[m][k]['choose'] for k in keys} for m in MODELS}}
    dec = {name: {sc: episode_decision(episodes[sc], timer_rule(name)) for sc in scenes} for name in TIMER_ORDER}
    dec['training_selected_timer'] = {sc: episode_decision(episodes[sc], timer_rule(timers[sc])) for sc in scenes}
    for m in MODELS: dec[m] = {sc: episode_decision(episodes[sc], lambda t, sc=sc, m=m: by_model[m][(sc[0], sc[1], t)]['score'] > 0) for sc in scenes}
    keyset = set(keys); mine = [x for x in rows if (x['stage'], x['scene'], x['checkpoint']) in keyset]
    for scope in list(stages) + (['pool'] if len(stages) > 1 else []):
        st = None if scope == 'pool' else scope
        out['snapshot_level'][scope] = {name: snapshot_eval(c, mine, st) for name, c in choose.items()}
        out['probabilities'][scope] = {m: probability_eval(list(by_model[m].values()), mine, st) for m in MODELS}
        out['cached_episode'][scope] = {name: episode_summary(d, episodes, st) for name, d in dec.items()}
    out['improved_over_never_in_every_stage'] = {name: improved_in_both_stages({s: out['cached_episode'][s][name] for s in stages}) for name in dec if name != 'never'}
    for k in keys:
        snap_rows.append({'stage': k[0], 'scene': k[1], 'checkpoint': k[2], **{f'{m}_score': by_model[m][k]['score'] for m in MODELS}, **{f'{m}_choose': by_model[m][k]['choose'] for m in MODELS},
                          'continue_successes_of_4': sum(x['y_continue'] for x in mine if (x['stage'], x['scene'], x['checkpoint']) == k),
                          'retreat5_successes_of_4': sum(x['y_retreat5'] for x in mine if (x['stage'], x['scene'], x['checkpoint']) == k)})
    for sc in scenes:
        ep = episodes[sc]
        for name, d in dec.items():
            x = d[sc]; t = x['decision_checkpoint']
            ep_rows.append({'stage': sc[0], 'scene': sc[1], 'method': name, 'timer_used': timers[sc] if name == 'training_selected_timer' else '', 'decision_checkpoint': '' if t is None else t,
                            'arm': x['arm'], 'score_at_decision': by_model[name][(sc[0], sc[1], t)]['score'] if (name in MODELS and t is not None) else '',
                            'y': x['y'], 'steps': x['steps'], 'branch_key': x['branch_key'], 'trunk_success': ep['trunk_success'], 'trunk_steps': ep['trunk_steps'],
                            'reached_checkpoints': ' '.join(str(t) for t in ep['snapshots']), 'repeat_used': 0})
    return out, snap_rows, ep_rows


# ---------------------------------------------------------------- the three fits
def split(feats, rows, held):
    """held: set of (stage, scene). Training gets nothing of a held scene: no snapshot, no repeat, no label."""
    tr_s = [(k, v['policy_feature']) for k, v in sorted(feats.items()) if k[:2] not in held]; te_s = [(k, v['policy_feature']) for k, v in sorted(feats.items()) if k[:2] in held]
    tr_r = [x for x in rows if (x['stage'], x['scene']) not in held]
    if {k[:2] for k, _ in tr_s} & held or {(x['stage'], x['scene']) for x in tr_r} & held: raise RuntimeError('a held-out scene is in the training fold')
    return tr_s, tr_r, te_s


def run_loso(feats, rows, episodes, cfg):
    scenes = sorted(episodes); preds = []; diags = []; folds = []; timers = {}
    for i, sc in enumerate(scenes):
        tr_s, tr_r, te_s = split(feats, rows, {sc}); model, diag = fit_fold(tr_s, tr_r, cfg)
        timers[sc], counts = select_timer([episodes[k] for k in scenes if k != sc])
        preds += [{'evaluation': 'loso', 'fold': i, 'fit_converged': converged(diag), **p} for p in predict_fold(model, te_s)]
        diags.append({'fold': i, 'held_out': {'stage': sc[0], 'scene': sc[1]}, 'training_selected_timer': timers[sc], 'training_timer_success_scenes': counts, **diag})
        folds.append({'fold': i, 'held_out_stage': sc[0], 'held_out_scene': sc[1], 'held_out_snapshots': len(te_s), 'training_scenes': sorted(f'{a}:{b}' for a, b in scenes if (a, b) != sc)})
    return preds, diags, folds, timers


def run_transfer(feats, rows, episodes, cfg, src, dst):
    held = {k for k in episodes if k[0] != src}; tr_s, tr_r, te_s = split(feats, rows, held); model, diag = fit_fold(tr_s, tr_r, cfg)
    te_s = [x for x in te_s if x[0][0] == dst]; name, counts = select_timer([e for k, e in sorted(episodes.items()) if k[0] == src])
    preds = [{'evaluation': f'transfer_{src}_to_{dst}', 'fold': 0, 'fit_converged': converged(diag), **p} for p in predict_fold(model, te_s)]
    return preds, {'train_stage': src, 'evaluated_stage': dst, 'training_selected_timer': name, 'training_timer_success_scenes': counts, **diag}, {k: name for k in episodes if k[0] == dst}


def export_model(model, diag, cfg, name):
    """Plain numbers of one development-pool fit. Not an evaluated model."""
    pre = model['pre'][name]; h = model['heads'][name]
    lr = lambda x: {'constant': x['constant']} if 'constant' in x else {'classes': [int(c) for c in x['lr'].classes_], 'coef': x['lr'].coef_.tolist(), 'intercept': x['lr'].intercept_.tolist()}
    return {'label': 'DEV_ONLY_NOT_INDEPENDENTLY_EVALUATED', 'model': name, 'definition': cfg['models'][name], 'decision_rule': cfg['decision_rule'],
            'fitted_on': 'all 36 development scenes of Stage 2B and 2D (the same scenes whose leave-one-scene-out results were already seen)',
            'input': {'policy_feature_dim': pre['dim'], 'checkpoints': pre['checkpoints'], 'order': cfg['features']['order']},
            'scaler1': {'mean': pre['scaler1'].mean_.tolist(), 'scale': pre['scaler1'].scale_.tolist()},
            'pca': {'mean': pre['pca'].mean_.tolist(), 'components': pre['pca'].components_.tolist()},
            'scaler2': {'mean': pre['scaler2'].mean_.tolist(), 'scale': pre['scaler2'].scale_.tolist()},
            'heads': lr(h) if name == 'M_pair' else {k: lr(v) for k, v in h.items()}, 'fit_diagnostics': diag[name], 'training': {k: diag[k] for k in ('training_scenes', 'training_snapshots', 'training_rows')}}


# ---------------------------------------------------------------- manifest / run / verify
def input_files(cfg):
    files = []
    for s in cfg['stages'].values():
        res = Path(s['results']); run = Path(s['run'])
        files += [Path(s['pairs']), Path(s['backup_list']), res / 'plan.json', res / 'observation_fields.json', res / 'dataset.json', res / 'restore_audit.json', res / 'techcheck.json']
        files += sorted((res / 'scenes').glob('*.json')) + sorted((run / 'branches' / 'inputs').glob('*.npz'))
    return sorted({p.as_posix() for p in files})


def input_dirs(cfg):
    return [Path(s[k]).resolve() for s in cfg['stages'].values() for k in ('run', 'results')]


def outside_inputs(out, cfg):
    o = Path(out).resolve()
    for d in input_dirs(cfg):
        if o == d or d in o.parents: raise SystemExit(f'Refused: the output folder {out} is inside the input location {d}')


def versions():
    import sklearn, scipy
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'scipy': scipy.__version__, 'platform': platform.platform()}


def cmd_manifest(a):
    cfg = read_json(a.config); outside_inputs(a.out, cfg); out = Path(a.out); out.mkdir(parents=True, exist_ok=False); files = input_files(cfg)
    missing = [f for f in files if not Path(f).exists()]; n = {s: len(list((Path(v['run']) / 'branches' / 'inputs').glob('*.npz'))) for s, v in cfg['stages'].items()}
    short = [f"{v['run']}/branches/inputs (found {n[s]} of {v['expected']['scenes']})" for s, v in cfg['stages'].items() if n[s] != v['expected']['scenes']]
    m = {'run_id': cfg['run_id'], 'created_utc': datetime.now(timezone.utc).isoformat(), 'nature': cfg['nature'], 'status': cfg['status'],
         'state': 'PARTIAL_INPUT_UNAVAILABLE' if (missing or short) else 'INPUTS_PRESENT', 'missing': missing + short,
         'config': {'path': a.config, 'sha256': sha256(a.config), 'content': cfg}, 'code': {'scripts/24_stage3a_selector.py': sha256('scripts/24_stage3a_selector.py')},
         'inputs': {f: sha256(f) for f in files if Path(f).exists()}, 'input_count': len(files), 'versions': versions(),
         'written_before': 'any preprocessing, fit, prediction or evaluation of this stage'}
    write_json(out / 'analysis_manifest.json', m); print(json.dumps({k: m[k] for k in ('state', 'missing', 'input_count', 'versions')}, indent=1))
    if m['state'] != 'INPUTS_PRESENT': raise SystemExit(2)


def changed(m):
    bad = [f for f, h in m['inputs'].items() if not Path(f).exists() or sha256(f) != h]
    new = sorted(set(input_files(m['config']['content'])) - set(m['inputs']))
    return bad, new


def cmd_verify(a):
    m = read_json(Path(a.out) / 'analysis_manifest.json'); bad, new = changed(m)
    r = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'files': len(m['inputs']), 'changed_or_missing': bad, 'new_files_in_input_locations': new, 'unchanged': not bad and not new}
    write_json(Path(a.out) / f'{a.label}.json', r); print(json.dumps({k: r[k] for k in ('files', 'changed_or_missing', 'new_files_in_input_locations', 'unchanged')}))
    if not r['unchanged']: raise SystemExit(2)


def cmd_run(a):
    out = Path(a.out); m = read_json(out / 'analysis_manifest.json'); cfg = m['config']['content']; bad, new = changed(m)
    if bad or new or sha256(m['config']['path']) != m['config']['sha256'] or any(sha256(f) != h for f, h in m['code'].items()) or m['state'] != 'INPUTS_PRESENT':
        raise SystemExit(f'Refused: inputs, config or code differ from the manifest ({bad[:3]} {new[:3]})')
    if versions() != m['versions']: raise SystemExit('Refused: library versions differ from the manifest')
    feats = {}; rows = []; episodes = {}; problems = []
    for stage in cfg['stages']:
        feats.update(load_features(cfg, stage)); r, e, p = load_labels(cfg, stage); rows += r; episodes.update(e); problems += p
    integ = integrity(cfg, feats, rows, episodes, problems); integ['commonality'] = commonality(cfg)
    integ['pass'] = not integ['problems'] and integ['commonality']['pass']; write_json(out / 'input_integrity.json', integ)
    print(json.dumps({'integrity_pass': integ['pass'], 'problems': integ['problems'], 'commonality': integ['commonality']['checks'], 'pool': integ['pool']}, indent=1))
    if not integ['pass']: raise SystemExit('Stopped: unresolved input or commonality problem; the stages are not combined')
    write_json(out / 'feature_schema.json', {'model_input_columns': list(ALLOWED_INPUTS), 'policy_feature': {'dim': cfg['features']['policy_feature_dim'], 'dtype': 'float32',
               'content': 'fixed encoder output of the frozen policy for the snapshot history, then last action (4), t/horizon, queue fraction', 'source': cfg['features']['source']},
               'model_vector': {'M_time': '4 = checkpoint one-hot [12, 52, 92, 132]', 'M_Q': '12 = 8 standardized PC scores + checkpoint one-hot', 'M_pair': 'same 12 as M_Q'},
               'preprocessing': cfg['features']['order'], 'fit_on': cfg['features']['fit_on'],
               'never_an_input': ['scene / episode id, seed, stage, host, driver, file name, path, hash, snapshot id, repeat id', 'cube pose, grasp / on-cube / static flag, stage label',
                                  'success or failure, trunk outcome, success step, final state, any arm outcome or count', 'T_ref, R5 duration, achieved displacement, state after the return, future trace or noise'],
               'read_for_integrity_only': ['rgb_history', 'state_history', 'last_action', 'queue', 't', 'remaining_budget']})
    write_json(out / 'model_config.json', {'logistic_regression': cfg['logistic_regression'], 'models': cfg['models'], 'decision_rule': cfg['decision_rule'], 'weights': cfg['weights'],
                                           'features': cfg['features'], 'versions': versions(), 'single_class_rule': 'binary head: the constant training label; multinomial: classes '
                                           'seen in training only, absent classes get probability 0', 'convergence_rule': 'one retry with max_iter 10000 for every model; a fit that still '
                                           'does not converge is recorded as failed, never dropped'})
    # leave-one-scene-out and the two stage-transfer fits; predictions carry no label
    preds, diags, folds, timers = run_loso(feats, rows, episodes, cfg); write_json(out / 'scene_folds.json', {'scheme': 'leave-one-scene-out', 'folds': folds})
    tr = {}; ttimers = {}; tpred = {}
    for src, dst in (('2B', '2D'), ('2D', '2B')):
        tpred[dst], tr[f'{src}_to_{dst}'], ttimers[dst] = run_transfer(feats, rows, episodes, cfg, src, dst)
    write_csv(out / 'oof_predictions.csv', preds + tpred['2D'] + tpred['2B']); pred_sha = sha256(out / 'oof_predictions.csv')
    write_json(out / 'oof_predictions.sha256.json', {'file': 'oof_predictions.csv', 'sha256': pred_sha, 'rows': len(preds) + len(tpred['2D']) + len(tpred['2B']),
                                                     'written_before': 'any held-out label was joined to these predictions'})
    failed = [d['fold'] for d in diags if not converged(d)] + [k for k, d in tr.items() if not converged(d)]
    write_json(out / 'fold_fit_diagnostics.json', {'loso': diags, 'transfer': tr, 'fits_not_converged': failed,
                                                   'retried_fits': sum(1 for d in diags + list(tr.values()) for h in (d['M_time']['continue'], d['M_time']['retreat5'], d['M_Q']['continue'], d['M_Q']['retreat5'], d['M_pair']) if h['retried']),
                                                   'degenerate_heads': sum(1 for d in diags + list(tr.values()) for h in (d['M_time']['continue'], d['M_time']['retreat5'], d['M_Q']['continue'], d['M_Q']['retreat5'], d['M_pair']) if h['degenerate'])})
    # evaluation: held-out labels enter only from here on
    res, snap_rows, ep_rows = evaluate(preds, rows, episodes, timers, list(cfg['stages'])); t_res = {}; t_snap = []; t_ep = []
    for dst, src in (('2D', '2B'), ('2B', '2D')):
        r, s, e = evaluate(tpred[dst], rows, episodes, ttimers[dst], [dst]); t_res[f'{src}_to_{dst}'] = r
        t_snap += [{'evaluation': f'transfer_{src}_to_{dst}', **x} for x in s]; t_ep += [{'evaluation': f'transfer_{src}_to_{dst}', **x} for x in e]
    write_csv(out / 'snapshot_decisions.csv', [{'evaluation': 'loso', **x} for x in snap_rows] + t_snap)
    write_csv(out / 'cached_episode_decisions.csv', [{'evaluation': 'loso', **x} for x in ep_rows] + t_ep)
    scope = ['Development experiment on scenes whose branch outcomes had been seen several times; the scene-held-out numbers are not an independent validation.',
             'Rows of one snapshot share the input; 512 rows are not 512 independent scenes. Independent units: 24 + 12 scenes.',
             'The cached episode table reads stored repeat-0 paths at four fixed checkpoints; it is not a new closed-loop run.',
             'Every fold has another model; no table here is the validation of one final model.', 'No p-value, interval, calibration or risk bound was computed.',
             'The 36-scene pool mixes the stages 24:12 and is an auxiliary description only.']
    write_json(out / 'stage_separated_summary.json', {'evaluation': 'leave-one-scene-out', 'predictions_sha256': pred_sha, 'training_selected_timer_by_fold': {f'{k[0]}:{k[1]}': v for k, v in sorted(timers.items())},
                                                      **res, 'fits_not_converged': failed, 'scope': scope, 'status': {**cfg['status'], 'observation_based_selector': 'TRAINED_DEV_ONLY'}})
    write_json(out / 'stage_transfer_summary.json', {'evaluation': 'fit on one stage, read on the other (both stages already seen; scene, host and stage differences are confounded)',
                                                     'predictions_sha256': pred_sha, **t_res, 'scope': scope})
    # development-pool export, after everything above
    model, diag = fit_fold(*split(feats, rows, set())[:2], cfg)
    for name in ('M_Q', 'M_pair'): write_json(out / 'models' / f'{name}_dev_pool.json', export_model(model, diag, cfg, name))
    write_json(out / 'run_status.json', {'completed': True, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'fits_not_converged': failed, 'predictions_sha256': pred_sha,
                                         'status': {**cfg['status'], 'observation_based_selector': 'TRAINED_DEV_ONLY'}})
    brief = lambda r: {s: {k: (v['success_scenes'], v['scenes']) for k, v in r['cached_episode'][s].items()} for s in r['cached_episode']}
    print(json.dumps({'loso_cached_episode_success': brief(res), 'transfer': {k: brief(v) for k, v in t_res.items()}, 'fits_not_converged': failed}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('manifest'); q.add_argument('--out', required=True); q.add_argument('--config', default='configs/stage3a_selector.json')
    q = s.add_parser('run'); q.add_argument('--out', required=True)
    q = s.add_parser('verify'); q.add_argument('--out', required=True); q.add_argument('--label', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
