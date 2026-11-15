"""Stage 3A selector: fold separation, input whitelist, label mapping, cached episode reading, baselines.

Synthetic records only. These tests say nothing about ManiSkill, the base policy or whether selection works.
"""
import copy, importlib.util, json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('sklearn')

_spec = importlib.util.spec_from_file_location('stage3a', Path(__file__).resolve().parents[1] / 'scripts' / '24_stage3a_selector.py')
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)

DIM = 20
CFG = {'checkpoints': [12, 52, 92, 132], 'repeats': [0, 1, 2, 3], 'horizon': 200,
       'features': {'policy_feature_dim': DIM, 'pca_components': 8, 'pca': {'svd_solver': 'full', 'whiten': False}},
       'logistic_regression': {'solver': 'lbfgs', 'C': 1.0, 'fit_intercept': True, 'class_weight': None, 'tol': 1e-7, 'max_iter': 2000, 'retry_max_iter': 10000}}


def world(seed=0, scenes=(('2B', 14), ('2D', 8)), label=None):
    """Synthetic pool. label(stage, scene, t, r, f) -> (y_continue, y_retreat5); default: R5 helps when feature 0 is positive."""
    g = np.random.default_rng(seed); feats = {}; rows = []; episodes = {}
    label = label or (lambda st, sc, t, r, f: (int(f[0] < 0 and g.random() < .8), int(f[0] > 0 and g.random() < .8)))
    for st, n in scenes:
        for i in range(n):
            sc = (1000 if st == '2B' else 2000) + i; reached = CFG['checkpoints'][:2 + (i % 3)]; trunk = int(i % 4 == 0)
            ep = {'stage': st, 'scene': sc, 'trunk_success': trunk, 'trunk_steps': 90 if trunk else 200, 'planned_checkpoints': list(CFG['checkpoints']), 'snapshots': list(reached),
                  'skips': {t: 'trunk_succeeded_before_checkpoint' for t in CFG['checkpoints'] if t not in reached}, 'r0': {}}
            for t in reached:
                f = g.normal(size=DIM); feats[(st, sc, t)] = {'policy_feature': f}
                for r in CFG['repeats']:
                    yc, yr = label(st, sc, t, r, f); rows.append({'stage': st, 'scene': sc, 'checkpoint': t, 'snapshot_id': f'{sc}:{t}', 'repeat': r, 'y_continue': yc, 'y_retreat5': yr,
                                                                  'joint_label': 2 * yc + yr})
                    if r == 0: ep['r0'][t] = {'continue': {'y': trunk, 'steps': ep['trunk_steps']}, 'retreat5': {'y': yr, 'steps': 150, 'success_step': None, 'branch_key': f'{st}:{sc}:{t}:r0:R'}}
            episodes[(st, sc)] = ep
    return feats, rows, episodes


@pytest.fixture(scope='module')
def W():
    return world()


@pytest.fixture(scope='module')
def loso(W):
    return s3.run_loso(*W, CFG)


# 1
def test_a_scene_is_held_out_as_a_whole(W, loso):
    feats, rows, episodes = W; preds, diags, folds, timers = loso; assert len(folds) == len(episodes) == 22
    for f, d in zip(folds, diags):
        held = (f['held_out_stage'], f['held_out_scene']); assert f'{held[0]}:{held[1]}' not in f['training_scenes'] and len(f['training_scenes']) == 21
        mine = [p for p in preds if p['fold'] == f['fold']]; assert {(p['stage'], p['scene']) for p in mine} == {held}
        assert len(mine) == 3 * f['held_out_snapshots'] == 3 * len(episodes[held]['snapshots'])
        assert d['training_snapshots'] == len(feats) - f['held_out_snapshots'] and d['training_rows'] == 4 * d['training_snapshots'] and abs(d['weight_mean'] - 1) < 1e-12
    tr_s, tr_r, te_s = s3.split(feats, rows, {('2B', 1000)})
    assert all(k[:2] != ('2B', 1000) for k, _ in tr_s) and all((x['stage'], x['scene']) != ('2B', 1000) for x in tr_r) and {k[:2] for k, _ in te_s} == {('2B', 1000)}
    assert 'y_continue' not in preds[0] and 'joint_label' not in preds[0]


# 2
def test_held_out_labels_do_not_change_fit_or_prediction(W, loso):
    feats, rows, episodes = W; sc = ('2B', 1003); flipped = [dict(x, y_continue=1 - x['y_continue'], y_retreat5=1 - x['y_retreat5'], joint_label=3 - x['joint_label']) if (x['stage'], x['scene']) == sc else x for x in rows]
    a = s3.predict_fold(s3.fit_fold(*s3.split(feats, rows, {sc})[:2], CFG)[0], s3.split(feats, rows, {sc})[2])
    b = s3.predict_fold(s3.fit_fold(*s3.split(feats, flipped, {sc})[:2], CFG)[0], s3.split(feats, flipped, {sc})[2]); assert a == b and len(a) > 0
    other = s3.predict_fold(s3.fit_fold(*s3.split(feats, flipped, {('2B', 1004)})[:2], CFG)[0], s3.split(feats, flipped, {('2B', 1004)})[2])
    base = s3.predict_fold(s3.fit_fold(*s3.split(feats, rows, {('2B', 1004)})[:2], CFG)[0], s3.split(feats, rows, {('2B', 1004)})[2]); assert other != base      # training labels do matter


# 3
def test_held_out_features_do_not_touch_the_training_scaler_or_pca(W):
    feats, rows, episodes = W; sc = ('2D', 2001); wild = {k: ({'policy_feature': v['policy_feature'] * 1e6 + 7} if k[:2] == sc else v) for k, v in feats.items()}
    m1, _ = s3.fit_fold(*s3.split(feats, rows, {sc})[:2], CFG); m2, _ = s3.fit_fold(*s3.split(wild, rows, {sc})[:2], CFG); p1, p2 = m1['pre']['M_Q'], m2['pre']['M_Q']
    assert np.array_equal(p1['scaler1'].mean_, p2['scaler1'].mean_) and np.array_equal(p1['pca'].components_, p2['pca'].components_) and np.array_equal(p1['scaler2'].scale_, p2['scaler2'].scale_)
    assert p1['pca'].n_components_ == 8 and m1['pre']['M_pair'] is p1 and m1['pre']['M_time']['kind'] == 'time_only'
    # fitted on unique snapshots, not on the four repeats of each
    F = np.stack([f for _, f in s3.split(feats, rows, {sc})[0]]); assert np.allclose(p1['scaler1'].mean_, F.mean(0)) and p1['scaler1'].n_samples_seen_ == len(F)


# 4
@pytest.mark.parametrize('name', ['stage', 'scene', 'seed', 'path', 't_ref', 'is_cubeA_grasped', 'y_continue', 'trunk_success', 'retreat_steps_used', 'snapshot_id', 'repeat', 'host'])
def test_any_other_input_column_is_refused(W, name):
    feats, rows, episodes = W; pre = s3.fit_preprocess(np.stack([v['policy_feature'] for v in feats.values()]), CFG); n = 3
    cols = {'policy_feature': np.zeros((n, DIM)), 'checkpoint': np.array([12, 52, 92])}; assert s3.design(cols, pre).shape == (n, 12)
    assert s3.design(cols, {'kind': 'time_only', 'checkpoints': CFG['checkpoints']}).shape == (n, 4)
    with pytest.raises(ValueError): s3.design({**cols, name: np.zeros(n)}, pre)
    with pytest.raises(ValueError): s3.design({'policy_feature': np.zeros((n, DIM + 1)), 'checkpoint': cols['checkpoint']}, pre)       # no substitute input
    with pytest.raises(ValueError): s3.design({'policy_feature': np.zeros((n, DIM)), 'checkpoint': np.array([12, 52, 60])}, pre)
    with pytest.raises(ValueError): s3.fit_preprocess(np.zeros((8, DIM)), CFG)


# 5
def test_one_choice_per_snapshot_is_applied_to_all_four_repeats(W, loso):
    feats, rows, episodes = W; preds = loso[0]; keys = [(p['stage'], p['scene'], p['checkpoint']) for p in preds if p['model'] == 'M_Q']; assert len(keys) == len(set(keys)) == len(feats)
    choose = {(p['stage'], p['scene'], p['checkpoint']): p['choose'] for p in preds if p['model'] == 'M_Q'}; e = s3.snapshot_eval(choose, rows)
    want = sum(x['y_retreat5'] if choose[(x['stage'], x['scene'], x['checkpoint'])] == 'retreat5' else x['y_continue'] for x in rows)
    assert e['success_blocks_selected'] == want and e['blocks'] == 4 * e['snapshots'] and e['c0_fail_blocks'] + e['c0_success_blocks'] == e['blocks']
    assert s3.snapshot_eval({k: 'continue' for k in choose}, rows)['scene_mean_difference'] == 0 and s3.snapshot_eval({k: 'continue' for k in choose}, rows)['c0_success_to_selected_fail'] == 0


# 6
def test_label_mapping_and_score_signs():
    assert s3.JOINT == {0: 'both_fail', 1: 'rescue', 2: 'harm', 3: 'both_succeed'} and [s3.bit(x) for x in ('0', '1', 0, 1)] == [0, 1, 0, 1]
    for bad in ('True', 'False', '', 'yes', 2, None, True, '1.0'):
        with pytest.raises(ValueError): s3.bit(bad)
    g = np.random.default_rng(1)

    def fixed(kind):       # rescue-only world: Continue always fails, R5 always succeeds (joint 1); harm-only world: the reverse (joint 2)
        return world(3, label=lambda st, sc, t, r, f: ((0, 1) if (kind == 'rescue') == (g.random() < .9) else (1, 0)))
    for kind, sign in (('rescue', 1), ('harm', -1)):
        feats, rows, _ = fixed(kind); model, diag = s3.fit_fold(*s3.split(feats, rows, set())[:2], CFG); p = s3.predict_fold(model, s3.split(feats, rows, {('2B', 1000)})[2])
        assert {x['model'] for x in p} == set(s3.MODELS) and all(np.sign(x['score']) == sign for x in p) and all((x['choose'] == 'retreat5') == (sign > 0) for x in p)
        for x in p:
            if x['model'] == 'M_pair':
                assert abs(x['p_00'] + x['p_01'] + x['p_10'] + x['p_11'] - 1) < 1e-9 and abs(x['score'] - (x['p_01'] - x['p_10'])) < 1e-12
                assert abs(x['q_continue'] - (x['p_10'] + x['p_11'])) < 1e-12 and abs(x['q_retreat5'] - (x['p_01'] + x['p_11'])) < 1e-12 and x['p_00'] == x['p_11'] == 0
            else: assert abs(x['score'] - (x['q_retreat5'] - x['q_continue'])) < 1e-12 and x['p_00'] is None
    # a score of exactly 0 is Continue
    feats, rows, _ = world(4, label=lambda *a: (0, 0)); p = s3.predict_fold(s3.fit_fold(*s3.split(feats, rows, set())[:2], CFG)[0], s3.split(feats, rows, {('2B', 1000)})[2])
    assert all(x['score'] == 0 and x['choose'] == 'continue' for x in p)


# 7
def test_single_class_and_absent_classes():
    X = np.random.default_rng(0).normal(size=(40, 3)); w = np.ones(40)
    h, d = s3.fit_binary(X, np.ones(40), w, CFG); assert d['degenerate'] and np.all(s3.predict_binary(h, X) == 1.)
    h, d = s3.fit_binary(X, np.zeros(40), w, CFG); assert d['only_class'] == 0 and np.all(s3.predict_binary(h, X) == 0.)
    h, d = s3.fit_multi(X, np.full(40, 2), w, CFG); p = s3.predict_multi(h, X); assert d['degenerate'] and np.all(p[:, 2] == 1) and p.sum() == 40
    y = np.array([0, 3] * 20); h, d = s3.fit_multi(X, y, w, CFG); p = s3.predict_multi(h, X)
    assert d['classes'] == [0, 3] and not d['degenerate'] and np.all(p[:, 1] == 0) and np.all(p[:, 2] == 0) and np.allclose(p.sum(1), 1) and d['converged']
    y = np.array([0, 1, 3, 1] * 10); h, d = s3.fit_multi(X, y, w, CFG); p = s3.predict_multi(h, X); assert np.all(p[:, 2] == 0) and (p[:, [0, 1, 3]] > 0).all()
    h, d = s3.fit_binary(X, (X[:, 0] > 0).astype(int), w, CFG); assert not d['degenerate'] and d['converged'] and d['attempts'][0]['max_iter'] == 2000
    w2 = s3.row_weights(['a'] * 8 + ['b'] * 16, {'a': 2, 'b': 4}); assert abs(w2.mean() - 1) < 1e-12 and abs(w2[:8].sum() - w2[8:].sum()) < 1e-12      # equal total weight per scene


# 8
def test_never_is_the_trunk_and_a_timer_is_the_stored_repeat0_branch(W):
    feats, rows, episodes = W
    for k, ep in episodes.items():
        d = s3.episode_decision(ep, s3.timer_rule('never')); assert (d['y'], d['steps'], d['arm'], d['decision_checkpoint']) == (ep['trunk_success'], ep['trunk_steps'], 'continue', None)
        for t in CFG['checkpoints']:
            d = s3.episode_decision(ep, s3.timer_rule(f'timer{t}'))
            if t in ep['snapshots']: assert (d['y'], d['decision_checkpoint'], d['branch_key']) == (ep['r0'][t]['retreat5']['y'], t, ep['r0'][t]['retreat5']['branch_key'])
            else: assert d['arm'] == 'continue' and d['y'] == ep['trunk_success']          # already ended: no intervention is assumed
    eps = list(episodes.values()); name, n = s3.select_timer(eps); assert set(n) == set(s3.TIMER_ORDER) and n[name] == max(n.values())
    assert n['never'] == sum(e['trunk_success'] for e in eps) and n['timer12'] == sum(e['r0'][12]['retreat5']['y'] for e in eps)
    tie = [dict(e, trunk_success=0, r0={t: {**v, 'retreat5': {**v['retreat5'], 'y': 0}, 'continue': {'y': 0, 'steps': e['trunk_steps']}} for t, v in e['r0'].items()}) for e in eps]
    assert s3.select_timer(tie)[0] == 'never'                                                # ties: Never first


# 9
def test_the_first_positive_checkpoint_is_final(W):
    ep = next(e for e in W[2].values() if len(e['snapshots']) == 4); asked = []
    scores = {12: -1., 52: .1, 92: 9., 132: 5.}; d = s3.episode_decision(ep, lambda t: asked.append(t) or scores[t] > 0)
    assert d['decision_checkpoint'] == 52 and asked == [12, 52] and d['y'] == ep['r0'][52]['retreat5']['y']       # the larger later score is never consulted
    d = s3.episode_decision(ep, lambda t: False); assert d['arm'] == 'continue'
    short = next(e for e in W[2].values() if len(e['snapshots']) == 2); asked = []; s3.episode_decision(short, lambda t: asked.append(t) or False); assert asked == short['snapshots']


# 10
def test_other_repeats_cannot_stand_for_an_episode(W):
    ep = next(iter(W[2].values()))
    for r in (1, 2, 3):
        with pytest.raises(ValueError): s3.branch_r0(ep, 12, repeat=r)
    assert s3.branch_r0(ep, 12)['branch_key'].endswith(':12:r0:R')


# 11
def test_an_unexplained_missing_checkpoint_is_an_error(W):
    ep = next(e for e in W[2].values() if len(e['snapshots']) == 2); s3.check_episode(ep)
    bad = dict(ep, skips={92: 'trunk_succeeded_before_checkpoint'})
    with pytest.raises(ValueError): s3.episode_decision(bad, s3.timer_rule('never'))
    ok = dict(ep, skips={92: 'not_all_three_choices_eligible', 132: 'trunk_ended_before_checkpoint'}); assert s3.episode_decision(ok, s3.timer_rule('timer92'))['arm'] == 'continue'
    wrong = copy.deepcopy(ep); wrong['r0'][12]['continue']['y'] = 1 - ep['trunk_success']
    with pytest.raises(ValueError): s3.check_episode(wrong)


# 12
def test_no_gain_appears_when_both_arms_always_have_the_same_outcome():
    g = np.random.default_rng(5); feats, rows, episodes = world(6, label=lambda st, sc, t, r, f: (lambda y: (y, y))(int(g.random() < .4)))
    for ep in episodes.values():
        for t in ep['r0']: ep['r0'][t]['retreat5']['y'] = ep['trunk_success']
    preds, diags, folds, timers = s3.run_loso(feats, rows, episodes, CFG); res, snap, eps = s3.evaluate(preds, rows, episodes, timers, ['2B', '2D'])
    for scope in ('2B', '2D', 'pool'):
        for name, e in res['snapshot_level'][scope].items(): assert e['scene_mean_difference'] == 0 and e['c0_fail_to_selected_success'] == e['c0_success_to_selected_fail'] == 0
        for name, e in res['cached_episode'][scope].items(): assert e['difference_from_never'] == 0 and not e['scenes_gained'] and not e['scenes_lost']
    assert not any(res['improved_over_never_in_every_stage'].values()) and set(timers.values()) == {'never'}


# 13
def test_outputs_stay_outside_the_inputs_and_inputs_are_only_read(tmp_path):
    cfg = {'stages': {'2B': {'run': str(tmp_path / 'runs' / 'a'), 'results': str(tmp_path / 'results' / 'a')}}}
    for bad in (tmp_path / 'runs' / 'a', tmp_path / 'runs' / 'a' / 'x', tmp_path / 'results' / 'a' / 'stage3a'):
        with pytest.raises(SystemExit): s3.outside_inputs(bad, cfg)
    s3.outside_inputs(tmp_path / 'runs' / 'stage3a_001', cfg)
    p = tmp_path / 'x.json'; s3.write_json(p, {'a': 1}); s3.write_csv(tmp_path / 'x.csv', [{'a': 1}])
    with pytest.raises(FileExistsError): s3.write_json(p, {'a': 2})
    with pytest.raises(FileExistsError): s3.write_csv(tmp_path / 'x.csv', [{'a': 2}])
    d = tmp_path / 'runs' / 'a' / 'branches' / 'inputs'; d.mkdir(parents=True); x = {f't12_{k}': np.zeros(3, np.float32) for k in s3.ONLINE_FIELDS}; x['t12_t'] = np.array(12); x['t12_remaining_budget'] = np.array(188)
    np.savez(d / '7.npz', **x); h = s3.sha256(d / '7.npz'); f = s3.load_features(cfg, '2B'); assert list(f) == [('2B', 7, 12)] and s3.sha256(d / '7.npz') == h
    assert f[('2B', 7, 12)]['inputs_sha256'] == s3.tree_hash({k[4:]: v for k, v in x.items()})
    np.savez(d / '8.npz', **x, t12_cubeA_pose=np.zeros(3))
    with pytest.raises(ValueError): s3.load_features(cfg, '2B')                              # an archive with anything but the declared online fields is refused


# 14
def test_improvement_in_one_stage_is_not_improvement_in_both():
    e = lambda d: {'difference_from_never': d}
    assert s3.improved_in_both_stages({'2B': e(2), '2D': e(1)}) and not s3.improved_in_both_stages({'2B': e(3), '2D': e(0)})
    assert not s3.improved_in_both_stages({'2B': e(5), '2D': e(-1)}) and not s3.improved_in_both_stages({})


def test_learnable_synthetic_signal_is_found_and_tables_are_consistent(W, loso):
    feats, rows, episodes = W; preds, diags, folds, timers = loso; res, snap, eps = s3.evaluate(preds, rows, episodes, timers, ['2B', '2D'])
    q = res['snapshot_level']['pool']; assert q['M_Q']['scene_mean_difference'] > q['M_time']['scene_mean_difference'] and q['M_Q']['scene_mean_difference'] > 0    # feature 0 decides, time does not
    assert q['always_continue']['scene_mean_difference'] == 0 and q['always_retreat5']['snapshots_choosing_retreat5'] == len(feats)
    for scope, n in (('2B', 14), ('2D', 8), ('pool', 22)):
        assert all(e['scenes'] == n for e in res['snapshot_level'][scope].values()) and all(e['scenes'] == n for e in res['cached_episode'][scope].values())
        e = res['cached_episode'][scope]; assert e['never']['success_scenes'] == e['never']['trunk_success_scenes'] and e['never']['scenes_intervened'] == 0
        for v in e.values(): assert v['success_scenes'] - v['trunk_success_scenes'] == v['difference_from_never'] == len(v['scenes_gained']) - len(v['scenes_lost'])
    assert len(snap) == len(feats) and len(eps) == len(episodes) * 9 and {x['repeat_used'] for x in eps} == {0}
    assert all(s3.converged(d) for d in diags) and 0 < res['probabilities']['pool']['M_Q']['brier_q_continue_scene_mean'] < .25
    assert set(res['probabilities']['pool']['M_pair']['mean_predicted_label_probability']) == set(s3.JOINT.values())
    tp, td, tt = s3.run_transfer(feats, rows, episodes, CFG, '2B', '2D'); assert {p['stage'] for p in tp} == {'2D'} and td['training_scenes'] == 14 and set(tt) == {k for k in episodes if k[0] == '2D'}
    json.dumps(res)
