"""Stage 3B: the full-feature input, unchanged controls, the entry criteria.

Synthetic records only. These tests say nothing about ManiSkill, the base policy or whether selection works.
"""
import importlib.util, inspect
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('sklearn')
from test_stage3a_selector import world, CFG, DIM, s3

_spec = importlib.util.spec_from_file_location('stage3b', Path(__file__).resolve().parents[1] / 'scripts' / '25_stage3b_full_feature.py')
b3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(b3)


@pytest.fixture(scope='module')
def W():
    return world()


@pytest.fixture(scope='module')
def loso(W):
    return b3.run_loso(*W, CFG)


def test_the_input_is_every_saved_dimension_standardised_on_training_snapshots_only(W):
    feats, rows, episodes = W; sc = ('2D', 2001); tr_s, tr_r, te_s = s3.split(feats, rows, {sc}); model, diag = b3.fit_fold(tr_s, tr_r, CFG); pre = model['pre_full']
    F = np.stack([f for _, f in tr_s]); assert pre['dim'] == DIM and np.allclose(pre['scaler'].mean_, F.mean(0)) and pre['scaler'].n_samples_seen_ == len(F)      # unique snapshots, not repeats
    assert diag['full_input']['columns'] == DIM + 4 and 'pca' not in pre
    X = b3.design_full({'policy_feature': F[:3], 'checkpoint': np.array([12, 52, 132])}, pre); assert X.shape == (3, DIM + 4)
    assert np.allclose(X[:, :DIM], (F[:3] - F.mean(0)) / F.std(0)) and X[:, DIM:].tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    wild = {k: ({'policy_feature': v['policy_feature'] * 1e6 + 7} if k[:2] == sc else v) for k, v in feats.items()}
    m2, _ = b3.fit_fold(*s3.split(wild, rows, {sc})[:2], CFG); assert np.array_equal(pre['scaler'].mean_, m2['pre_full']['scaler'].mean_) and np.array_equal(pre['scaler'].scale_, m2['pre_full']['scaler'].scale_)
    flipped = [dict(x, y_continue=1 - x['y_continue'], y_retreat5=1 - x['y_retreat5'], joint_label=3 - x['joint_label']) if (x['stage'], x['scene']) == sc else x for x in rows]
    assert b3.predict_fold(model, te_s) == b3.predict_fold(b3.fit_fold(*s3.split(feats, flipped, {sc})[:2], CFG)[0], te_s)                                      # held-out labels play no part


@pytest.mark.parametrize('name', ['stage', 'scene', 'seed', 'host', 'path', 't_ref', 'is_cubeA_grasped', 'cube_pose', 'y_continue', 'trunk_success', 'state_history', 'queue', 'rgb_history'])
def test_nothing_but_the_saved_feature_and_the_checkpoint_enters(name):
    pre = b3.fit_full_preprocess(np.random.default_rng(0).normal(size=(30, DIM)), CFG); cols = {'policy_feature': np.zeros((2, DIM)), 'checkpoint': np.array([12, 92])}
    assert b3.design_full(cols, pre).shape == (2, DIM + 4)
    with pytest.raises(ValueError): b3.design_full({**cols, name: np.zeros(2)}, pre)
    for bad in (DIM - 1, DIM + 1, 8):                                                   # another dimension is refused, never cut or padded
        with pytest.raises(ValueError): b3.design_full({'policy_feature': np.zeros((2, bad)), 'checkpoint': cols['checkpoint']}, pre)
        with pytest.raises(ValueError): b3.fit_full_preprocess(np.zeros((30, bad)), CFG)
    with pytest.raises(ValueError): b3.design_full({'policy_feature': np.zeros((2, DIM)), 'checkpoint': np.array([12, 60])}, pre)
    src = inspect.getsource(b3.fit_fold) + inspect.getsource(b3.design_full) + inspect.getsource(b3.fit_full_preprocess); assert 'PCA' not in src.replace('No PCA', '') and 'pca' not in src


def test_the_stage3a_controls_are_unchanged_and_the_folds_rows_and_weights_are_shared(W, loso):
    feats, rows, episodes = W; preds, diags, folds, timers = loso; old_p, old_d, old_f, old_t = s3.run_loso(feats, rows, episodes, CFG)
    strip = lambda ps, models: sorted((p['stage'], p['scene'], p['checkpoint'], p['model'], p['score'], p['choose']) for p in ps if p['model'] in models)
    assert strip(preds, b3.BASE) == strip(old_p, b3.BASE) and folds == old_f and timers == old_t and len(folds) == len(episodes)
    assert {p['model'] for p in preds} == set(b3.ALL) and len(preds) == 5 * len(feats)
    for d, o in zip(diags, old_d): assert (d['training_rows'], d['training_snapshots'], d['weight_mean'], d['class_counts_joint']) == (o['training_rows'], o['training_snapshots'], o['weight_mean'], o['class_counts_joint'])
    for f in folds:
        held = (f['held_out_stage'], f['held_out_scene']); assert {(p['stage'], p['scene']) for p in preds if p['fold'] == f['fold']} == {held} and f'{held[0]}:{held[1]}' not in f['training_scenes']
    assert all(h['attempts'][0]['max_iter'] == 2000 for d in diags for h in b3.heads(d) if not h['degenerate'])


def test_scores_choices_and_the_evaluators(W, loso):
    feats, rows, episodes = W; preds, diags, folds, timers = loso; res, snap, eps = b3.evaluate(preds, rows, episodes, timers)
    for p in preds:
        if p['model'] == 'M_Q_full': assert abs(p['score'] - (p['q_retreat5'] - p['q_continue'])) < 1e-12 and p['p_00'] is None
        if p['model'] == 'M_pair_full': assert abs(p['score'] - (p['p_01'] - p['p_10'])) < 1e-12 and abs(p['p_00'] + p['p_01'] + p['p_10'] + p['p_11'] - 1) < 1e-9
        assert (p['choose'] == 'retreat5') == (p['score'] > 0)                           # a score of exactly 0 is Continue
    assert len(snap) == len(feats) and len(eps) == len(episodes) * (6 + len(b3.ALL)) and {x['repeat_used'] for x in eps} == {0}
    for scope in ('2B', '2D', 'pool'):
        e = res['cached_episode'][scope]; assert e['never']['success_scenes'] == e['never']['trunk_success_scenes'] and e['never']['scenes_intervened'] == 0
        for v in e.values(): assert v['difference_from_never'] == len(v['scenes_gained']) - len(v['scenes_lost'])
        for m in b3.ALL:
            q = res['snapshot_level'][scope][m]; assert q['blocks'] == 4 * q['snapshots']                               # one choice per snapshot for its four repeats
    # the first positive checkpoint is final and is read from the repeat-0 branch
    for x in eps:
        if x['method'] in b3.CANDIDATES and x['arm'] == 'retreat5':
            sc = (x['stage'], x['scene']); t = x['decision_checkpoint']; by = {p['checkpoint']: p['score'] for p in preds if p['model'] == x['method'] and (p['stage'], p['scene']) == sc}
            assert by[t] > 0 and all(by[u] <= 0 for u in episodes[sc]['snapshots'] if u < t) and x['y'] == episodes[sc]['r0'][t]['retreat5']['y'] and x['branch_key'].endswith(f':{t}:r0:R')
    assert res['snapshot_level']['pool']['M_Q_full']['scene_mean_difference'] > res['snapshot_level']['pool']['M_time']['scene_mean_difference']      # the synthetic signal is learnable
    with pytest.raises(ValueError): b3.evaluate([p for p in preds if p['scene'] != 1000], rows, episodes, timers)                                     # a missing scene is refused
    with pytest.raises(ValueError): b3.evaluate([p for p in preds if not (p['model'] == 'M_Q_full' and p['checkpoint'] == 12)], rows, episodes, timers)


def summary(n2b, n2d, lost2b=(), lost2d=(), never=(8, 3), timer=(8, 3)):
    def one(n, base, lost, total):
        return {'success_scenes': n, 'scenes': total, 'difference_from_never': n - base, 'scenes_lost': list(lost), 'scenes_gained': list(range(n - base + len(lost))), 'scenes_intervened': 5}
    e = {'2B': {'cand': one(n2b, never[0], lost2b, 24), 'never': one(never[0], never[0], (), 24), 'training_selected_timer': one(timer[0], never[0], (), 24)},
         '2D': {'cand': one(n2d, never[1], lost2d, 12), 'never': one(never[1], never[1], (), 12), 'training_selected_timer': one(timer[1], never[1], (), 12)}}
    e['pool'] = {k: one(e['2B'][k]['success_scenes'] + e['2D'][k]['success_scenes'], sum(never), tuple(e['2B'][k]['scenes_lost']) + tuple(e['2D'][k]['scenes_lost']), 36) for k in e['2B']}
    return {'cached_episode': e}


def test_entry_criteria_are_applied_as_written():
    g = lambda *a, complete=True, **k: b3.gate('cand', summary(*a, **k), complete)
    ok = g(9, 4); assert ok['pass'] and all(ok['criteria'].values())                                  # +1 and +1, nothing lost
    assert g(10, 5, lost2b=(1,))['pass']                                                              # one baseline success lost is still allowed
    c = g(10, 3)['criteria']; assert not c['4_net_gain_of_at_least_one_scene_in_each_stage'] and c['2_at_least_two_more_scenes_than_never_and_than_timer_over_both_stages']      # all gain in one stage
    c = g(9, 3)['criteria']; assert not c['2_at_least_two_more_scenes_than_never_and_than_timer_over_both_stages'] and c['1_not_below_never_and_timer_in_each_stage']             # one scene is not enough
    c = g(11, 2)['criteria']; assert not c['1_not_below_never_and_timer_in_each_stage'] and not c['4_net_gain_of_at_least_one_scene_in_each_stage']                               # below Never in one stage
    assert not g(10, 5, lost2b=(1,), lost2d=(2,))['criteria']['3_at_most_one_baseline_success_lost_over_both_stages']
    c = g(9, 4, timer=(9, 3))['criteria']; assert c['1_not_below_never_and_timer_in_each_stage'] and not c['2_at_least_two_more_scenes_than_never_and_than_timer_over_both_stages']  # must also beat the timer by two
    assert not g(9, 4, timer=(10, 3))['criteria']['1_not_below_never_and_timer_in_each_stage']
    assert not g(9, 4, complete=False)['pass'] and not g(8, 3)['pass']                                # equal to Never is not a pass
    assert set(ok['numbers']) == {'2B', '2D', 'pool'} and ok['numbers']['pool']['never'] == 11


def test_choice_between_two_passing_candidates_follows_the_fixed_order():
    def G(name, n, lost, interv, ok=True):
        return {'candidate': name, 'pass': ok, 'numbers': {'pool': {'candidate': n, 'scenes_lost': list(range(lost)), 'scenes_intervened': interv}}}
    assert b3.choose([G('M_Q_full', 13, 0, 9, False), G('M_pair_full', 12, 1, 9, False)]) is None
    assert b3.choose([G('M_Q_full', 13, 1, 9), G('M_pair_full', 14, 1, 20)]) == 'M_pair_full'         # more successful scenes
    assert b3.choose([G('M_Q_full', 13, 1, 9), G('M_pair_full', 13, 0, 20)]) == 'M_pair_full'         # then fewer scenes lost
    assert b3.choose([G('M_Q_full', 13, 1, 9), G('M_pair_full', 13, 1, 8)]) == 'M_pair_full'          # then fewer interventions
    assert b3.choose([G('M_pair_full', 13, 1, 9), G('M_Q_full', 13, 1, 9)]) == 'M_Q_full'             # then M_Q_full
    assert b3.choose([G('M_Q_full', 20, 0, 1, False), G('M_pair_full', 13, 1, 9)]) == 'M_pair_full'   # a failing candidate is never chosen


def test_outputs_stay_outside_inputs_and_earlier_results(tmp_path, monkeypatch):
    cfg = {'stages': {'2B': {'run': str(tmp_path / 'runs' / 'a'), 'results': str(tmp_path / 'results' / 'a')}}}; monkeypatch.chdir(tmp_path)
    for bad in (tmp_path / 'runs' / 'a' / 'x', tmp_path / 'results' / 'stage3a', tmp_path / 'results' / 'stage3a' / 'new', tmp_path / 'runs' / 'stage3a_001'):
        with pytest.raises(SystemExit): b3.outside(bad, cfg)
    b3.outside(tmp_path / 'runs' / 'stage3b_001', cfg)
    assert 'simulation' not in inspect.getsource(b3) and 'torch' not in inspect.getsource(b3) and 'make_env' not in inspect.getsource(s3)      # no simulator, no base policy
