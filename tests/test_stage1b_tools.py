"""Simulator-free parts of the Stage 1B scripts: seed bookkeeping, demo structure checks, summaries."""
import importlib.util, json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), ROOT / 'scripts' / name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


MAIN = json.loads((ROOT / 'configs/splits.json').read_text())
EVAL = json.loads((ROOT / 'configs/policy_eval_splits.json').read_text())


def test_declared_policy_eval_sets_are_disjoint_from_every_split():
    ev = load('09_eval_policy.py')
    assert ev.resolve_seeds(EVAL, 'policy_monitor', MAIN) == list(range(50000, 50020))
    assert len(ev.resolve_seeds(EVAL, 'policy_select', MAIN)) == 50
    assert len(ev.resolve_seeds(EVAL, 'policy_qualify_A', MAIN)) == len(ev.resolve_seeds(EVAL, 'policy_qualify_B', MAIN)) == 100
    assert ev.resolve_seeds(EVAL, 'policy_select', MAIN, limit=3) == [50100, 50101, 50102]


def test_policy_eval_refuses_leakage_and_unknown_sets():
    ev = load('09_eval_policy.py')
    bad = {'splits': {**EVAL['splits'], 'policy_monitor': [50000, 90000]}}       # 90000 is a Test seed
    with pytest.raises(ValueError): ev.resolve_seeds(bad, 'policy_monitor', MAIN)
    bad = {'splits': {**EVAL['splits'], 'policy_select': [50000, 50100]}}        # shared with policy_monitor
    with pytest.raises(ValueError): ev.resolve_seeds(bad, 'policy_select', MAIN)
    with pytest.raises(ValueError): ev.resolve_seeds(EVAL, 'policy_monitor', MAIN, demo_seeds=[3, 50005])
    with pytest.raises(KeyError): ev.resolve_seeds(EVAL, 'test', MAIN)
    with pytest.raises(KeyError): ev.resolve_seeds(EVAL, 'calibration', MAIN)


def test_policy_eval_summary_counts_and_interval():
    ev = load('09_eval_policy.py')
    rows = [{'success_once': int(i < 3), 'steps': 100 if i < 3 else 200, 'policy_calls': 10, 'policy_inference_seconds': 5., 'wall_seconds': 6.,
             'success_at_end': int(i < 2)} for i in range(10)]
    s = ev.summarize(rows, 200)
    assert s['n'] == 10 and s['success_once'] == 3 and s['success_at_end'] == 2 and s['mean_steps_successful'] == 100
    assert s['wilson95'][0] < .3 < s['wilson95'][1] and abs(s['seconds_per_policy_call'] - .5) < 1e-12


def test_candidate_selection_rule_and_tie_break():
    sel = load('11_select_candidate.py'); seeds = list(range(50100, 50150))
    def c(it, k): return {'iteration': it, 'success_once': k, 'n': 50, 'seeds': seeds, 'set': 'policy_select'}
    assert sel.choose([c(60000, 30), c(80000, 34), c(100000, 31)])['iteration'] == 80000
    assert sel.choose([c(60000, 34), c(80000, 34), c(100000, 31)])['iteration'] == 80000      # tie -> later iteration
    assert sel.choose([c(100000, 34), c(60000, 34), c(80000, 34)])['iteration'] == 100000     # independent of argument order
    with pytest.raises(ValueError): sel.choose([c(60000, 1), {**c(80000, 2), 'set': 'policy_qualify_A'}])
    with pytest.raises(ValueError): sel.choose([c(60000, 1), {**c(80000, 2), 'seeds': seeds[:10], 'n': 10}])
