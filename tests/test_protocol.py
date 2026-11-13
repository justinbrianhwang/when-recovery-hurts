from pathlib import Path
import numpy as np
import pytest
from hci.io import *
from hci.decisions import *
from hci.schema import state_parts
from hci.trees import clone,maxdiff

def test_seed_key_stable_independent_of_global_rng():
    a=seed_for(1,12,0);np.random.seed(100);np.random.randn(100)
    assert seed_for(1,12,0)==a
    assert len({seed_for(1,12,r) for r in range(10)})==10

def test_splits_reject_leakage():
    with pytest.raises(ValueError):validate_splits({'splits':{'train':[1,2],'test':[2,3]}})
    assert validate_splits({'splits':{'train':[1,2],'test':[3]}})

def test_no_overwrite(tmp_path):
    p=tmp_path/'x.json';write_json(p,{'x':1})
    with pytest.raises(FileExistsError):write_json(p,{'x':2})

def test_whitelist_does_not_read_object_pose():
    o={'agent':{'qpos':np.ones((1,9)),'qvel':np.zeros((1,9))},
       'extra':{'tcp_pose':np.ones((1,7)),'object_pose':np.full((1,7),999.),'success':np.ones(1)}}
    x=np.concatenate(state_parts(o),-1)
    assert x.shape==(1,25) and np.max(x)==1

def test_tree_clone_and_diff():
    a={'x':np.ones((2,2))};b=clone(a);b['x'][0,0]=2
    assert a['x'][0,0]==1 and maxdiff(a,b)==1

def test_registry_unique_and_fair_actions():
    r=default_registry();assert len({c['id'] for c in r['candidates']})==len(r['candidates'])
    assert {c.get('action') for c in r['candidates'] if c['family']=='timer'}=={1,2}

def test_q_vs_joint_can_disagree_for_legitimate_risk_penalty():
    q=np.array([.8,.85,.9]);joint=np.array([[.1,.1,.05,.75],[.05,.25,.15,.55]])
    assert choose_action({'kind':'q','threshold':0},eligible=[1,1,1],episode=1,t=12,q=q)==2
    assert choose_action({'kind':'joint','penalty':1,'threshold':0},eligible=[1,1,1],episode=1,t=12,q=q,joint=joint)==0

def test_mask_prevents_expensive_retreat_when_budget_insufficient():
    q=np.array([.1,.2,.99])
    assert choose_action({'kind':'q','threshold':0},eligible=[1,1,0],episode=1,t=12,q=q)==1

def test_never_has_no_intervention():
    assert choose_action({'kind':'never'},eligible=[1,1,1],episode=1,t=12)==0

def test_random_trigger_reproducible():
    c={'kind':'random','probability':.2,'action':1}
    assert choose_action(c,eligible=[1,1,1],episode=1,t=12)==choose_action(c,eligible=[1,1,1],episode=1,t=12)

def test_test_locked_before_gpu_import():
    from hci.freeze import run_frozen
    with pytest.raises(PermissionError):run_frozen('does-not-exist','nothing','WRONG')

def test_configs_valid():
    root=Path(__file__).resolve().parents[1]
    for p in (root/'configs').glob('*.yaml'):load_config(p)
