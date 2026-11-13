"""Mock physics tests: validate control flow, NOT ManiSkill physics or GPU compatibility."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from hci.policies import QueuePolicy
from hci.simulation import Runtime,trace,compare_traces
from hci.trees import clone
from hci.io import seed_for

class MockEnv:
    def __init__(self):
        self.device='cpu';self.single_action_space=SimpleNamespace(shape=(4,))
        self._episode_rng=np.random.default_rng(1);self._elapsed_steps=torch.zeros(1,dtype=torch.int32)
        self.agent=SimpleNamespace(get_controller_state=lambda:{},set_controller_state=lambda x:None,
          controller=SimpleNamespace(controllers={'arm':SimpleNamespace(config=SimpleNamespace(
            frame='root_translation',normalize_action=True,use_target=False,pos_upper=.1,pos_lower=-.1))}),
          robot=SimpleNamespace(pose=SimpleNamespace(q=torch.tensor([[1.,0.,0.,0.]]))))
        self.p=torch.zeros((1,3))
    def obs(self):
        return {'agent':{'qpos':torch.zeros((1,9)),'qvel':torch.zeros((1,9))},
                'extra':{'tcp_pose':torch.cat([self.p,torch.tensor([[1.,0.,0.,0.]])],-1)}}
    def reset(self,seed):
        self.p=torch.zeros((1,3));self._elapsed_steps=torch.zeros(1,dtype=torch.int32)
        return self.obs(),{'success':torch.tensor([False])}
    def get_state_dict(self):return {'p':self.p.clone()}
    def set_state_dict(self,s):self.p=s['p'].clone()
    def step(self,a):
        self.p+=a[:,:3]*.1;self._elapsed_steps+=1
        return self.obs(),0.,torch.tensor([False]),torch.tensor([False]),{'success':torch.tensor([False])}

class MockPolicy(QueuePolicy):
    def __init__(self):super().__init__(8,2)
    def convert(self,raw):return clone(raw)
    def generate(self,episode,t,repeat=0):
        return np.random.default_rng(seed_for(episode,t,repeat)).uniform(-.1,.1,(8,4)).astype(np.float32)

@pytest.fixture
def rt():
    c={'env':{'horizon':40},'protocol':{'min_intervention_step':12,'decision_stride':4,'minimum_post_retreat_steps':4},
       'retreat':{'steps':6,'mode':'up','distance_m':.03,'max_translation_per_step_m':.005,'lookback_steps':8}}
    return Runtime(MockEnv(),MockPolicy(),c).reset(1)

def test_policy_and_physics_restore_together(rt):
    for _ in range(12):rt.step_base()
    snap=rt.snapshot();a=trace(rt,16);rt.restore(snap);b=trace(rt,16)
    result=compare_traces(a,b)
    assert result['state_max_abs']==result['action_max_abs']==0 and result['outcome_equal']

def test_refresh_differs_inside_chunk(rt):
    for _ in range(12):rt.step_base()
    assert len(rt.policy.queue)==4 and rt.eligible()[1]
    snap=rt.snapshot();a=rt.step_base();rt.restore(snap);rt.force(1);b=rt.step_base()
    assert not np.allclose(a,b)

def test_refresh_not_allowed_at_natural_boundary(rt):
    for _ in range(16):rt.step_base()
    assert not rt.policy.queue and not rt.eligible()[1]

def test_at_most_one_intervention(rt):
    for _ in range(12):rt.step_base()
    rt.force(1)
    with pytest.raises(ValueError):rt.force(2)

def test_retreat_consumes_original_horizon(rt):
    for _ in range(12):rt.step_base()
    rt.force(2);assert rt.t==18
    res=rt.complete();assert res['steps']==40 and res['reason']=='horizon'

def test_restore_after_other_branch(rt):
    for _ in range(12):rt.step_base()
    s=rt.snapshot();a=trace(rt,16);rt.restore(s);rt.force(2);rt.complete();rt.restore(s);b=trace(rt,16)
    assert compare_traces(a,b)['state_max_abs']==0

def test_gate_never_acts_after_first_intervention(rt):
    calls=[]
    def choose(r):calls.append(r.t);return 1
    rt.complete(choose)
    assert calls==[12]
