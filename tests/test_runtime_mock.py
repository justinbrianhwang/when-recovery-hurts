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
       'retreat':{'max_steps':6,'arrival_tolerance_m':.005,'mode':'up','distance_m':.03,'max_translation_per_step_m':.005,'lookback_steps':8}}
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
    # 3 cm target, 5 mm/step, 5 mm arrival tolerance: arrives after 5 real control steps (was a fixed 6 before Stage 1B)
    rt.force(2);assert rt.t==17
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

# --- Retreat: commanded target vs measured displacement (Stage 1B) ---
def retreat_rt(max_steps,gain=1.,distance=.03):
    c={'env':{'horizon':60},'protocol':{'min_intervention_step':12,'decision_stride':4,'minimum_post_retreat_steps':4},
       'retreat':{'max_steps':max_steps,'arrival_tolerance_m':.005,'mode':'up','distance_m':distance,
                  'max_translation_per_step_m':.005,'lookback_steps':8}}
    env=MockEnv();step=env.step
    def lagging(a):
        b=a.clone();b[:,:3]*=gain;return step(b)   # the arm achieves only `gain` of each commanded increment
    env.step=lagging
    r=Runtime(env,MockPolicy(),c).reset(1)
    for _ in range(12):r.step_base()
    return r

def test_retreat_stops_early_when_target_reached():
    r=retreat_rt(40);r.force(2);i=r.retreat_info
    # perfect tracking: 3 cm at 5 mm/step is within the 5 mm tolerance after 5 steps; the other 35 are not consumed
    assert i['reached'] and i['end_reason']=='reached' and i['steps_used']==5 and r.t==17
    assert i['target_error_m']<=.005+1e-9 and abs(i['commanded_distance_m']-.03)<1e-9

def test_retreat_reports_shortfall_instead_of_assuming_the_command_was_achieved():
    r=retreat_rt(6,gain=.4);r.force(2);i=r.retreat_info
    assert not i['reached'] and i['end_reason']=='budget_exhausted' and i['steps_used']==6 and r.t==18
    assert i['achieved_displacement_m']<i['commanded_distance_m'] and abs(i['achieved_displacement_m']-.012)<1e-6
    assert i['command_saturated_steps']==6

def test_retreat_closed_loop_reaches_target_despite_lag_within_budget():
    r=retreat_rt(40,gain=.4);r.force(2);i=r.retreat_info
    assert i['reached'] and 5<i['steps_used']<=40 and r.t==12+i['steps_used']

def test_retreat_eligibility_uses_worst_case_budget():
    r=retreat_rt(40)                      # horizon 60, t=12: 48 remaining > 40+4
    assert r.eligible()[2]
    r.c['env']['horizon']=56;r.horizon=56  # 44 remaining, not > 44
    assert r.eligible()[1] and not r.eligible()[2]

def test_legacy_fixed_step_config_is_rejected():
    r=retreat_rt(40);r.c['retreat']['steps']=12
    with pytest.raises(ValueError):r.eligible()

def test_retreat_info_is_restored_with_the_snapshot():
    r=retreat_rt(40);s=r.snapshot();r.force(2);res=r.complete()
    assert res['retreat']['steps_used']==5
    r.restore(s);assert r.retreat_info is None and r.complete()['retreat'] is None
