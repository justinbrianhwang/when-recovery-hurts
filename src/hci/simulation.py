from __future__ import annotations
from collections import deque
import copy, random, time
import numpy as np
import torch
from .trees import clone, as_numpy, maxdiff

RNG_ATTRS=('_main_rng','_episode_rng','_batched_main_rng','_batched_episode_rng')

def make_env(c):
    import gymnasium as gym
    import mani_skill.envs  # registration only
    e=c['env']
    # Unwrap deliberately: our runner owns the horizon; no autoreset or hidden TimeLimit state.
    return gym.make(e['id'],num_envs=1,sim_backend='physx_cpu',obs_mode=e['obs_mode'],
                    control_mode=e['control_mode'],robot_uids=e['robot_uid'],reward_mode='sparse',
                    render_mode='rgb_array',reconfiguration_freq=1).unwrapped

def rng_state(v):
    # ManiSkill's BatchedRNG forwards __reduce_ex__ to its members, so copy.deepcopy raises.
    # Save generator *states* and restore in place; this also keeps _episode_rng aliased to _batched_episode_rng[0].
    if type(v).__name__=='BatchedRNG':return ('batched',[r.get_state() for r in v.rngs])
    if isinstance(v,np.random.RandomState):return ('random_state',v.get_state())
    return ('object',copy.deepcopy(v))

def set_rng_state(env,name,saved):
    kind,value=saved
    if kind=='object':setattr(env,name,copy.deepcopy(value));return
    cur=getattr(env,name)
    if kind=='batched':
        if type(cur).__name__!='BatchedRNG' or len(cur.rngs)!=len(value):raise ValueError(f'RNG layout changed for {name}')
        for r,x in zip(cur.rngs,value):r.set_state(x)
    else:
        if not isinstance(cur,np.random.RandomState):raise ValueError(f'RNG type changed for {name}')
        cur.set_state(value)

def save_env(env):
    out={'state':clone(env.get_state_dict()),'controller':clone(env.agent.get_controller_state()),
         'elapsed':clone(env._elapsed_steps),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
         'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
         'env_rng':{a:rng_state(getattr(env,a)) for a in RNG_ATTRS if hasattr(env,a)}}
    return out

def restore_env(env,s,seed,actions,atol):
    """Restore by prefix action replay: reset(seed), re-execute the logged actions, verify, then restore RNGs.

    env.set_state_dict is NOT used. Measured on the pinned ManiSkill/PhysX CPU backend: rewriting rigid-actor
    state (even with identical values) changes the next contact solve (7e-5 after one step, 6e-3 after eight),
    while reset(seed)+same actions is bit-identical. Valid only for the SAME seed/scene.
    """
    env.reset(seed=int(seed))
    for a in actions:env.step(torch.as_tensor(np.asarray(a,np.float32)[None],device=env.device))
    err=maxdiff(clone(env.get_state_dict()),s['state'])
    if err>atol:raise RuntimeError(f'Prefix replay did not reproduce the snapshot state (max abs {err} > {atol})')
    if maxdiff(env._elapsed_steps,s['elapsed'])!=0:raise RuntimeError('Prefix replay step counter mismatch')
    for a,v in s['env_rng'].items():set_rng_state(env,a,v)
    random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'])
    if s['cuda_rng'] is not None:torch.cuda.set_rng_state_all(s['cuda_rng'])
    return err

def success(info):
    if 'success' not in info:raise KeyError('Task must expose an explicit success signal')
    a=as_numpy(info['success']).reshape(-1)
    if len(a)!=1:raise ValueError('Single environment required')
    return bool(a[0])

class Runtime:
    def __init__(self,env,policy,c):
        self.env=env;self.policy=policy;self.c=c
        self.horizon=c['env']['horizon'];self.adim=env.single_action_space.shape[0]
    def reset(self,episode):
        self.episode=int(episode);self.repeat=0;self.t=0;self.intervention=None;self.intervention_t=None;self.actions=[]
        self.raw,info=self.env.reset(seed=self.episode)
        self.done=success(info);self.y=int(self.done);self.reason='success' if self.done else None
        self.policy.reset(self.policy.convert(self.raw))
        self.tcp_history=[as_numpy(self.raw['extra']['tcp_pose'][0,:3]).copy()]
        self.initial_state=clone(self.env.get_state_dict())
        return self
    def snapshot(self):
        return {'env':save_env(self.env),'policy':self.policy.snapshot(),'raw':clone(self.raw),
                't':self.t,'done':self.done,'y':self.y,'reason':self.reason,'repeat':self.repeat,
                'intervention':self.intervention,'intervention_t':self.intervention_t,
                'tcp_history':clone(self.tcp_history),'episode':self.episode,'actions':clone(self.actions)}
    def restore(self,s):
        if s['episode']!=self.episode:raise ValueError('Snapshot belongs to a different episode/scene')
        atol=self.c.get('audit',{}).get('state_atol',0.)
        self.replay_error=restore_env(self.env,s['env'],self.episode,s['actions'],atol)
        self.actions=clone(s['actions']);self.policy.restore(s['policy']);self.raw=clone(s['raw'])
        for k in ('t','done','y','reason','repeat','intervention','intervention_t','tcp_history'):
            setattr(self,k,clone(s[k]))
    def advance(self,action):
        if self.done:raise RuntimeError('Attempted action after completion')
        a=np.asarray(action,np.float32)
        if a.shape!=(self.adim,) or not np.isfinite(a).all():raise ValueError('Invalid action')
        if np.max(np.abs(a))>1.00001:raise ValueError('Action outside normalized limits')
        self.policy.last_action=a.copy();self.actions.append(a.copy())
        self.raw,_,term,_,info=self.env.step(torch.as_tensor(a[None],device=self.env.device))
        self.t+=1;self.policy.observe(self.policy.convert(self.raw))
        self.tcp_history.append(as_numpy(self.raw['extra']['tcp_pose'][0,:3]).copy())
        self.y=int(success(info))
        if self.y:self.done=True;self.reason='success'
        elif bool(as_numpy(term).reshape(-1)[0]):self.done=True;self.reason='env_termination_without_success'
        elif self.t>=self.horizon:self.done=True;self.reason='horizon'
        return a
    def step_base(self):return self.advance(self.policy.action(self.episode,self.t,self.repeat))
    def eligible(self):
        p=self.c['protocol'];rem=len(self.policy.queue)
        ok=(not self.done and self.intervention is None and self.t>=p['min_intervention_step']
            and self.t%p['decision_stride']==0 and 0<rem<self.policy.act_horizon)
        return np.array([True,ok,ok and self.horizon-self.t>self.c['retreat']['steps']+p['minimum_post_retreat_steps']])
    def force(self,action):
        if action==0:return
        if action not in (1,2) or self.intervention is not None:raise ValueError('Invalid/repeated intervention')
        if not self.eligible()[action]:raise ValueError('Ineligible intervention (must be inside a chunk and within budget)')
        self.intervention=int(action);self.intervention_t=self.t;self.policy.clear()
        if action==2:self.retreat()
    def retreat(self):
        """Bounded EE movement, not a reset and not assumed safe."""
        r=self.c['retreat'];controller=self.env.agent.controller.controllers['arm'];cfg=controller.config
        if not cfg.frame.startswith('root_translation') or not cfg.normalize_action or cfg.use_target:
            raise ValueError('Unsupported controller frame/normalization/target semantics')
        rootq=as_numpy(self.env.agent.robot.pose.q).reshape(-1)
        if not np.allclose(np.abs(rootq),[1,0,0,0],atol=1e-5):
            raise ValueError('This retreat adapter requires an unrotated Panda base')
        scale=np.broadcast_to(np.asarray(cfg.pos_upper,float),(3,))
        low=np.broadcast_to(np.asarray(cfg.pos_lower,float),(3,))
        if not np.allclose(low,-scale):raise ValueError('Symmetric normalized translation limits required')
        current=self.tcp_history[-1].copy()
        if r['mode']=='up':target=current+np.array([0.,0.,r['distance_m']])
        elif r['mode']=='retrace':
            idx=max(0,len(self.tcp_history)-1-r['lookback_steps']);v=self.tcp_history[idx]-current
            d=np.linalg.norm(v);target=current+v*min(1.,r['distance_m']/max(d,1e-12))
        else:raise ValueError('Unsupported retreat mode')
        # Hold the last commanded gripper position, not an arbitrary open/close operation.
        grip=float(self.policy.last_action[-1]) if self.policy.last_action is not None else 0.
        for _ in range(r['steps']):
            if self.done:break
            delta=target-self.tcp_history[-1]
            norm=np.linalg.norm(delta)
            delta*=min(1.,r['max_translation_per_step_m']/max(norm,1e-12))
            a=np.zeros(self.adim,np.float32);a[:3]=np.clip(delta/scale,-1,1);a[-1]=grip
            self.advance(a)
        self.policy.clear()
    def complete(self,chooser=None):
        while not self.done:
            if chooser is not None and self.intervention is None and self.eligible()[1]:
                action=chooser(self)
                self.force(int(action))
                if self.done:break
            self.step_base()
        return {'success':self.y,'steps':self.t,'reason':self.reason,'intervention':self.intervention,
                'intervention_step':self.intervention_t,'policy_calls':self.policy.inference_calls,
                'policy_inference_seconds':self.policy.inference_seconds}

def trace(runtime,steps):
    out=[]
    for _ in range(steps):
        if runtime.done:break
        a=runtime.step_base()
        out.append({'a':a,'state':clone(runtime.env.get_state_dict()),
                    'obs':clone(runtime.policy.history[-1]),'y':runtime.y})
    return out

def compare_traces(a,b):
    if len(a)!=len(b):return {'pass':False,'length_mismatch':True,'state_max_abs':None,'obs_max_abs':None,'action_max_abs':None}
    return {'length_mismatch':False,
            'state_max_abs':max([maxdiff(x['state'],y['state']) for x,y in zip(a,b)]+[0.]),
            'obs_max_abs':max([maxdiff(x['obs'],y['obs']) for x,y in zip(a,b)]+[0.]),
            'action_max_abs':max([maxdiff(x['a'],y['a']) for x,y in zip(a,b)]+[0.]),
            'outcome_equal':all(x['y']==y['y'] for x,y in zip(a,b))}
