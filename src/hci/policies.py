from __future__ import annotations
from collections import deque
from types import SimpleNamespace
from pathlib import Path
import importlib.util, sys, time
import numpy as np
import torch
from .io import read_json, sha256, seed_for, UPSTREAM_COMMIT
from .trees import clone, as_numpy
from .schema import state_parts, sensor_names

def load_upstream(root):
    root=Path(root).resolve()
    import subprocess
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if commit!=UPSTREAM_COMMIT:raise ValueError(f'Upstream revision mismatch: {commit}')
    path=root/'examples/baselines/diffusion_policy/train_rgbd.py'
    sys.path.insert(0,str(path.parent))
    name='_hci_upstream_rgb_dp'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name]

class QueuePolicy:
    """Owns every mutable execution buffer. Snapshot it together with the simulator."""
    def __init__(self,act_horizon,obs_horizon):
        self.act_horizon=act_horizon;self.obs_horizon=obs_horizon
        self.history=deque(maxlen=obs_horizon);self.queue=[];self.last_action=None
        self.inference_seconds=0.;self.inference_calls=0
    def observe(self,obs):
        self.history.append(clone(obs))
        while len(self.history)<self.obs_horizon:self.history.appendleft(clone(obs))
    def clear(self):self.queue=[]
    def reset(self,obs):
        self.history.clear();self.queue=[];self.last_action=None
        self.inference_seconds=0.;self.inference_calls=0;self.observe(obs)
    def snapshot(self):
        return clone({'history':list(self.history),'queue':self.queue,'last_action':self.last_action,
                      'inference_seconds':self.inference_seconds,'inference_calls':self.inference_calls})
    def restore(self,s):
        self.history=deque(clone(s['history']),maxlen=self.obs_horizon)
        self.queue=clone(s['queue']);self.last_action=clone(s['last_action'])
        self.inference_seconds=s['inference_seconds'];self.inference_calls=s['inference_calls']
    def action(self,episode,t,repeat=0):
        if not self.queue:
            started=time.perf_counter();a=self.generate(episode,t,repeat)
            self.inference_seconds+=time.perf_counter()-started;self.inference_calls+=1
            if a.shape[0]!=self.act_horizon or not np.isfinite(a).all():raise ValueError('Invalid policy action chunk')
            self.queue=[x.copy() for x in a]
        self.last_action=self.queue.pop(0);return self.last_action.copy()

class DiffusionPolicy(QueuePolicy):
    def __init__(self,env,artifact,upstream,device='cuda'):
        self.artifact=Path(artifact);self.meta=read_json(self.artifact/'policy_meta.json')
        if self.meta['observation_schema']!='rgb_qpos_qvel_tcp_v1':raise ValueError('Policy observation whitelist mismatch')
        if sha256(self.artifact/'weights.pt')!=self.meta['weights_sha256']:raise ValueError('Checkpoint hash mismatch')
        a=self.meta['args'];super().__init__(a['act_horizon'],a['obs_horizon'])
        self.device=torch.device(device)
        self.cameras=sensor_names(env._init_raw_obs)
        raw=self.convert(env._init_raw_obs)
        import gymnasium as gym
        st=raw['state'].shape[-1];shape=tuple(raw['rgb'].shape[1:])
        spaces=SimpleNamespace(single_action_space=env.single_action_space,
            single_observation_space=gym.spaces.Dict({
                'state':gym.spaces.Box(-np.inf,np.inf,(self.obs_horizon,st),np.float32),
                'rgb':gym.spaces.Box(0,255,(self.obs_horizon,*shape),np.uint8)}))
        module=load_upstream(upstream)
        self.agent=module.Agent(spaces,SimpleNamespace(**a)).to(self.device).eval()
        weights=torch.load(self.artifact/'weights.pt',map_location=self.device,weights_only=True)
        self.agent.load_state_dict(weights,strict=True)
        self.adim=env.single_action_space.shape[0]
        self.reset(raw)
    def convert(self,raw):
        return {'state':torch.cat(state_parts(raw),-1).to(self.device).float(),
                'rgb':torch.cat([raw['sensor_data'][n]['rgb'] for n in self.cameras],-1).to(self.device)}
    def stacked(self):return {k:torch.stack([x[k] for x in self.history],1) for k in ('state','rgb')}
    @torch.no_grad()
    def generate(self,episode,t,repeat=0):
        devices=list(range(torch.cuda.device_count())) if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.last_seed=seed_for(episode,t,repeat,'dp')   # logging only (Stage 2B inference trace); the seed rule is unchanged
            torch.manual_seed(self.last_seed)
            a=self.agent.get_action(self.stacked())
            if self.device.type=='cuda':torch.cuda.synchronize()
        return np.clip(as_numpy(a[0]),-1,1).astype(np.float32)
    @torch.no_grad()
    def features(self,t,horizon):
        obs=self.stacked();obs['rgb']=obs['rgb'].permute(0,1,4,2,3)
        z=as_numpy(self.agent.encode_obs(obs,eval_mode=True)[0]).astype(np.float32)
        last=np.zeros(self.adim,dtype=np.float32) if self.last_action is None else self.last_action
        return np.concatenate([z,last,np.array([t/horizon,len(self.queue)/self.act_horizon],np.float32)])

class RandomChunkPolicy(QueuePolicy):
    """Plumbing tests ONLY. Outputs cannot be used as research policy results."""
    def __init__(self,adim,act_horizon=8,obs_horizon=2):
        super().__init__(act_horizon,obs_horizon);self.adim=adim
    def generate(self,episode,t,repeat=0):
        return np.random.default_rng(seed_for(episode,t,repeat)).uniform(-.05,.05,(self.act_horizon,self.adim)).astype(np.float32)
