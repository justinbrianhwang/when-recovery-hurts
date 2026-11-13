from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import yaml

UPSTREAM_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"

def json_default(v):
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    raise TypeError(f"Not JSON serializable: {type(v)}")

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=json_default).encode()).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path, value, *, overwrite=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not overwrite: raise FileExistsError(f"Refusing overwrite: {p}")
    text=json.dumps(value,indent=2,ensure_ascii=False,default=json_default,allow_nan=False)+'\n'
    # Exclusive open prevents accidentally overwriting previous experiment results.
    with p.open('w' if overwrite else 'x',encoding='utf-8') as f: f.write(text)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_config(path):
    c=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if c['env']['sim_backend'] != 'physx_cpu' or c['env']['num_envs'] != 1:
        raise ValueError('v0.3 supports single-environment CPU physics only; GPU rendering/inference are used. Validate parallel physics separately.')
    if c['env']['obs_mode']!='rgb': raise ValueError('Research adapter requires RGB + whitelisted proprioception.')
    if c['protocol']['max_interventions']!=1: raise ValueError('At most one intervention is supported.')
    if c['protocol']['success_definition']!='success_once_absorbing': raise ValueError('Unsupported success definition.')
    if c['policy']['act_horizon']<2: raise ValueError('Refresh needs a remaining action chunk.')
    if c['policy']['obs_horizon']+c['policy']['act_horizon']-1>c['policy']['pred_horizon']:
        raise ValueError('Invalid diffusion horizons.')
    if c['env']['control_mode'] not in ('pd_ee_delta_pos','pd_ee_delta_pose'):
        raise ValueError('Retreat adapter only supports Panda root-frame EE delta controllers.')
    return c

def new_run(path, command, config=None):
    p=Path(path);p.mkdir(parents=True,exist_ok=False)
    write_json(p/'run.json',{'command':command,'created_utc':datetime.now(timezone.utc).isoformat(),
                           'config':config,'config_sha256':digest(config),'status':'STARTED'})
    return p

def finish_run(path, **kwargs):
    p=Path(path)/'run.json';v=read_json(p);v.update(status='COMPLETE',**kwargs);write_json(p,v,overwrite=True)

def command_output(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=30)
        return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except (OSError,subprocess.TimeoutExpired) as e: return {'returncode':-1,'error':str(e)}

def runtime_info():
    import importlib.metadata as md
    pkgs={}
    for n in ['torch','torchvision','mani_skill','sapien','gymnasium','diffusers','numpy','scipy','PyYAML']:
        try: pkgs[n]=md.version(n)
        except md.PackageNotFoundError: pkgs[n]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':pkgs,
            'nvidia_smi':command_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'])}

def seed_for(episode, step, repeat=0, stream='policy'):
    # Counter-keyed seed, independent of how often the PRNG has been called.
    raw=f'{int(episode)}|{int(step)}|{int(repeat)}|{stream}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')%(2**31-1)

def seed_ids(manifest, split, limit=None):
    if split not in manifest['splits']: raise KeyError(split)
    ids=list(manifest['splits'][split])
    if limit is not None:
        if limit<=0: raise ValueError('episodes must be positive')
        ids=ids[:limit]
    if not ids: raise ValueError('Empty split.')
    return ids

def validate_splits(m):
    seen=set()
    for name,ids in m['splits'].items():
        if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate seed in {name}')
        overlap=seen.intersection(ids)
        if overlap: raise ValueError(f'Seed leakage: {overlap}')
        seen.update(ids)
    return True
