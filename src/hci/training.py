"""Training-side additions that do not edit the pinned upstream trainer.

The upstream `save_ckpt` stores only `agent` and `ema_agent`. A full checkpoint additionally needs the optimizer,
LR scheduler, the complete EMA state and the RNG streams. The upstream script calls
`diffusion_policy.evaluate.evaluate(...)` from `evaluate_and_save_best(iteration)` at iteration 0, every
`eval_freq` iterations and once more after the loop. We wrap that single function: it reads the trainer's own
objects from the caller's module globals, writes the checkpoint, then runs the unmodified upstream evaluation.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib, random, sys
import numpy as np
import torch

REQUIRED=('agent','ema_agent','ema','optimizer','lr_scheduler')
RESUME_SCOPE=('training-state resume, sample order non-identical: model, EMA, optimizer, LR scheduler and global RNG '
              'streams are stored; the position inside the current RandomSampler epoch is not. No resume loader has '
              'been validated; a run restarted from this file must be reported as a separate warm-start experiment.')

def source_fingerprint(root='.'):
    """Hash of the project code that defines training/evaluation behaviour (the server copy has no .git)."""
    root=Path(root);h=hashlib.sha256();files=sorted([*root.glob('src/hci/*.py'),*root.glob('scripts/*.py'),*root.glob('configs/*.yaml')])
    if not files:raise FileNotFoundError(f'No project sources under {root.resolve()}')
    for f in files:
        h.update(f.relative_to(root).as_posix().encode());h.update(b'\0')
        h.update(f.read_bytes().replace(b'\r\n',b'\n'));h.update(b'\0')
    return h.hexdigest()

def pack_rng():
    kind,keys,pos,has_gauss,cached=np.random.get_state()
    return {'python':random.getstate(),'numpy':(kind,torch.from_numpy(keys.astype(np.int64)),int(pos),int(has_gauss),float(cached)),
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def unpack_rng(s):
    random.setstate(s['python']);kind,keys,pos,has_gauss,cached=s['numpy']
    np.random.set_state((kind,keys.numpy().astype(np.uint32),pos,has_gauss,cached));torch.set_rng_state(s['torch_cpu'])
    if s['torch_cuda'] is not None:torch.cuda.set_rng_state_all(s['torch_cuda'])

def save_full_checkpoint(objects,iteration,folder,meta):
    """Write `full_<iteration>.pt`; never overwrites. `iteration` is the upstream tag (see `updates_completed`)."""
    missing=[k for k in REQUIRED if k not in objects]
    if missing:raise KeyError(f'Trainer objects not found for a full checkpoint: {missing}')
    path=Path(folder)/f'full_{int(iteration):06d}.pt'
    if path.exists():raise FileExistsError(f'Refusing overwrite: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    scheduler=objects['lr_scheduler']
    state={'agent':objects['agent'].state_dict(),'ema_agent':objects['ema_agent'].state_dict(),
           'ema':objects['ema'].state_dict(),'optimizer':objects['optimizer'].state_dict(),
           'lr_scheduler':scheduler.state_dict(),'iteration_tag':int(iteration),
           # The upstream loop is 0-based and evaluates after the update, so tag i (< total) means i+1 optimizer updates.
           'updates_completed':int(scheduler.last_epoch),'rng':pack_rng(),'meta':meta,'resume_scope':RESUME_SCOPE}
    tmp=path.with_suffix('.pt.tmp');torch.save(state,tmp);tmp.rename(path)
    return path

def install_checkpoint_hook(folder,meta,every):
    """Save a full checkpoint whenever the upstream trainer evaluates at a multiple of `every` (and at the end)."""
    ev=importlib.import_module('diffusion_policy.evaluate');original=ev.evaluate
    def evaluate(n,agent,eval_envs,device,sim_backend,*args,**kwargs):
        caller=sys._getframe(1);iteration=caller.f_locals.get('iteration')
        if iteration is None:raise RuntimeError('Checkpoint hook: caller has no `iteration` (upstream trainer changed?)')
        g=caller.f_globals
        if g.get('ema_agent') is not agent:raise RuntimeError('Checkpoint hook: evaluated agent is not the trainer EMA agent')
        if iteration>0 and iteration%every==0:
            path=save_full_checkpoint(g,iteration,folder,meta);print(f'FULL CHECKPOINT: {path}',flush=True)
        return original(n,agent,eval_envs,device,sim_backend,*args,**kwargs)
    ev.evaluate=evaluate
    return original
