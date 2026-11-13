from .io import new_run,write_json,finish_run,runtime_info
from .simulation import make_env,save_env,restore_env
from .trees import clone,maxdiff
import numpy as np
import torch

def smoke(c,out):
    p=new_run(out,'smoke',c);env=make_env(c)
    try:
        raw,info=env.reset(seed=7)
        if not raw.get('sensor_data'):raise ValueError('No actual camera observations')
        dims={k:list(v['rgb'].shape) for k,v in raw['sensor_data'].items()}
        actions=np.random.default_rng(41).uniform(-.03,.03,(12,env.single_action_space.shape[0])).astype(np.float32)
        for a in actions[:4]:env.step(torch.tensor(a[None],device=env.device))
        s=save_env(env)
        def rollout():
            out=[]
            for a in actions[4:]:
                obs,_,_,_,info=env.step(torch.tensor(a[None],device=env.device))
                out.append((clone(env.get_state_dict()),clone(obs),clone(info['success'])))
            return out
        first=rollout();replay_err=restore_env(env,s,7,actions[:4],c['audit']['state_atol']);second=rollout()
        diffs={'state_max_abs':max(maxdiff(x[0],y[0]) for x,y in zip(first,second)),
               'obs_max_abs':max(maxdiff(x[1],y[1]) for x,y in zip(first,second)),
               'success_max_abs':max(maxdiff(x[2],y[2]) for x,y in zip(first,second))}
        report={'camera_shapes':dims,**diffs,'restore_method':'prefix_action_replay','replay_state_max_abs':replay_err,'pass':diffs['state_max_abs']<=c['audit']['state_atol'] and
            diffs['obs_max_abs']<=c['audit']['observation_atol'] and diffs['success_max_abs']==0,
            'runtime':runtime_info(),'scope':'Random-action installation smoke only. Not policy/research validation.'}
        write_json(p/'smoke.json',report);finish_run(p);return report
    finally:env.close()
