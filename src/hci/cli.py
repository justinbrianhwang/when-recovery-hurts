from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .io import load_config,read_json,write_json,new_run,finish_run,runtime_info,command_output,validate_splits,seed_ids

def main():
    p=argparse.ArgumentParser(prog='hci',description='Harm-constrained robotic intervention research starter')
    s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('doctor');q.add_argument('--out',required=True);q.add_argument('--cpu-only',action='store_true')
    q=s.add_parser('init-splits');q.add_argument('--out',required=True)
    q=s.add_parser('init-candidates');q.add_argument('--out',required=True)
    q=s.add_parser('smoke');q.add_argument('--config',required=True);q.add_argument('--out',required=True)
    for name in ('audit','collect','evaluate'):
        q=s.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--policy',required=True)
        q.add_argument('--upstream',default='third_party/ManiSkill');q.add_argument('--out',required=True)
        q.add_argument('--splits',default='configs/splits.json');q.add_argument('--episodes',type=int)
        q.add_argument('--device',default='cuda',choices=['cuda','cpu'])
        if name!='audit':q.add_argument('--audit',required=True)
        if name=='evaluate':
            q.add_argument('--heads',required=True);q.add_argument('--candidates',default='configs/candidates.json')
            q.add_argument('--split',required=True,choices=['dev','calibration'])
    q=s.add_parser('fit-heads');q.add_argument('--data',required=True);q.add_argument('--out',required=True)
    q.add_argument('--steps',type=int,default=2000);q.add_argument('--seed',type=int,default=1);q.add_argument('--threads',type=int,default=4)
    q=s.add_parser('summarize');q.add_argument('--run',required=True);q.add_argument('--out',required=True)
    q=s.add_parser('certify');q.add_argument('--dev',required=True);q.add_argument('--calibration',required=True)
    q.add_argument('--out',required=True);q.add_argument('--epsilon',type=float,default=.05);q.add_argument('--delta',type=float,default=.05)
    q=s.add_parser('sample-size');q.add_argument('--epsilon',type=float,default=.05);q.add_argument('--delta',type=float,default=.05);q.add_argument('--candidates',type=int,default=19)
    q=s.add_parser('freeze');q.add_argument('--root',default='.');q.add_argument('--out',required=True)
    for key in ('config','policy','heads','registry','splits','selection','audit'):q.add_argument('--'+key,required=True)
    q.add_argument('--upstream',default='third_party/ManiSkill')
    q=s.add_parser('verify-freeze');q.add_argument('--frozen',required=True)
    q=s.add_parser('frozen-test');q.add_argument('--frozen',required=True);q.add_argument('--out',required=True)
    q.add_argument('--approval',required=True);q.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.cmd=='doctor':
        out=new_run(a.out,'doctor');v=runtime_info();v['vulkan']=command_output(['vulkaninfo','--summary'])
        try:
            import torch
            v['cuda_available']=torch.cuda.is_available()
        except ImportError:v['cuda_available']=False
        v['simulation_import']=command_output([sys.executable,'-c','import mani_skill, sapien, gymnasium; print(mani_skill.__version__)'])
        v['pass']=bool(a.cpu_only or (v['cuda_available'] and v['vulkan']['returncode']==0 and v['simulation_import']['returncode']==0))
        v['scope']='CPU-only flag skips GPU validation; successful GPU info does not replace RGB smoke test.'
        write_json(out/'doctor.json',v);finish_run(out);print(json.dumps(v,indent=2))
        if not v['pass']:raise SystemExit(2)
    elif a.cmd=='init-splits':
        v={'version':1,'grouping':'Entire initial-scene seed, all checkpoints, branches and repeats stay together.',
           'splits':{'audit':list(range(100,105)),'train':list(range(10000,10120)),
                     'dev':list(range(20000,20100)),'calibration':list(range(30000,30400)),
                     'test':list(range(90000,90400))}}
        validate_splits(v);write_json(a.out,v)
    elif a.cmd=='init-candidates':
        from .decisions import default_registry
        write_json(a.out,default_registry())
    elif a.cmd=='smoke':
        from .smoke import smoke
        report=smoke(load_config(a.config),a.out)
        print(json.dumps(report,indent=2))
        if not report['pass']:raise SystemExit(2)
    elif a.cmd in ('audit','collect','evaluate'):
        from . import experiment as e
        c=load_config(a.config);splits=read_json(a.splits);validate_splits(splits)
        if a.cmd=='audit':
            v=e.audit(c,a.policy,a.upstream,a.out,seed_ids(splits,'audit',a.episodes),a.device)
            print(json.dumps(v,indent=2))
            if not v['pass']:raise SystemExit(2)
        elif a.cmd=='collect':e.collect(c,a.policy,a.upstream,a.out,splits,a.episodes,a.audit,a.device)
        else:e.evaluate(c,a.policy,a.upstream,a.heads,read_json(a.candidates),a.out,splits,a.split,a.episodes,a.audit,a.device)
    elif a.cmd=='fit-heads':
        import torch
        torch.set_num_threads(a.threads)
        from .models import train_heads
        train_heads(a.data,a.out,steps=a.steps,seed=a.seed)
    elif a.cmd=='summarize':
        from .reporting import summarize
        summarize(a.run,a.out)
    elif a.cmd=='certify':
        from .reporting import certify
        v=certify(a.dev,a.calibration,a.out,a.epsilon,a.delta)
        print(json.dumps(v['selected_by_family'],indent=2))
    elif a.cmd=='sample-size':
        from .statistics import zero_event_requirement
        n=zero_event_requirement(a.epsilon,a.delta,a.candidates)
        print(json.dumps({'zero_observed_harm_baseline_successes_required_bonferroni':n,
                          'warning':'Necessary zero-event design example, NOT a power analysis or total episode count.'},indent=2))
    elif a.cmd=='freeze':
        from .freeze import freeze
        d=vars(a);d.pop('cmd');freeze(**d)
    elif a.cmd=='verify-freeze':
        from .freeze import verify
        verify(a.frozen);print('Frozen inputs verified.')
    elif a.cmd=='frozen-test':
        from .freeze import run_frozen
        run_frozen(a.frozen,a.out,a.approval,a.device)
