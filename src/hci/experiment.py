from __future__ import annotations
from pathlib import Path
import json,time
import numpy as np
from .io import new_run,write_json,read_json,finish_run,digest,sha256,runtime_info,seed_ids,validate_splits
from .trees import clone,maxdiff
from .simulation import make_env,Runtime,trace,compare_traces
from .policies import DiffusionPolicy
from .models import Predictors
from .decisions import choose_action

SCHEMA='rgb_qpos_qvel_tcp_v1'

def runtime(c,policy,upstream,device='cuda'):
    env=make_env(c)
    try:
        p=DiffusionPolicy(env,policy,upstream,device=device)
        m=p.meta
        for key,val in [('env_id',c['env']['id']),('control_mode',c['env']['control_mode']),('horizon',c['env']['horizon']),('robot_uid',c['env']['robot_uid'])]:
            if m[key]!=val:raise ValueError(f'Policy metadata mismatch: {key} {m[key]} != {val}')
        for key in ('obs_horizon','act_horizon','pred_horizon'):
            if m['args'][key]!=c['policy'][key]:raise ValueError(f'Policy horizon mismatch: {key}')
        return Runtime(env,p,c)
    except Exception:env.close();raise

def fingerprint(c,policy):
    return {'config_sha256':digest(c),'policy_sha256':sha256(Path(policy)/'weights.pt'),
            'policy_metadata_sha256':sha256(Path(policy)/'policy_meta.json')}

def require_audit(path,c,policy):
    a=read_json(path)
    if not a.get('pass'):raise ValueError('Restore/policy replay audit did not pass')
    for k,v in fingerprint(c,policy).items():
        if a[k]!=v:raise ValueError(f'Audit no longer matches {k}')
    now=runtime_info()
    if digest(a['runtime'])!=digest(now):raise ValueError('Runtime/host changed since audit; rerun audit on this host')

def audit(c,policy,upstream,out,seeds,device='cuda'):
    p=new_run(out,'audit',c);r=runtime(c,policy,upstream,device);results=[]
    try:
        for seed in seeds:
            r.reset(seed)
            while not r.done and r.t<c['protocol']['min_intervention_step']:r.step_base()
            if r.done:results.append({'seed':seed,'pass':False,'reason':'Episode completed before audit snapshot'});continue
            s=r.snapshot();first=trace(r,c['audit']['replay_steps']);r.restore(s)
            second=trace(r,c['audit']['replay_steps']);cmp=compare_traces(first,second)
            r.restore(s)
            if r.eligible()[2]:r.force(2);r.complete()
            else:r.complete()
            r.restore(s);third=trace(r,c['audit']['replay_steps']);after=compare_traces(first,third)
            def passes(x):
                return (not x.get('length_mismatch',True) and x.get('outcome_equal',False)
                    and x['state_max_abs']<=c['audit']['state_atol'] and x['action_max_abs']<=c['audit']['action_atol']
                    and x['obs_max_abs']<=c['audit']['observation_atol'])
            results.append({'seed':seed,'repeat':cmp,'after_intervention':after,'pass':passes(cmp) and passes(after)})
        report={'pass':bool(results) and all(x['pass'] for x in results),'results':results,
                **fingerprint(c,policy),'runtime':runtime_info(),
                'scope':'Same-scene restore + policy replay at tested states; not proof of global simulator determinism.'}
        write_json(p/'audit.json',report);finish_run(p,audit_pass=report['pass'])
        return report
    finally:r.env.close()

def collect(c,policy,upstream,out,splits,limit,audit_path,device='cuda'):
    require_audit(audit_path,c,policy);validate_splits(splits)
    ids=seed_ids(splits,'train',limit);p=new_run(out,'collect',c);(p/'episodes').mkdir()
    meta={**fingerprint(c,policy),'split':'train','seeds':ids,'observation_schema':SCHEMA,
          'source_type':'real_sim','split_manifest_sha256':digest(splits),'repeat_coupling':'same frozen past; future policy seed keyed by episode, physical step, repeat',
          'success_definition':c['protocol']['success_definition'],'completed':False}
    write_json(p/'dataset.json',meta);r=runtime(c,policy,upstream,device)
    started=time.perf_counter();count=0;completed=[]
    try:
        for seed in ids:
            r.reset(seed);rows=[];ncheck=0
            while not r.done:
                if r.t in c['collection']['checkpoint_steps'] and r.eligible()[2]:
                    s=r.snapshot();x=r.policy.features(r.t,r.horizon);t=r.t
                    for repeat in range(c['collection']['repeats']):
                        outcomes=[]
                        for a in range(3):
                            r.restore(s);r.repeat=repeat;r.force(a);outcomes.append(r.complete());count+=1
                        for a in range(3):rows.append((x,a,outcomes[0]['success'],outcomes[a]['success'],t,repeat))
                    r.restore(s);ncheck+=1
                r.step_base()
            if rows:
                np.savez_compressed(p/'episodes'/f'{seed}.npz',x=np.stack([z[0] for z in rows]),
                  action=np.array([z[1] for z in rows]),y0=np.array([z[2] for z in rows]),y=np.array([z[3] for z in rows]),
                  step=np.array([z[4] for z in rows]),repeat=np.array([z[5] for z in rows]),episode=np.full(len(rows),seed))
            completed.append({'seed':seed,'checkpoints':ncheck,'rows':len(rows),'trunk_success':r.y})
            write_json(p/'episodes'/f'{seed}.json',completed[-1])
            print(f'collected seed={seed} checkpoints={ncheck} branch_rollouts_total={count}',flush=True)
        meta.update(completed=True,episode_summaries=completed,branch_rollouts=count,wall_seconds=time.perf_counter()-started)
        write_json(p/'dataset.json',meta,overwrite=True);finish_run(p,branch_rollouts=count)
        return meta
    finally:r.env.close()

def evaluate(c,policy,upstream,heads,registry,out,splits,split,limit,audit_path,device='cuda',allow_test=False):
    if split=='test' and not allow_test:raise ValueError('Final Test is locked. Use frozen-test with explicit approval.')
    if split not in ('dev','calibration','test'):raise ValueError('Use dev, calibration, or locked test')
    require_audit(audit_path,c,policy);validate_splits(splits);ids=seed_ids(splits,split,limit)
    p=new_run(out,'evaluate',c);candidates=registry['candidates']
    if len({x['id'] for x in candidates})!=len(candidates):raise ValueError('Duplicate candidate IDs')
    hp=Path(heads);hm=read_json(hp.parent/'training.json')
    for key,value in fingerprint(c,policy).items():
        if key in hm and hm[key]!=value:raise ValueError(f'Predictor training provenance mismatch: {key}')
    predictor=Predictors(heads);r=runtime(c,policy,upstream,device)
    write_json(p/'evaluation.json',{**fingerprint(c,policy),'heads_sha256':sha256(heads),'registry_sha256':digest(registry),
      'registry':registry,'split':split,'seeds':ids,'split_manifest_sha256':digest(splits),'source_type':'real_sim',
      'success_definition':c['protocol']['success_definition'],'runtime':runtime_info()})
    started=time.perf_counter();rows=[]
    try:
        with (p/'episodes.jsonl').open('x',encoding='utf-8') as f:
            for seed in ids:
                r.reset(seed);init=clone(r.initial_state);base=r.complete();r.reset(seed)
                mismatch=maxdiff(init,r.initial_state);noop=r.complete()
                if mismatch>c['audit']['state_atol']:raise RuntimeError('Reset seed did not reproduce initial state')
                if base['success']!=noop['success']:raise RuntimeError('No-op paired outcome mismatch; stop and audit coupling')
                for cand in candidates:
                    if cand['kind']=='never':res=dict(base);decision_sec=0.
                    else:
                        r.reset(seed)
                        if maxdiff(init,r.initial_state)>c['audit']['state_atol']:raise RuntimeError('Candidate starts from different initial state')
                        decision_time=[0.]
                        def chooser(rt):
                            tick=time.perf_counter();q=j=None
                            if cand['kind'] in ('q','joint','risk_q'):q,j=predictor.predict(rt.policy.features(rt.t,rt.horizon))
                            a=choose_action(cand,eligible=rt.eligible(),episode=seed,t=rt.t,q=q,joint=j)
                            decision_time[0]+=time.perf_counter()-tick;return a
                        res=r.complete(chooser);decision_sec=decision_time[0]
                    row={'episode':seed,'candidate':cand['id'],'family':cand['family'],'y0':base['success'],
                         'yg':res['success'],'baseline_steps':base['steps'],'noop_y':noop['success'],
                         **{k:v for k,v in res.items() if k!='success'},'decision_seconds':decision_sec}
                    rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                print(f'evaluated {split} seed={seed} candidates={len(candidates)}',flush=True)
        finish_run(p,wall_seconds=time.perf_counter()-started,n_rows=len(rows))
        return rows
    finally:r.env.close()
