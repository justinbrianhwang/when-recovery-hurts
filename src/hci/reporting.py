from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json,csv
import numpy as np
from .io import read_json,write_json,digest,sha256,new_run,finish_run
from .statistics import paired_metrics,bootstrap_gain,binomial_safe_p,holm_reject,cp_upper

def load_evaluation(folder):
    p=Path(folder);meta=read_json(p/'evaluation.json')
    if read_json(p/'run.json')['status']!='COMPLETE':raise ValueError('Evaluation is incomplete; never drop failed episodes silently')
    rows=[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines() if x.strip()]
    ids=[x['id'] for x in meta['registry']['candidates']];seeds=meta['seeds']
    indexed={}
    for row in rows:
        key=(row['candidate'],row['episode'])
        if key in indexed:raise ValueError('Duplicate candidate/episode pair')
        indexed[key]=row
    expected={(c,s) for c in ids for s in seeds}
    if set(indexed)!=expected:raise ValueError('Missing or unexpected candidate/episode rows')
    for s in seeds:
        vals={indexed[(c,s)]['y0'] for c in ids}
        if len(vals)!=1:raise ValueError('Inconsistent baseline outcome across candidates')
    return meta,{c:[indexed[c,s] for s in seeds] for c in ids}

def summarize(folder,out):
    meta,groups=load_evaluation(folder);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    result=[]
    for cand,rows in groups.items():
        a=[r['y0'] for r in rows];b=[r['yg'] for r in rows];m=paired_metrics(a,b)
        m.update(candidate=cand,family=rows[0]['family'],gain_ci95_pp=bootstrap_gain(a,b),
                 intervention_rate=float(np.mean([r['intervention'] is not None for r in rows])),
                 policy_inference_seconds_mean=float(np.mean([r['policy_inference_seconds'] for r in rows])),
                 decision_seconds_mean=float(np.mean([r['decision_seconds'] for r in rows])))
        # The first-success subset is the denominator, not all triggered episodes.
        pos=[r for r in rows if r['y0']==1]
        tr=[r for r in pos if r['intervention'] is not None]
        m['false_alarm_rate_conditional']=len(tr)/len(pos) if pos else None
        m['harm_given_false_alarm']=sum(r['yg']==0 for r in tr)/len(tr) if tr else None
        result.append(m)
    write_json(out/'metrics.json',{'split':meta['split'],'source_type':meta['source_type'],'metrics':result,
      'caution':'Pointwise test confidence intervals are descriptive; they do not replace the calibration guarantee or multiple-comparison handling.'})
    keys=['candidate','family','n','baseline_success_rate','success_rate','gain_pp','conditional_harm','conditional_rescue','intervention_rate']
    with (out/'metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(result)
    lines=['# Evaluation report',f"Split: {meta['split']} | source: {meta['source_type']}",'',
           '| Candidate | Success | Gain (pp) | Harm conditional | Rescue conditional |','|---|---:|---:|---:|---:|']
    def rate(v):return 'undefined' if v is None else f'{v:.4f}'
    for r in result:lines.append(f"| {r['candidate']} | {r['success_rate']:.4f} | {r['gain_pp']:+.2f} | {rate(r['conditional_harm'])} | {rate(r['conditional_rescue'])} |")
    lines+=['','A harm denominator of zero is undefined, not evidence of safety.','No numerical success improvement is claimed without real held-out evaluation.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result

def certify(devdir,caldir,out,epsilon=.05,delta=.05):
    dev,dg=load_evaluation(devdir);cal,cg=load_evaluation(caldir)
    if dev['split']!='dev' or cal['split']!='calibration':raise ValueError('Require development and calibration, not test')
    if dev['source_type']!='real_sim' or cal['source_type']!='real_sim':raise ValueError('Synthetic rows cannot certify a research controller')
    for key in ('config_sha256','policy_sha256','policy_metadata_sha256','heads_sha256','registry_sha256','split_manifest_sha256'):
        if dev[key]!=cal[key]:raise ValueError(f'Provenance mismatch: {key}')
    if set(dev['seeds'])&set(cal['seeds']):raise ValueError('Dev/calibration seed leakage')
    specs=cal['registry']['candidates'];hyp=[c for c in specs if c['kind']!='never']
    ps=[];stats={}
    for c in specs:
        rows=cg[c['id']];m=paired_metrics([r['y0'] for r in rows],[r['yg'] for r in rows])
        if c['kind']=='never' and m['harm_count']:raise ValueError('Never-intervene has harm; invalid pairing')
        m['p_value']=0. if c['kind']=='never' else binomial_safe_p(m['harm_count'],m['baseline_successes'],epsilon)
        stats[c['id']]=m
        if c['kind']!='never':ps.append(m['p_value'])
    accept=holm_reject(ps,delta);certified={'never'}
    for c,ok in zip(hyp,accept):
        stats[c['id']]['holm_rejected_unsafe_null']=ok
        if ok:certified.add(c['id'])
    devscores={c:np.mean([r['yg'] for r in rows]) for c,rows in dg.items()}
    chosen={}
    for family in sorted({c['family'] for c in specs}):
        allowed=[c['id'] for c in specs if c['family']==family and c['id'] in certified]
        chosen[family]=max(allowed,key=lambda k:(devscores[k],k)) if allowed else 'never'
    p=new_run(out,'certify')
    result={'epsilon':epsilon,'delta':delta,'procedure':'exact binomial p-values + Holm across fixed non-baseline candidates',
      'guarantee_scope':'Same coupled episode distribution; probability over calibration samples. Not a per-state, OOD, hardware, or latency safety guarantee.',
      'registry_sha256':cal['registry_sha256'],'heads_sha256':cal['heads_sha256'],'config_sha256':cal['config_sha256'],
      'policy_sha256':cal['policy_sha256'],'split_manifest_sha256':cal['split_manifest_sha256'],
      'calibration_evaluation':str(Path(caldir).resolve()),'development_evaluation':str(Path(devdir).resolve()),
      'calibration_rows_sha256':sha256(Path(caldir)/'episodes.jsonl'),'development_rows_sha256':sha256(Path(devdir)/'episodes.jsonl'),
      'certified_candidates':sorted(certified),'selected_by_family':chosen,'calibration_statistics':stats,
      'selection_rule':'Maximize development final-success rate ONLY among simultaneously certified candidates; fallback never.'}
    write_json(p/'selection.json',result);finish_run(p);return result
