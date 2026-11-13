from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
from .io import sha256,write_json,read_json,digest,runtime_info

APPROVAL='RUN_FROZEN_TEST'

def freeze(root,out,config,policy,heads,registry,splits,selection,audit,upstream):
    root=Path(root).resolve();out=Path(out)
    from .io import load_config
    from .experiment import fingerprint,require_audit
    c=load_config(config);sel=read_json(selection);reg=read_json(registry);sp=read_json(splits)
    require_audit(audit,c,policy)
    for k,v in fingerprint(c,policy).items():
        if k in sel and sel[k]!=v:raise ValueError(f'Selection mismatch {k}')
    if sel['registry_sha256']!=digest(reg) or sel['split_manifest_sha256']!=digest(sp) or sel['heads_sha256']!=sha256(heads):
        raise ValueError('Selected controller no longer matches inputs')
    include=[Path(config),Path(heads),Path(heads).parent/'training.json',Path(registry),Path(splits),Path(selection),Path(audit),
             Path(policy)/'weights.pt',Path(policy)/'policy_meta.json']
    include+=list((root/'src').rglob('*.py'))+list((root/'scripts').rglob('*.py'))+list((root/'scripts').rglob('*.sh'))
    include+=[root/'pyproject.toml',root/'requirements/constraints-linux-py311.txt']
    for name in ('calibration_evaluation','development_evaluation'):
        d=Path(sel[name]);include.extend(d/x for x in ('episodes.jsonl','evaluation.json','run.json'))
    files={str(p.resolve()):sha256(p) for p in sorted(set(include))}
    args={'config':str(Path(config).resolve()),'policy':str(Path(policy).resolve()),'heads':str(Path(heads).resolve()),
          'registry':str(Path(registry).resolve()),'splits':str(Path(splits).resolve()),'selection':str(Path(selection).resolve()),
          'audit':str(Path(audit).resolve()),'upstream':str(Path(upstream).resolve())}
    out.mkdir(parents=True,exist_ok=False)
    pip=subprocess.run(['python','-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (out/'pip-freeze.txt').write_text(pip)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files_sha256':files,'args':args,
              'runtime':runtime_info(),'pip_freeze_sha256':sha256(out/'pip-freeze.txt'),
              'upstream_commit':subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
              'upstream_diff_sha256':digest(subprocess.check_output(['git','-C',str(upstream),'diff','HEAD'],text=True)),
              'selected_by_family':sel['selected_by_family'],'note':'Operational lock, not a cryptographic access-control system.'}
    write_json(out/'manifest.json',manifest);return manifest

def verify(folder):
    p=Path(folder);m=read_json(p/'manifest.json')
    changed=[s for s,h in m['files_sha256'].items() if not Path(s).is_file() or sha256(s)!=h]
    if changed:raise ValueError('Frozen inputs changed: '+str(changed))
    if digest(runtime_info())!=digest(m['runtime']):raise ValueError('Runtime changed since freeze')
    if sha256(p/'pip-freeze.txt')!=m['pip_freeze_sha256']:raise ValueError('Saved dependency record changed')
    current=subprocess.check_output(['python','-m','pip','freeze'],text=True)
    saved=(p/'pip-freeze.txt').read_text()
    if sorted(current.splitlines())!=sorted(saved.splitlines()):raise ValueError('Installed dependency environment changed')
    upstream=m['args']['upstream']
    if subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()!=m['upstream_commit']:raise ValueError('Upstream commit changed')
    if digest(subprocess.check_output(['git','-C',upstream,'diff','HEAD'],text=True))!=m['upstream_diff_sha256']:raise ValueError('Upstream source edits changed')
    return m

def run_frozen(folder,out,approval,device='cuda'):
    if approval!=APPROVAL:raise PermissionError('Explicit final Test approval token is required')
    p=Path(folder);m=verify(folder)
    # Exclusive marker intentionally also blocks blind reruns after a crash. Document an amendment first.
    write_json(p/'TEST_STARTED.json',{'out':str(Path(out).resolve()),'started_utc':datetime.now(timezone.utc).isoformat()})
    from .io import load_config
    from .experiment import evaluate
    a=m['args'];reg=read_json(a['registry']);ids=set(m['selected_by_family'].values())|{'never'}
    reg['candidates']=[c for c in reg['candidates'] if c['id'] in ids]
    reg['frozen_selected_by_family']=m['selected_by_family']
    return evaluate(load_config(a['config']),a['policy'],a['upstream'],a['heads'],reg,out,read_json(a['splits']),
                    'test',None,a['audit'],device,allow_test=True)
