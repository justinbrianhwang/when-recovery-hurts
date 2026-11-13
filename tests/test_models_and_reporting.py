import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from hci.io import write_json,read_json,digest
from hci.models import train_heads,Predictors
from hci.reporting import summarize,certify,load_evaluation

def test_small_heads_train_and_reload(tmp_path):
    torch.set_num_threads(1)
    d=tmp_path/'branches';(d/'episodes').mkdir(parents=True)
    write_json(d/'dataset.json',{'split':'train','completed':True,'config_sha256':'a','policy_sha256':'b','observation_schema':'x'})
    rng=np.random.default_rng(42);n=60;x=rng.normal(size=(n,12)).astype(np.float32)
    a=np.arange(n)%3;y0=(x[:,0]>0).astype(int);y=((x[:,0]+a*.1)>0).astype(int)
    np.savez_compressed(d/'episodes/1.npz',x=x,action=a,y0=y0,y=y)
    out=tmp_path/'heads';train_heads(d,out,steps=8,batch_size=16)
    model=Predictors(out/'heads.pt');q,j=model.predict(x[0])
    assert q.shape==(3,) and j.shape==(2,4) and np.allclose(j.sum(1),1,atol=1e-6)

def make_eval(p,split,seed_offset=0,source='unit_test_fixture',bad=False):
    p.mkdir()
    specs=[{'id':'never','family':'never','kind':'never'},
           {'id':'good','family':'joint','kind':'joint'}, {'id':'bad','family':'q','kind':'q'}]
    m={'registry':{'candidates':specs},'seeds':list(range(seed_offset,seed_offset+100)),
       'split':split,'source_type':source,**{k:'x' for k in ['config_sha256','policy_sha256','policy_metadata_sha256','heads_sha256','registry_sha256','split_manifest_sha256']}}
    write_json(p/'evaluation.json',m);write_json(p/'run.json',{'status':'COMPLETE'})
    rows=[]
    for s in m['seeds']:
        i=s-seed_offset;a=int(i<80)
        for c in specs:
            b=a if c['id']=='never' else (int(i>=25) if c['id']=='bad' else int(i<95))
            rows.append({'episode':s,'candidate':c['id'],'family':c['family'],'y0':a,'yg':b,'intervention':None if c['id']=='never' else 1,
             'policy_inference_seconds':0.,'decision_seconds':0.})
    (p/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    return p

def test_reporting_fixture_is_labelled_not_robot_evidence(tmp_path):
    p=make_eval(tmp_path/'run','dev');v=summarize(p,tmp_path/'summary')
    assert len(v)==3
    assert read_json(tmp_path/'summary/metrics.json')['source_type']=='unit_test_fixture'

def test_synthetic_cannot_certify(tmp_path):
    a=make_eval(tmp_path/'d','dev');b=make_eval(tmp_path/'c','calibration',1000)
    with pytest.raises(ValueError,match='Synthetic'):certify(a,b,tmp_path/'s')

def test_missing_episode_is_rejected(tmp_path):
    p=make_eval(tmp_path/'d','dev');f=p/'episodes.jsonl';f.write_text('\n'.join(f.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError,match='Missing'):load_evaluation(p)

def test_selection_uses_dev_success_and_holm(tmp_path):
    # Temporary unit-test fixtures only; no fixture is shipped as experimental evidence.
    a=make_eval(tmp_path/'d','dev',source='real_sim');b=make_eval(tmp_path/'c','calibration',1000,source='real_sim')
    s=certify(a,b,tmp_path/'s')
    assert 'good' in s['certified_candidates'] and 'bad' not in s['certified_candidates']
    assert s['selected_by_family']['joint']=='good' and s['selected_by_family']['q']=='never'

def test_dev_cal_leakage_is_rejected(tmp_path):
    a=make_eval(tmp_path/'d','dev',source='real_sim');b=make_eval(tmp_path/'c','calibration',source='real_sim')
    with pytest.raises(ValueError,match='leakage'):certify(a,b,tmp_path/'s')
