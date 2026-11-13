from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .io import write_json,sha256,digest

class Head(nn.Module):
    def __init__(self,d,out,width=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,out))
    def forward(self,x):return self.net(x)

class Predictors:
    def __init__(self,path,device='cpu'):
        self.path=Path(path);self.device=torch.device(device)
        v=torch.load(path,map_location='cpu',weights_only=True)
        self.mean=v['mean'].to(self.device);self.std=v['std'].to(self.device);self.dim=len(self.mean)
        self.q=Head(self.dim,3,v['width']).to(self.device);self.joint=Head(self.dim,8,v['width']).to(self.device)
        self.q.load_state_dict(v['q']);self.joint.load_state_dict(v['joint']);self.q.eval();self.joint.eval()
        self.fingerprint=sha256(path)
    @torch.no_grad()
    def predict(self,x):
        z=(torch.as_tensor(x,dtype=torch.float32,device=self.device)-self.mean)/self.std
        q=self.q(z).sigmoid().cpu().numpy()
        j=self.joint(z).reshape(2,4).softmax(-1).cpu().numpy()
        return q,j

def train_heads(branchdir,out,*,steps=2000,batch_size=256,seed=1,width=128):
    p=Path(branchdir)
    import json
    meta=json.loads((p/'dataset.json').read_text())
    if meta['split']!='train':raise ValueError('Only train branches can fit predictors')
    if not meta.get('completed'):raise ValueError('Dataset collection is incomplete; do not silently train on a failed run')
    if steps<1 or batch_size<1:raise ValueError('Training counts must be positive')
    arrays=[]
    for f in sorted((p/'episodes').glob('*.npz')):
        with np.load(f,allow_pickle=False) as z:arrays.append({k:z[k] for k in ('x','action','y0','y')})
    if not arrays:raise ValueError('No completed branch data')
    x=np.concatenate([a['x'] for a in arrays]);act=np.concatenate([a['action'] for a in arrays]).astype(int)
    y0=np.concatenate([a['y0'] for a in arrays]).astype(int);y=np.concatenate([a['y'] for a in arrays]).astype(int)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Invalid features')
    if not set(act)=={0,1,2}:raise ValueError('All three action labels required')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x=torch.as_tensor(x,dtype=torch.float32);mu=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-5);x=(x-mu)/sd
    q=Head(x.shape[1],3,width);joint=Head(x.shape[1],8,width)
    optq=torch.optim.AdamW(q.parameters(),lr=1e-3,weight_decay=1e-4)
    optj=torch.optim.AdamW(joint.parameters(),lr=1e-3,weight_decay=1e-4)
    acts=torch.as_tensor(act);ys=torch.as_tensor(y,dtype=torch.float32);cls=torch.as_tensor(2*y0+y)
    indices_joint=np.flatnonzero(act>0)
    losses=[]
    for step in range(steps):
        iq=rng.integers(0,len(x),batch_size);ij=rng.choice(indices_joint,batch_size,replace=True)
        predq=q(x[iq]).gather(1,acts[iq,None]).squeeze(1)
        lq=nn.functional.binary_cross_entropy_with_logits(predq,ys[iq])
        optq.zero_grad();lq.backward();optq.step()
        predj=joint(x[ij]).reshape(-1,2,4)[torch.arange(batch_size),acts[ij]-1]
        lj=nn.functional.cross_entropy(predj,cls[ij])
        optj.zero_grad();lj.backward();optj.step()
        if step%100==0 or step==steps-1:losses.append({'step':step+1,'q_loss':float(lq.detach()),'joint_loss':float(lj.detach())})
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.save({'q':q.state_dict(),'joint':joint.state_dict(),'mean':mu,'std':sd,'width':width},out/'heads.pt')
    write_json(out/'training.json',{'source_dataset':str(p.resolve()),'dataset_manifest_sha256':sha256(p/'dataset.json'),
      'config_sha256':meta['config_sha256'],'policy_sha256':meta['policy_sha256'],'observation_schema':meta['observation_schema'],
      'feature_description':'Frozen DP encoder + proprio history + last action + normalized time/queue length',
      'steps':steps,'seed':seed,'n_rows':len(x),'feature_dim':x.shape[1],'losses':losses,
      'q_parameters':sum(v.numel() for v in q.parameters()),'joint_parameters':sum(v.numel() for v in joint.parameters()),
      'class_counts':np.bincount(2*y0[act>0]+y[act>0],minlength=4),
      'heads_sha256':sha256(out/'heads.pt'),'interpretation':'pilot estimators; output scores are not calibrated guarantees'})
