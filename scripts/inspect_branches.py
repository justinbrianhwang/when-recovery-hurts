import argparse,json
from pathlib import Path
import numpy as np
from hci.io import read_json
p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();root=Path(a.data)
meta=read_json(root/'dataset.json');counts={k:np.zeros(4,dtype=int) for k in range(3)};nfiles=0
for f in sorted((root/'episodes').glob('*.npz')):
    with np.load(f,allow_pickle=False) as z:
        cls=2*z['y0']+z['y'];acts=z['action']
        for u in range(3):counts[u]+=np.bincount(cls[acts==u].astype(int),minlength=4)
    nfiles+=1
print(json.dumps({'completed':meta.get('completed'),'episode_files':nfiles,'class_order':['both_fail','rescue','harm','both_success'],
 'counts':{str(k):v.tolist() for k,v in counts.items()},'branch_rollouts':meta.get('branch_rollouts'),
 'wall_seconds':meta.get('wall_seconds'),
 'warning':'Rows are correlated checkpoints/repeats. These counts are diagnostic and not independent-sample risk certification.'},indent=2))
