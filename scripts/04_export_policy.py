from pathlib import Path
import argparse,torch
from hci.io import read_json,write_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
a=p.parse_args();run=Path(a.run).resolve();ckpt=Path(a.checkpoint).resolve()
if not ckpt.is_relative_to(run/'checkpoints'):p.error('Select a checkpoint from this training run, using development evaluation only')
v=read_json(run/'policy_meta_training.json');weights=torch.load(ckpt,map_location='cpu',weights_only=True)
if 'ema_agent' not in weights:p.error('Expected the official EMA checkpoint; refusing silent fallback')
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
torch.save(weights['ema_agent'],out/'weights.pt')
m={'args':v['args'],'observation_schema':v['observation_schema'],'upstream_commit':v['upstream_commit'],
 'env_id':v['config']['env']['id'],'control_mode':v['config']['env']['control_mode'],'horizon':v['config']['env']['horizon'],
 'robot_uid':v['config']['env']['robot_uid'],'weights_sha256':sha256(out/'weights.pt'),
 'source_checkpoint':str(ckpt),'source_checkpoint_sha256':sha256(ckpt),'source_training_metadata':v}
write_json(out/'policy_meta.json',m)
print('Exported frozen base policy:',out)
