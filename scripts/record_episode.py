"""Record a development episode, including intermediate retreat steps. Never uses test seeds."""
from pathlib import Path
import argparse,time
import imageio.v2 as imageio
import numpy as np
from hci.io import load_config,read_json,write_json,new_run,finish_run
from hci.experiment import runtime
from hci.trees import as_numpy
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--policy',required=True)
p.add_argument('--upstream',default='third_party/ManiSkill');p.add_argument('--seed',type=int,default=20000)
p.add_argument('--action',choices=['continue','refresh','retreat_retry'],default='continue')
p.add_argument('--at',type=int,default=12);p.add_argument('--out',required=True);p.add_argument('--splits',default='configs/splits.json')
a=p.parse_args();splits=read_json(a.splits)
if a.seed in splits['splits']['test'] or a.seed in splits['splits']['calibration']:
    p.error('Video exploration is restricted to audit/train/dev seeds')
if a.seed not in set(splits['splits']['audit']+splits['splits']['train']+splits['splits']['dev']):
    p.error('Declare the seed in audit/train/dev first')
c=load_config(a.config);out=new_run(a.out,'record-development',c);r=runtime(c,a.policy,a.upstream);r.reset(a.seed)
act=['continue','refresh','retreat_retry'].index(a.action)
def frame():
    images=[as_numpy(r.raw['sensor_data'][n]['rgb'])[0] for n in sorted(r.raw['sensor_data'])]
    return np.concatenate(images,axis=1).astype(np.uint8)
try:
    with imageio.get_writer(str(out/'episode.mp4'),fps=20) as writer:
        writer.append_data(frame());old=r.advance
        def advance(action):
            z=old(action);writer.append_data(frame());return z
        r.advance=advance
        result=r.complete(lambda rt:act if rt.t>=a.at and rt.eligible()[act] else 0)
    write_json(out/'episode.json',{'seed':a.seed,'requested_action':a.action,'requested_step':a.at,
      'result':result,'note':'Video fps is presentation-only; control step budget defines the experiment.'})
    finish_run(out)
finally:r.env.close()
