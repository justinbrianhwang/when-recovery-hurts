"""Run the upstream RGB Diffusion Policy with an explicit observation whitelist.
Dry-run is default; --execute starts paid GPU work. Upstream source is not edited.
"""
from pathlib import Path
import argparse,os,runpy,sys,re
import h5py
from hci.io import load_config,write_json,UPSTREAM_COMMIT,sha256
from hci.policies import load_upstream

p=argparse.ArgumentParser()
p.add_argument('--config',required=True);p.add_argument('--demo',required=True);p.add_argument('--run',required=True)
p.add_argument('--upstream',default='third_party/ManiSkill');p.add_argument('--num-demos',type=int,default=20)
p.add_argument('--total-iters',type=int,default=500);p.add_argument('--batch-size',type=int,default=64)
p.add_argument('--seed',type=int,default=1);p.add_argument('--eval-freq',type=int,default=500)
p.add_argument('--execute',action='store_true')
a=p.parse_args();c=load_config(a.config)
if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run):p.error('Use an alphanumeric run name without slashes')
if min(a.num_demos,a.total_iters,a.batch_size,a.eval_freq)<=0:p.error('All counts must be positive')
path=Path(a.demo).expanduser().resolve()
if not path.is_file():p.error(f'Dataset not found: {path}')
with h5py.File(path,'r') as f:
    keys=[k for k in f if k.startswith('traj_')]
    if len(keys)<a.num_demos:p.error(f'Only {len(keys)} replayed trajectories are available; requested {a.num_demos}')
run_name='policy/'+a.run
args={**c['policy'],'env_id':c['env']['id'],'control_mode':c['env']['control_mode'],
 'sim_backend':'physx_cpu','obs_mode':'rgb','max_episode_steps':c['env']['horizon'],
 'exp_name':run_name,'seed':a.seed,'demo_path':str(path),'num_demos':a.num_demos,
 'total_iters':a.total_iters,'batch_size':a.batch_size,'eval_freq':a.eval_freq,
 'num_eval_envs':1,'num_eval_episodes':5,'capture_video':False,'track':False,'save_freq':a.eval_freq}
argv=[]
for k,v in args.items():
    key='--'+k.replace('_','-')
    if isinstance(v,bool):argv.append(key if v else '--no-'+k.replace('_','-'))
    elif isinstance(v,list):argv.extend([key,*map(str,v)])
    else:argv.extend([key,str(v)])
script=Path(a.upstream).resolve()/'examples/baselines/diffusion_policy/train_rgbd.py'
print('UPSTREAM:',script);print('ARGS:',argv);print('WHITELIST: RGB + qpos/qvel/TCP pose only')
if not a.execute:print('DRY RUN: add --execute after reviewing resource use.');raise SystemExit(0)
load_upstream(a.upstream)  # verify exact commit and importability
out=Path('runs')/run_name;out.mkdir(parents=True,exist_ok=False)
write_json(out/'policy_meta_training.json',{'args':args,'config':c,'upstream_commit':UPSTREAM_COMMIT,
 'observation_schema':'rgb_qpos_qvel_tcp_v1','demo_sha256':sha256(path),'demo_path':str(path),
 'training_note':'Official algorithm; observation whitelist and ordering patched in memory only. GPU execution not validated by package builder.'})
from hci.observations import install_training_patches
install_training_patches()
os.environ['WANDB_MODE']='disabled'
sys.argv=[str(script),*argv]
runpy.run_path(str(script),run_name='__main__')
