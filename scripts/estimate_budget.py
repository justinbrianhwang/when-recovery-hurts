"""User-supplied rate only; no live cloud price is assumed."""
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--seconds-per-rollout',type=float,required=True)
p.add_argument('--trajectories',type=int,default=120);p.add_argument('--checkpoints',type=int,default=4)
p.add_argument('--actions',type=int,default=3);p.add_argument('--repeats',type=int,default=4)
p.add_argument('--hourly-usd',type=float);p.add_argument('--overhead-factor',type=float,default=1.3)
a=p.parse_args()
if min(a.seconds_per_rollout,a.trajectories,a.checkpoints,a.actions,a.repeats,a.overhead_factor)<=0:p.error('All workload arguments must be positive')
n=a.trajectories*a.checkpoints*a.actions*a.repeats;hours=n*a.seconds_per_rollout*a.overhead_factor/3600
print(json.dumps({'branch_rollouts_upper_bound':n,'estimated_gpu_instance_hours':hours,
 'compute_cost_usd':None if a.hourly_usd is None else hours*a.hourly_usd,
 'excludes':'Policy training, calibration candidate sweeps, final Test, storage and network; benchmark on your selected host first.'},indent=2))
