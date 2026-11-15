"""Vision-V1 training launcher: the pinned upstream RGB Diffusion Policy trainer with the visual encoder replaced in
memory by the ImageNet-pretrained ResNet-18 encoder of hci.vision_v1. Upstream source is not edited.

Everything else is what scripts/03_train_dp.py does for the legacy policy: same observation whitelist patches, same
arguments from the config, same full-checkpoint hook. Dry-run is the default; --execute starts GPU work and is only
called by scripts/vision_v1/run.sh after the authorisation check.
"""
from pathlib import Path
import argparse, importlib, os, re, runpy, sys
import h5py
from hci.io import load_config, write_json, UPSTREAM_COMMIT, sha256
from hci.policies import load_upstream
from hci import vision_v1 as V

p = argparse.ArgumentParser()
p.add_argument('--config', required=True); p.add_argument('--demo', required=True); p.add_argument('--run', required=True)
p.add_argument('--pretrained', required=True, help='the verified torchvision ResNet18 IMAGENET1K_V1 file')
p.add_argument('--upstream', default='third_party/ManiSkill'); p.add_argument('--num-demos', type=int, required=True)
p.add_argument('--total-iters', type=int, required=True); p.add_argument('--batch-size', type=int, required=True)
p.add_argument('--seed', type=int, required=True); p.add_argument('--eval-freq', type=int, required=True)
p.add_argument('--num-eval-episodes', type=int, default=5, help='episodes of the upstream in-training evaluator (unseeded; descriptive log, never a selection criterion)')
p.add_argument('--data-manifest', required=True); p.add_argument('--probe', action='store_true', help='non-candidate throughput probe; its weights are never exported')
p.add_argument('--execute', action='store_true'); p.add_argument('--plan', help='required with --execute'); p.add_argument('--auth-run', help='run folder holding authorization.json; required with --execute')
a = p.parse_args(); c = load_config(a.config)
if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run): p.error('Use an alphanumeric run name without slashes')
if a.probe != a.run.startswith('vision_v1_probe'): p.error('a probe run is named vision_v1_probe*, and only a probe is')
if min(a.num_demos, a.total_iters, a.batch_size, a.eval_freq, a.num_eval_episodes) <= 0 or a.total_iters % a.eval_freq: p.error('counts must be positive and total-iters a multiple of eval-freq')
path = Path(a.demo).expanduser().resolve()
if not path.is_file() or not Path(a.data_manifest).is_file(): p.error('dataset or data manifest not found')
V.load_pretrained_state(a.pretrained)                      # verifies the file and its SHA256 before anything else
with h5py.File(path, 'r') as f:
    if len([k for k in f if k.startswith('traj_')]) < a.num_demos: p.error('fewer replayed trajectories than requested')
run_name = 'policy/' + a.run
args = {**c['policy'], 'env_id': c['env']['id'], 'control_mode': c['env']['control_mode'], 'sim_backend': 'physx_cpu', 'obs_mode': 'rgb',
        'max_episode_steps': c['env']['horizon'], 'exp_name': run_name, 'seed': a.seed, 'demo_path': str(path), 'num_demos': a.num_demos, 'total_iters': a.total_iters,
        'batch_size': a.batch_size, 'eval_freq': a.eval_freq, 'num_eval_envs': 1, 'num_eval_episodes': a.num_eval_episodes, 'capture_video': False, 'track': False}
argv = []
for k, v in args.items():
    key = '--' + k.replace('_', '-')
    if isinstance(v, bool): argv.append(key if v else '--no-' + k.replace('_', '-'))
    elif isinstance(v, list): argv.extend([key, *map(str, v)])
    else: argv.extend([key, str(v)])
script = Path(a.upstream).resolve() / 'examples/baselines/diffusion_policy/train_rgbd.py'
print('UPSTREAM:', script); print('ARGS:', argv); print('ENCODER:', V.ARCH_ID); print('WHITELIST: RGB + qpos/qvel/TCP pose only')
if not a.execute: print('DRY RUN: add --execute (through scripts/vision_v1/run.sh, after authorisation).'); raise SystemExit(0)
if not a.plan or not a.auth_run: p.error('--execute needs --plan and --auth-run')
V.require_authorization(a.auth_run, a.plan); V.require_plan_code(a.plan)      # also when this file is called directly, not only through run.sh
load_upstream(a.upstream)                                  # exact commit, importability
out = Path('runs') / run_name; out.mkdir(parents=True, exist_ok=False)
from hci.io import runtime_info
from hci.training import install_checkpoint_hook, source_fingerprint
meta = {'args': args, 'config': c, 'upstream_commit': UPSTREAM_COMMIT, 'observation_schema': V.OBSERVATION_SCHEMA, 'architecture_id': V.ARCH_ID, 'encoder': V.SPEC,
        'pretrained_file_sha256': sha256(a.pretrained), 'probe': bool(a.probe), 'demo_sha256': sha256(path), 'demo_path': str(path), 'data_manifest_sha256': sha256(a.data_manifest),
        'data_manifest': a.data_manifest, 'project_source_sha256': source_fingerprint('.'), 'full_checkpoint_freq': a.eval_freq, 'runtime': runtime_info(),
        'training_note': 'Official algorithm and trainer; observation whitelist and ordering patched in memory as for the legacy policy; the visual encoder class is '
                         'replaced in memory by hci.vision_v1.ResNet18TwoCamEncoder. Nothing else differs from the legacy training command.'}
write_json(out / 'policy_meta_training.json', meta)
from hci.observations import install_training_patches
install_training_patches()
plain = importlib.import_module('diffusion_policy.plain_conv'); made = []
plain.PlainConv = V.encoder_factory(a.pretrained, made)    # the trainer's `from diffusion_policy.plain_conv import PlainConv` now yields the ResNet encoder
install_checkpoint_hook(out / 'checkpoints', {k: v for k, v in meta.items() if k != 'runtime'}, a.eval_freq)
os.environ['WANDB_MODE'] = 'disabled'
sys.argv = [str(script), *argv]
try:
    runpy.run_path(str(script), run_name='__main__')
finally:
    ok = len(made) == 2 and all(isinstance(m, V.ResNet18TwoCamEncoder) and m.pretrained_loaded for m in made)
    write_json(out / 'encoder_build_check.json', {'encoders_built': len(made), 'expected': 2, 'meaning': 'agent and ema_agent', 'all_resnet18_with_pretrained_weights': bool(ok)})
    if not ok: raise SystemExit('The trainer did not build exactly two pretrained ResNet-18 encoders; this run is not a Vision-V1 run')
