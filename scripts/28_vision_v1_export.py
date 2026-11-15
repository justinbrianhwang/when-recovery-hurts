"""Export the EMA weights of a Vision-V1 training checkpoint as a frozen policy, with the architecture id and the
encoder specification in the metadata. The legacy exporter (scripts/04_export_policy.py) is unchanged and refuses
nothing new; this one refuses anything that is not a Vision-V1 run or not the declared checkpoint tag."""
from pathlib import Path
import argparse, torch
from hci.io import read_json, write_json, sha256
from hci import vision_v1 as V

p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
p.add_argument('--pretrained', required=True); p.add_argument('--iteration', type=int, required=True, help='the predeclared checkpoint tag (80000)')
p.add_argument('--plan', required=True); p.add_argument('--auth-run', required=True, help='run folder holding authorization.json')
a = p.parse_args(); run = Path(a.run).resolve(); ckpt = Path(a.checkpoint).resolve()
V.require_authorization(a.auth_run, a.plan); V.require_plan_code(a.plan)
if a.iteration != 80000: p.error('the predeclared candidate is tag 80000; no other checkpoint is exported')
if not ckpt.is_relative_to(run / 'checkpoints') or ckpt.name != f'full_{a.iteration:06d}.pt': p.error('the checkpoint must be full_<iteration>.pt of this training run')
v = read_json(run / 'policy_meta_training.json'); build = read_json(run / 'encoder_build_check.json')
if v.get('architecture_id') != V.ARCH_ID or v.get('encoder') != V.SPEC or v.get('probe'): p.error('not a Vision-V1 candidate training run (or a probe run)')
if not build['all_resnet18_with_pretrained_weights']: p.error('the training run did not build pretrained ResNet-18 encoders')
state = torch.load(ckpt, map_location='cpu', weights_only=True)
if 'ema_agent' not in state: p.error('Expected the EMA weights in the checkpoint; refusing silent fallback')
if int(state['iteration_tag']) != a.iteration: p.error(f"checkpoint tag {state['iteration_tag']} is not {a.iteration}")
problems = V.check_state_dict(state['ema_agent'], a.pretrained)
if problems: p.error('; '.join(problems))
out = Path(a.out); out.mkdir(parents=True, exist_ok=False); torch.save(state['ema_agent'], out / 'weights.pt')
m = {'args': v['args'], 'observation_schema': v['observation_schema'], 'architecture_id': V.ARCH_ID, 'encoder': V.SPEC, 'upstream_commit': v['upstream_commit'],
     'env_id': v['config']['env']['id'], 'control_mode': v['config']['env']['control_mode'], 'horizon': v['config']['env']['horizon'], 'robot_uid': v['config']['env']['robot_uid'],
     'weights_sha256': sha256(out / 'weights.pt'), 'weights': 'EMA', 'iteration': a.iteration, 'train_seed': v['args']['seed'], 'source_checkpoint': str(ckpt),
     'source_checkpoint_sha256': sha256(ckpt), 'pretrained_file_sha256': v['pretrained_file_sha256'], 'state_dict_checks': 'ResNet backbone present; head Linear(1024, 256); '
     'BatchNorm running statistics equal to the ImageNet file; convolution weights differ from it', 'source_training_metadata': v}
write_json(out / 'policy_meta.json', m); print('Exported frozen Vision-V1 policy:', out, m['weights_sha256'][:16])
