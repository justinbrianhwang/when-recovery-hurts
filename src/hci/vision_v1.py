"""Vision-V1: an ImageNet-pretrained ResNet-18 visual encoder for the Diffusion Policy (new base-policy cycle).
See amendments/AMENDMENT_012 (draft). PREPARATION: nothing here has been trained or evaluated.

Only the visual encoder family changes. The pinned upstream `Agent` (action U-Net, DDPM scheduler, loss) is used
unchanged and is not edited on disk: the name `PlainConv` that the upstream module looks up when it builds its
encoder is replaced in memory, for training and for deployment, by `ResNet18TwoCamEncoder` with the same constructor
signature and the same output width (256). The legacy path (hci.policies.DiffusionPolicy, hci.experiment.runtime) is
not touched; a policy without `architecture_id` in its metadata is loaded exactly as before.

Encoder contract (fixed before any outcome):
  input    float tensor [N, 6, 128, 128] in [0, 1]: two RGB cameras concatenated on the channel axis in sorted camera
           name order (base_camera, hand_camera), exactly what the upstream agent passes to its encoder
  per cam  ImageNet mean/std normalisation, native 128x128 (no resize, no crop, no augmentation)
  backbone ONE torchvision ResNet-18 shared by both cameras (the two images are stacked on the batch axis),
           initialised from IMAGENET1K_V1, global average pooling, classification layer removed: 512 per camera
  head     concatenate the two cameras (1024) -> Linear(1024, 256) -> ReLU (the legacy encoder also ends in Linear+ReLU)
  training every convolution, every BatchNorm affine parameter and the head are trained jointly with the policy.
           BatchNorm layers always run with their ImageNet running statistics (never updated): the upstream EMA
           averages parameters only, so updated running statistics would not reach the exported EMA weights and
           training and deployment would disagree. This is a requirement of train/deploy consistency, recorded as
           the one deviation from "fine-tune everything".
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from .io import read_json, sha256
from .schema import sensor_names
from .policies import DiffusionPolicy, QueuePolicy, load_upstream

ARCH_ID = 'dp_resnet18_imagenet1k_v1_shared_twocam_fc256_v1'
LEGACY_ARCH = 'legacy_plainconv_v0'
OBSERVATION_SCHEMA = 'rgb_qpos_qvel_tcp_v1'
PRETRAINED = {'name': 'torchvision ResNet18_Weights.IMAGENET1K_V1', 'url': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
              'file': 'resnet18-f37072fd.pth', 'sha256': 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',
              'license': 'torchvision (BSD-3-Clause); weights trained on ImageNet-1k', 'num_params_with_fc': 11689512}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 128
CAMERAS = ('base_camera', 'hand_camera')
FEATURE_DIM = 256
SPEC = {'architecture_id': ARCH_ID, 'input': 'float [N, 6, 128, 128] in [0, 1]; channels 0-2 = base_camera RGB, 3-5 = hand_camera RGB (sorted camera names)',
        'cameras': list(CAMERAS), 'image_size': IMAGE_SIZE, 'resize': 'none', 'augmentation': 'none', 'normalisation': {'mean': list(IMAGENET_MEAN), 'std': list(IMAGENET_STD)},
        'backbone': 'torchvision resnet18, shared by both cameras, fc removed, global average pooling (512 per camera)', 'head': 'Linear(1024, 256) + ReLU',
        'feature_dim': FEATURE_DIM, 'batchnorm': 'eval mode in training and deployment (ImageNet running statistics, never updated); affine weight and bias trainable',
        'trainable': 'all backbone convolutions, BatchNorm affine parameters, head', 'pretrained': PRETRAINED}


def load_pretrained_state(path):
    """The verified ImageNet file, without the 1000-class layer. A wrong or missing file is an error, never a silent random init."""
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f'pretrained weights not found: {path} (random initialisation is not a substitute)')
    if sha256(path) != PRETRAINED['sha256']: raise ValueError(f'{path} is not the declared ResNet18 IMAGENET1K_V1 file (SHA256 differs)')
    state = torch.load(path, map_location='cpu', weights_only=True)
    return {k: v for k, v in state.items() if not k.startswith('fc.')}


class ResNet18TwoCamEncoder(nn.Module):
    """Drop-in for the upstream PlainConv(in_channels, out_dim, pool_feature_map, last_act)."""
    def __init__(self, in_channels=6, out_dim=FEATURE_DIM, pool_feature_map=True, last_act=True, pretrained=None):
        super().__init__()
        from torchvision.models import resnet18
        if in_channels != 3 * len(CAMERAS): raise ValueError(f'this encoder takes exactly two RGB cameras (6 channels), got {in_channels}; no camera is dropped or padded')
        if out_dim != FEATURE_DIM or not pool_feature_map or not last_act: raise ValueError('the downstream U-Net contract is 256 features from a pooled map with a final ReLU')
        self.out_dim = out_dim; self.backbone = resnet18(weights=None); self.backbone.fc = nn.Identity()
        if pretrained is not None:
            missing, unexpected = self.backbone.load_state_dict(load_pretrained_state(pretrained), strict=False)
            if missing or unexpected: raise ValueError(f'pretrained weights do not fit the backbone (missing {missing}, unexpected {unexpected})')
        self.pretrained_loaded = pretrained is not None
        self.fc = nn.Sequential(nn.Linear(512 * len(CAMERAS), out_dim), nn.ReLU())
        nn.init.zeros_(self.fc[0].bias)                                     # as the upstream encoder does for its layers
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self._freeze_bn_statistics()

    def _freeze_bn_statistics(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()

    def train(self, mode=True):
        super().train(mode); self._freeze_bn_statistics(); return self

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 6 or tuple(image.shape[2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f'expected [N, 6, {IMAGE_SIZE}, {IMAGE_SIZE}], got {tuple(image.shape)}; images are not resized or cropped')
        if not torch.is_floating_point(image): raise ValueError('expected a float image in [0, 1] (the upstream agent divides uint8 by 255)')
        n = image.shape[0]; cams = torch.cat([image[:, 0:3], image[:, 3:6]], 0)            # [2N, 3, H, W]: base cameras, then hand cameras
        z = self.backbone((cams - self.mean) / self.std)                                   # [2N, 512]
        return self.fc(torch.cat([z[:n], z[n:]], 1))                                       # [N, 1024] -> [N, 256]


def encoder_factory(pretrained, made=None):
    """What stands in for `PlainConv`. `made` collects the encoders that were built, so a caller can prove which class the agent got."""
    def build(in_channels=3, out_dim=FEATURE_DIM, pool_feature_map=False, last_act=True):
        enc = ResNet18TwoCamEncoder(in_channels=in_channels, out_dim=out_dim, pool_feature_map=pool_feature_map, last_act=last_act, pretrained=pretrained)
        if made is not None: made.append(enc)
        return enc
    return build


@contextmanager
def encoder_swapped(module, pretrained, made=None):
    """Temporarily replace the encoder class a module will instantiate. The original name is restored on exit, so a
    legacy agent built afterwards from the same module is the legacy agent."""
    if not hasattr(module, 'PlainConv'): raise AttributeError(f'{module.__name__} has no PlainConv to replace; the pinned upstream source changed')
    original = module.PlainConv; module.PlainConv = encoder_factory(pretrained, made)
    try: yield
    finally: module.PlainConv = original


def parameter_counts(agent):
    n = lambda m: int(sum(p.numel() for p in m.parameters())); t = lambda m: int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    return {'visual_encoder': n(agent.visual_encoder), 'visual_encoder_trainable': t(agent.visual_encoder), 'noise_pred_net': n(agent.noise_pred_net), 'total': n(agent), 'total_trainable': t(agent)}


def check_state_dict(state, pretrained=None):
    """An exported Vision-V1 state dict: ResNet keys present, no legacy encoder keys, BatchNorm statistics still the ImageNet ones."""
    keys = set(state); problems = []
    if not any(k.startswith('visual_encoder.backbone.layer4.') for k in keys): problems.append('no ResNet backbone in the state dict')
    if any(k.startswith('visual_encoder.cnn.') for k in keys): problems.append('legacy PlainConv keys in a Vision-V1 state dict')
    if tuple(state.get('visual_encoder.fc.0.weight', torch.empty(0)).shape) != (FEATURE_DIM, 1024): problems.append('head is not Linear(1024, 256)')
    if pretrained is not None:
        ref = load_pretrained_state(pretrained)
        moved = [k for k in ref if 'running_' in k and not torch.equal(state['visual_encoder.backbone.' + k].cpu(), ref[k])]
        if moved: problems.append(f'{len(moved)} BatchNorm running statistics differ from the ImageNet file (they must never be updated)')
        same = [k for k in ref if k.endswith('conv1.weight') or k.endswith('conv2.weight')]
        if all(torch.equal(state['visual_encoder.backbone.' + k].cpu(), ref[k]) for k in same): problems.append('no convolution weight differs from the ImageNet file (the backbone was not trained)')
    return problems


class VisionV1Policy(DiffusionPolicy):
    """Deployment adapter for a Vision-V1 policy. Same queue, history, observation whitelist, camera order, seed rule and
    inference path as DiffusionPolicy; only the agent's visual encoder class differs. Loading is strict."""
    def __init__(self, env, artifact, upstream, device='cuda'):
        self.artifact = Path(artifact); self.meta = read_json(self.artifact / 'policy_meta.json')
        if self.meta.get('architecture_id') != ARCH_ID: raise ValueError(f"not a Vision-V1 policy (architecture_id={self.meta.get('architecture_id')!r})")
        if self.meta['observation_schema'] != OBSERVATION_SCHEMA: raise ValueError('Policy observation whitelist mismatch')
        if self.meta.get('encoder') != SPEC: raise ValueError('the encoder specification stored with the policy differs from this code; refusing to guess the preprocessing')
        if sha256(self.artifact / 'weights.pt') != self.meta['weights_sha256']: raise ValueError('Checkpoint hash mismatch')
        a = self.meta['args']; QueuePolicy.__init__(self, a['act_horizon'], a['obs_horizon'])
        self.device = torch.device(device); self.cameras = sensor_names(env._init_raw_obs)
        if list(self.cameras) != list(CAMERAS): raise ValueError(f'camera order {self.cameras} is not {list(CAMERAS)}')
        raw = self.convert(env._init_raw_obs)
        import gymnasium as gym
        st = raw['state'].shape[-1]; shape = tuple(raw['rgb'].shape[1:])
        if shape != (IMAGE_SIZE, IMAGE_SIZE, 6): raise ValueError(f'RGB observation is {shape}, not ({IMAGE_SIZE}, {IMAGE_SIZE}, 6)')
        spaces = SimpleNamespace(single_action_space=env.single_action_space, single_observation_space=gym.spaces.Dict({
            'state': gym.spaces.Box(-np.inf, np.inf, (self.obs_horizon, st), np.float32), 'rgb': gym.spaces.Box(0, 255, (self.obs_horizon, *shape), np.uint8)}))
        module = load_upstream(upstream); made = []
        with encoder_swapped(module, None, made):          # no pretrained file is needed at deployment: every weight comes from the strict load
            self.agent = module.Agent(spaces, SimpleNamespace(**a)).to(self.device).eval()
        if len(made) != 1 or self.agent.visual_encoder is not made[0]: raise RuntimeError('the agent was not built with the Vision-V1 encoder')
        weights = torch.load(self.artifact / 'weights.pt', map_location=self.device, weights_only=True)
        self.agent.load_state_dict(weights, strict=True); self.adim = env.single_action_space.shape[0]; self.reset(raw)


def architecture_of(artifact):
    """Architecture of an exported policy. A policy exported before Vision-V1 has no `architecture_id` and is the legacy one."""
    arch = read_json(Path(artifact) / 'policy_meta.json').get('architecture_id', LEGACY_ARCH)
    if arch not in (ARCH_ID, LEGACY_ARCH): raise ValueError(f'unknown architecture_id {arch!r}; refusing to load it as something else')
    return arch


def make_policy(env, artifact, upstream, device='cuda'):
    return (VisionV1Policy if architecture_of(artifact) == ARCH_ID else DiffusionPolicy)(env, artifact, upstream, device=device)


def runtime(c, policy, upstream, device='cuda'):
    """hci.experiment.runtime with the architecture dispatch. For a legacy policy it builds the same objects through the same class."""
    from .simulation import make_env, Runtime
    env = make_env(c)
    try:
        p = make_policy(env, policy, upstream, device); m = p.meta
        for key, val in [('env_id', c['env']['id']), ('control_mode', c['env']['control_mode']), ('horizon', c['env']['horizon']), ('robot_uid', c['env']['robot_uid'])]:
            if m[key] != val: raise ValueError(f'Policy metadata mismatch: {key} {m[key]} != {val}')
        for key in ('obs_horizon', 'act_horizon', 'pred_horizon'):
            if m['args'][key] != c['policy'][key]: raise ValueError(f'Policy horizon mismatch: {key}')
        return Runtime(env, p, c)
    except Exception:
        env.close(); raise


def require_authorization(run, plan_path):
    """Execution needs a user-written authorisation that names the plan by hash and carries a cap. Nothing in this
    repository writes that file. Called by every entry point that can train, export or evaluate."""
    p = Path(run) / 'authorization.json'
    if not p.exists(): raise SystemExit(f'Refused: no authorization ({p}). paid_compute: NOT_AUTHORIZED. Execution needs a separate explicit approval by the user.')
    a = read_json(p); cap = a.get('cap_usd')
    if a.get('approved_by') != 'user' or a.get('paid_compute_authorized') is not True or not isinstance(cap, (int, float)) or isinstance(cap, bool) or not cap > 0:
        raise SystemExit('Refused: the authorization is not an explicit user approval with a positive spending cap')
    if a.get('plan_sha256') != sha256(plan_path): raise SystemExit('Refused: the authorization belongs to another plan')
    return a


def require_plan_code(plan_path):
    """The code files of the plan still have the planned hashes."""
    bad = [f for f, h in read_json(plan_path)['code'].items() if sha256(f) != h]
    if bad: raise SystemExit(f'Refused: these differ from the plan: {bad}')


# ---------------------------------------------------------------- evaluation bookkeeping (no simulator needed)
OUTCOME_LABELS = ('TARGET_MET', 'PROMISING_BELOW_TARGET', 'BELOW_GATE')


def select_seed(counts, gate=50):
    """counts: {train_seed: successes of the fixed 80k EMA candidate on vision_v1_select}. More successes wins; a tie goes to seed 1.
    If no candidate reaches the gate the qualification set is not consumed."""
    if set(counts) != {1, 2}: raise ValueError('exactly the two declared training seeds are compared')
    best = 1 if counts[1] >= counts[2] else 2
    return {'selected_train_seed': best if max(counts.values()) >= gate else None, 'success_once': {str(k): int(v) for k, v in sorted(counts.items())},
            'gate': gate, 'status': 'SELECTED' if max(counts.values()) >= gate else 'NO_VIABLE_DEV_CANDIDATE',
            'rule': 'more success_once on vision_v1_select; tie -> seed 1; both below the gate -> stop before qualification'}


def paired_counts(new, old):
    """new, old: {scene: 0/1} on the same scenes."""
    if set(new) != set(old): raise ValueError('the two policies were not evaluated on the same scenes')
    c = {'both_succeed': 0, 'new_only': 0, 'legacy_only': 0, 'both_fail': 0}
    for s in new: c[('both_succeed' if old[s] else 'new_only') if new[s] else ('legacy_only' if old[s] else 'both_fail')] += 1
    return c


def outcome_label(new_successes, legacy_successes, n=100):
    """Fixed engineering labels on vision_v1_qualify; not a significance test."""
    if n != 100: raise ValueError('the labels are defined for the full 100-scene qualification set')
    if new_successes >= 60 and new_successes - legacy_successes >= 10: return 'TARGET_MET'
    return 'PROMISING_BELOW_TARGET' if new_successes >= 50 else 'BELOW_GATE'
