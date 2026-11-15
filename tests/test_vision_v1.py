"""Vision-V1: the ResNet-18 encoder contract, its fit into the unchanged upstream agent, export/reload parity, seed gates,
the single-use qualification guard, outcome labels.

CPU only, random tensors and synthetic records. These tests say nothing about manipulation success, GPU memory or speed.
Agent-level tests need the pinned upstream Diffusion Policy source (third_party/ManiSkill, or HCI_UPSTREAM_DP_DIR) and
are skipped without it; simulator packages that the upstream file imports but the Agent does not use are stubbed when absent.
"""
import copy, importlib.abc, importlib.machinery, importlib.util, inspect, json, os, sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import numpy as np
import pytest
torch = pytest.importorskip('torch'); pytest.importorskip('torchvision')
from hci import vision_v1 as V
from hci.io import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / 'artifacts' / 'pretrained' / V.PRETRAINED['file']
needs_pretrained = pytest.mark.skipif(not PRE.is_file(), reason='the ImageNet ResNet-18 file is not on this machine')
ARGS = NS(obs_horizon=2, act_horizon=8, pred_horizon=16, diffusion_step_embed_dim=64, unet_dims=[64, 128, 256], n_groups=8)
STATE_DIM = 25


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), ROOT / 'scripts' / name); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S = _script('29_vision_v1.py')


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stands in for simulator-side packages only when they are not installed (appended last, so real packages win)."""
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] == 'mani_skill': return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = MagicMock(); m.__name__ = spec.name; m.__path__ = []; m.__spec__ = spec; m.__loader__ = self; return m

    def exec_module(self, module): pass


@pytest.fixture(scope='module')
def upstream():
    for d in (os.environ.get('HCI_UPSTREAM_DP_DIR'), ROOT / 'third_party' / 'ManiSkill' / 'examples' / 'baselines' / 'diffusion_policy'):
        if d and (Path(d) / 'train_rgbd.py').is_file(): break
    else: pytest.skip('pinned upstream Diffusion Policy source not available')
    pytest.importorskip('diffusers'); pytest.importorskip('gymnasium')
    if not any(isinstance(f, _Stub) for f in sys.meta_path): sys.meta_path.append(_Stub())
    if importlib.util.find_spec('tensorboard') is None: sys.modules.setdefault('torch.utils.tensorboard', MagicMock())      # the trainer's log writer; the Agent does not use it
    if str(d) not in sys.path: sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location('_vision_v1_test_upstream', Path(d) / 'train_rgbd.py'); m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); m.device = torch.device('cpu'); return m


def spaces():
    import gymnasium as gym
    return NS(single_action_space=gym.spaces.Box(-1, 1, (4,), np.float32), single_observation_space=gym.spaces.Dict({
        'state': gym.spaces.Box(-np.inf, np.inf, (2, STATE_DIM), np.float32), 'rgb': gym.spaces.Box(0, 255, (2, 128, 128, 6), np.uint8)}))


def agent_of(upstream, arch, pretrained=None):
    if arch == 'legacy': return upstream.Agent(spaces(), ARGS)
    made = []
    with V.encoder_swapped(upstream, pretrained, made): a = upstream.Agent(spaces(), ARGS)
    assert len(made) == 1 and a.visual_encoder is made[0]; return a


def obs(seed=0, batch=1):
    g = torch.Generator().manual_seed(seed)
    return {'state': torch.randn((batch, 2, STATE_DIM), generator=g), 'rgb': torch.randint(0, 256, (batch, 2, 128, 128, 6), generator=g, dtype=torch.uint8)}


def act(agent, o, seed=7):
    torch.manual_seed(seed); return agent.get_action({k: v.clone() for k, v in o.items()})


# ---------------------------------------------------------------- encoder contract
def test_output_shape_and_refusal_of_any_other_input():
    enc = V.ResNet18TwoCamEncoder(in_channels=6, out_dim=256, pool_feature_map=True).eval(); x = torch.rand(3, 6, 128, 128)
    z = enc(x); assert z.shape == (3, 256) and (z >= 0).all() and enc.out_dim == 256                     # Linear + ReLU, as the legacy encoder
    for bad in (torch.rand(3, 3, 128, 128), torch.rand(3, 6, 224, 224), torch.rand(3, 6, 128, 96), torch.rand(6, 128, 128), torch.randint(0, 255, (3, 6, 128, 128), dtype=torch.uint8)):
        with pytest.raises(ValueError): enc(bad)                                                        # one camera, another size, missing batch, uint8: nothing is resized, dropped or converted silently
    for kw in ({'in_channels': 3}, {'in_channels': 8}, {'out_dim': 128}, {'pool_feature_map': False}, {'last_act': False}):
        with pytest.raises(ValueError): V.ResNet18TwoCamEncoder(**{'in_channels': 6, 'out_dim': 256, 'pool_feature_map': True, **kw})
    assert V.SPEC['resize'] == 'none' and V.SPEC['cameras'] == ['base_camera', 'hand_camera'] and V.SPEC['feature_dim'] == 256


def test_two_cameras_keep_their_order_and_share_one_backbone():
    torch.manual_seed(0); enc = V.ResNet18TwoCamEncoder().eval(); a = torch.rand(2, 3, 128, 128); b = torch.rand(2, 3, 128, 128)
    ab = enc(torch.cat([a, b], 1)); ba = enc(torch.cat([b, a], 1)); assert not torch.allclose(ab, ba)       # swapping the cameras changes the feature: order matters
    feat = lambda x: enc.backbone((x - enc.mean) / enc.std)
    want = enc.fc(torch.cat([feat(a), feat(b)], 1)); assert torch.allclose(ab, want, atol=1e-6)             # base camera first, hand camera second, the same backbone for both
    assert sum(1 for _ in enc.backbone.modules() if isinstance(_, torch.nn.Conv2d)) == 20 and enc.backbone.conv1.in_channels == 3      # one ResNet-18, 3-channel stem (no 6-channel stem from scratch)
    same = enc(torch.cat([a, a], 1)); assert not torch.allclose(same, ab)                                   # the second camera is used, not dropped


def test_imagenet_normalisation_is_applied_per_camera():
    enc = V.ResNet18TwoCamEncoder().eval(); seen = []
    enc.backbone.register_forward_pre_hook(lambda m, inp: seen.append(inp[0].clone()))
    x = torch.rand(1, 6, 128, 128); enc(x); got = seen[0]; mean = torch.tensor(V.IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(V.IMAGENET_STD).view(1, 3, 1, 1)
    assert got.shape == (2, 3, 128, 128) and torch.allclose(got[0], ((x[:, 0:3] - mean) / std)[0]) and torch.allclose(got[1], ((x[:, 3:6] - mean) / std)[0])
    assert (V.IMAGENET_MEAN, V.IMAGENET_STD) == ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) and 'mean' not in enc.state_dict()      # constants, not weights


@needs_pretrained
def test_pretrained_weights_are_loaded_verified_and_never_replaced_by_random_ones(tmp_path):
    from torchvision.models import resnet18
    ref = torch.load(PRE, map_location='cpu', weights_only=True); enc = V.ResNet18TwoCamEncoder(pretrained=PRE); sd = enc.backbone.state_dict()
    assert enc.pretrained_loaded and all(torch.equal(sd[k], v) for k, v in ref.items() if not k.startswith('fc.')) and len(sd) == len(ref) - 2 + 20      # no fc; 20 BatchNorm step counters that the old file format does not carry
    rand = V.ResNet18TwoCamEncoder(); assert not rand.pretrained_loaded and not torch.equal(rand.backbone.conv1.weight, ref['conv1.weight'])
    assert sha256(PRE) == V.PRETRAINED['sha256'] and V.PRETRAINED['url'].endswith(V.PRETRAINED['file']) and sum(v.numel() for k, v in ref.items() if 'num_batches' not in k and 'running' not in k) == 11689512
    with pytest.raises(FileNotFoundError): V.ResNet18TwoCamEncoder(pretrained=tmp_path / 'missing.pth')            # missing file: an error, not random initialisation
    bad = tmp_path / 'resnet18.pth'; torch.save(resnet18(weights=None).state_dict(), bad)
    with pytest.raises(ValueError): V.ResNet18TwoCamEncoder(pretrained=bad)                                          # another file with the right shape: refused by hash
    (tmp_path / 'junk.pth').write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError): V.load_pretrained_state(tmp_path / 'junk.pth')


def test_batchnorm_statistics_never_move_and_everything_else_trains():
    torch.manual_seed(0); enc = V.ResNet18TwoCamEncoder(); enc.train(); bn = [m for m in enc.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(bn) == 20 and enc.training and not any(m.training for m in bn)                                 # train() leaves the BatchNorm layers in eval mode
    before = {k: v.clone() for k, v in enc.state_dict().items()}; opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    for _ in range(2): opt.zero_grad(); enc(torch.rand(2, 6, 128, 128)).pow(2).mean().backward(); opt.step()
    after = enc.state_dict(); moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert not any('running_' in k or 'num_batches' in k for k in moved)                                      # statistics untouched in training mode
    assert 'backbone.conv1.weight' in moved and 'backbone.layer4.1.conv2.weight' in moved and 'backbone.bn1.weight' in moved and 'fc.0.weight' in moved      # stem, last block, BN affine, head all train
    assert all(p.requires_grad for p in enc.parameters()) and all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in enc.named_parameters() if n.startswith('backbone.conv1') or n.startswith('fc.0'))
    a = enc.train()(torch.ones(1, 6, 128, 128) * .3); b = enc.eval()(torch.ones(1, 6, 128, 128) * .3); assert torch.equal(a, b)        # train and eval mode give the same feature


# ---------------------------------------------------------------- inside the unchanged upstream agent
def test_the_unet_and_everything_but_the_encoder_are_the_legacy_ones(upstream):
    torch.manual_seed(0); new = agent_of(upstream, 'v1'); torch.manual_seed(0); old = agent_of(upstream, 'legacy')
    assert isinstance(new.visual_encoder, V.ResNet18TwoCamEncoder) and type(old.visual_encoder).__name__ == 'PlainConv' and upstream.PlainConv.__name__ == 'PlainConv'      # the swap is undone
    shapes = lambda a: {k: tuple(v.shape) for k, v in a.noise_pred_net.state_dict().items()}; assert shapes(new) == shapes(old)                # identical action U-Net
    cond = 2 * (256 + STATE_DIM); first = next(k for k in shapes(new) if k.endswith('cond_encoder.1.weight')); assert shapes(new)[first][1] == 64 + cond      # diffusion-step embedding + global condition
    assert new.encode_obs({'state': torch.zeros(3, 2, STATE_DIM), 'rgb': torch.zeros(3, 2, 6, 128, 128, dtype=torch.uint8)}, eval_mode=True).shape == (3, cond)
    assert (new.num_diffusion_iters, new.obs_horizon, new.act_horizon, new.pred_horizon, new.act_dim) == (100, 2, 8, 16, 4) == (old.num_diffusion_iters, old.obs_horizon, old.act_horizon, old.pred_horizon, old.act_dim)
    assert new.noise_scheduler.config == old.noise_scheduler.config
    c = V.parameter_counts(new); o = V.parameter_counts(old); assert c['noise_pred_net'] == o['noise_pred_net'] and c['visual_encoder'] == 11176512 + 1024 * 256 + 256 and c['total_trainable'] == c['total']
    assert not hasattr(new, 'aug')                                                                            # no augmentation in either path


def test_action_shape_and_same_seed_determinism(upstream):
    torch.manual_seed(1); a = agent_of(upstream, 'v1').eval(); o = obs(3)
    x = act(a, o, 5); y = act(a, o, 5); z = act(a, o, 6); assert x.shape == (1, 8, 4) and torch.equal(x, y) and not torch.equal(x, z) and torch.isfinite(x).all() and x.abs().max() <= 1
    o2 = obs(4); assert not torch.equal(act(a, o2, 5), x)                                                     # another observation, same noise: another chunk


def test_training_form_and_deployment_form_of_an_observation_give_the_same_condition(upstream):
    torch.manual_seed(2); a = agent_of(upstream, 'v1'); o = obs(5, batch=2)
    dataset_form = {'state': o['state'], 'rgb': o['rgb'].permute(0, 1, 4, 2, 3)}                              # what the upstream Dataset yields: channel-first uint8
    a.train(); t = a.encode_obs(dataset_form, eval_mode=False); a.eval(); d = a.encode_obs({'state': o['state'], 'rgb': o['rgb'].permute(0, 1, 4, 2, 3)}, eval_mode=True)
    assert torch.equal(t, d)                                                                                  # no train-only preprocessing, no BatchNorm drift
    hooked = []; a.visual_encoder.register_forward_pre_hook(lambda m, inp: hooked.append(inp[0]))
    act(a, {k: v[:1] for k, v in o.items()}); x = hooked[0]; assert x.shape == (2, 6, 128, 128) and x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1      # get_action: HWC uint8 -> CHW float in [0, 1]
    assert torch.equal(x, o['rgb'][:1].permute(0, 1, 4, 2, 3).flatten(end_dim=1).float() / 255.0)             # the camera channels arrive in their stored order


def test_gradients_reach_the_backbone_through_the_policy_loss(upstream):
    torch.manual_seed(3); a = agent_of(upstream, 'v1'); o = obs(6, batch=2)
    loss = a.compute_loss({'state': o['state'], 'rgb': o['rgb'].permute(0, 1, 4, 2, 3)}, torch.rand(2, 16, 4) * 2 - 1); loss.backward()
    g = lambda n: dict(a.named_parameters())[n].grad; assert torch.isfinite(loss) and all(g(n) is not None and g(n).abs().sum() > 0 for n in ('visual_encoder.backbone.conv1.weight', 'visual_encoder.backbone.layer4.1.conv2.weight', 'visual_encoder.fc.0.weight'))


@needs_pretrained
@pytest.mark.parametrize('arch', ['v1', 'legacy'])
def test_ema_export_and_strict_reload_reproduce_the_actions(upstream, arch, tmp_path):
    from diffusers.training_utils import EMAModel
    torch.manual_seed(4); pre = PRE if arch == 'v1' else None; agent = agent_of(upstream, arch, pre); ema_agent = agent_of(upstream, arch, pre)
    ema = EMAModel(parameters=agent.parameters(), power=0.75); opt = torch.optim.AdamW(agent.parameters(), lr=1e-3); o = obs(7, batch=2)
    for _ in range(3):
        opt.zero_grad(); agent.compute_loss({'state': o['state'], 'rgb': o['rgb'].permute(0, 1, 4, 2, 3)}, torch.rand(2, 16, 4) * 2 - 1).backward(); opt.step(); ema.step(agent.parameters())
    ema.copy_to(ema_agent.parameters()); torch.save(ema_agent.state_dict(), tmp_path / 'weights.pt'); state = torch.load(tmp_path / 'weights.pt', map_location='cpu', weights_only=True)
    fresh = agent_of(upstream, arch, None); fresh.load_state_dict(state, strict=True); ema_agent.eval(); fresh.eval(); q = obs(8)
    assert torch.equal(act(ema_agent, q), act(fresh, q))                                                      # the exported file alone reproduces the EMA policy (no pretrained file needed at deployment)
    if arch == 'v1':
        assert V.check_state_dict(state, PRE) == []                                                           # ResNet keys, 1024->256 head, BatchNorm statistics still ImageNet, convolutions moved
        with pytest.raises(RuntimeError): agent_of(upstream, 'legacy').load_state_dict(state, strict=True)     # never loadable as the legacy architecture
        untrained = agent_of(upstream, 'v1', PRE).state_dict(); assert any('was not trained' in p for p in V.check_state_dict(untrained, PRE))
        moved = dict(state); moved['visual_encoder.backbone.bn1.running_mean'] = state['visual_encoder.backbone.bn1.running_mean'] + 1; assert any('running statistics' in p for p in V.check_state_dict(moved, PRE))
    else:
        assert any('no ResNet backbone' in p for p in V.check_state_dict(state)) and any('legacy PlainConv' in p for p in V.check_state_dict(state))
        with pytest.raises(RuntimeError): agent_of(upstream, 'v1').load_state_dict(state, strict=True)


def test_architecture_dispatch_keeps_the_legacy_path_and_refuses_unknown_ids(tmp_path):
    from hci.policies import DiffusionPolicy
    leg = tmp_path / 'legacy'; leg.mkdir(); write_json(leg / 'policy_meta.json', {'observation_schema': V.OBSERVATION_SCHEMA}); assert V.architecture_of(leg) == V.LEGACY_ARCH
    new = tmp_path / 'new'; new.mkdir(); write_json(new / 'policy_meta.json', {'architecture_id': V.ARCH_ID}); assert V.architecture_of(new) == V.ARCH_ID
    odd = tmp_path / 'odd'; odd.mkdir(); write_json(odd / 'policy_meta.json', {'architecture_id': 'resnet34'})
    with pytest.raises(ValueError): V.architecture_of(odd)
    with pytest.raises(FileNotFoundError): V.architecture_of(tmp_path / 'missing')
    assert issubclass(V.VisionV1Policy, DiffusionPolicy) and V.VisionV1Policy.generate is DiffusionPolicy.generate and V.VisionV1Policy.convert is DiffusionPolicy.convert      # same seed rule, same observation path
    src = inspect.getsource(V.make_policy); assert 'DiffusionPolicy)(env, artifact, upstream, device=device)' in src
    real = ROOT / 'artifacts/policies/stack_1d_s1_d500_clean_v1_i080000'
    if real.is_dir(): assert V.architecture_of(real) == V.LEGACY_ARCH and 'architecture_id' not in read_json(real / 'policy_meta.json')      # the frozen legacy policy is untouched and still legacy
    with pytest.raises(ValueError): V.VisionV1Policy(None, leg, None)                                          # a legacy artifact is never loaded as Vision-V1
    write_json(new / 'policy_meta.json', {'architecture_id': V.ARCH_ID, 'observation_schema': V.OBSERVATION_SCHEMA, 'encoder': {**V.SPEC, 'resize': 'bilinear 224'}}, overwrite=True)
    with pytest.raises(ValueError): V.VisionV1Policy(None, new, None)                                          # another preprocessing than this code implements


# ---------------------------------------------------------------- seeds, single use, labels
MAN = {'splits': {'vision_v1_select': list(range(50400, 50500)), 'vision_v1_qualify': list(range(50500, 50600))}}
USED = {'splits/train': list(range(10000, 10120)), 'splits/calibration': list(range(30000, 30400)), 'splits/test': list(range(90000, 90400)), 'policy_eval/policy_qualify_B': list(range(50300, 50400)),
        'demonstration_scenes': list(range(0, 567))}


def test_new_seed_sets_are_frozen_disjoint_and_never_patched():
    assert S.new_seeds(MAN, 'vision_v1_select', USED) == list(range(50400, 50500)) and S.new_seeds(MAN, 'vision_v1_qualify', USED) == list(range(50500, 50600))
    for name in ('policy_qualify_B', 'policy_qualify_A', 'test', 'calibration', 'policy_select'):
        with pytest.raises(KeyError): S.new_seeds(MAN, name, USED)                                             # no other set can be reached through this driver
    for src in ({'x': [50450]}, {'splits/test': [50599]}, {'demonstration_scenes': [50400]}):
        with pytest.raises(ValueError): S.new_seeds(MAN, 'vision_v1_select', {**USED, **src})                  # one used seed stops everything; it is not replaced
    for bad in ({**MAN['splits'], 'vision_v1_select': list(range(50401, 50501))}, {**MAN['splits'], 'vision_v1_qualify': list(range(50500, 50599))}, {**MAN['splits'], 'extra': [1]},
                {'vision_v1_select': MAN['splits']['vision_v1_select']}, {**MAN['splits'], 'vision_v1_qualify': list(range(50499, 50599))}):
        with pytest.raises(ValueError): S.new_seeds({'splits': bad}, 'vision_v1_select', USED)
    real = read_json(ROOT / 'configs' / 'vision_v1_splits.json'); assert real['splits'] == MAN['splits']
    splits = read_json(ROOT / 'configs' / 'splits.json'); pe = read_json(ROOT / 'configs' / 'policy_eval_splits.json'); new = set(sum(real['splits'].values(), []))
    assert not new & {s for v in splits['splits'].values() for s in v} and not new & {s for v in pe['splits'].values() for s in v}
    src = inspect.getsource(S)
    for name in ('policy_qualify', 'calibration', "'test'"): assert name not in src.replace("'policy_qualify_A', 'policy_qualify_B', 'calibration', 'test'", '').replace('policy_qualify_B (not reused', '').replace('policy_qualify_B is a different', '').replace("['splits/calibration', 'splits/test']", '').replace('calibration and Test are compared by seed number only', '').replace("'calibration', 'final Test'", '').replace("'calibration': 'NOT_RUN'", '')


def test_selection_rule_gate_and_labels():
    assert V.select_seed({1: 55, 2: 61})['selected_train_seed'] == 2 and V.select_seed({1: 61, 2: 55})['selected_train_seed'] == 1 and V.select_seed({1: 58, 2: 58})['selected_train_seed'] == 1      # tie -> seed 1
    d = V.select_seed({1: 49, 2: 49}); assert d['selected_train_seed'] is None and d['status'] == 'NO_VIABLE_DEV_CANDIDATE'
    assert V.select_seed({1: 50, 2: 12})['status'] == 'SELECTED' and V.select_seed({1: 12, 2: 50})['selected_train_seed'] == 2
    for bad in ({1: 60}, {1: 60, 2: 60, 3: 99}, {1: 60, 3: 70}):
        with pytest.raises(ValueError): V.select_seed(bad)                                                     # exactly the two declared seeds; no third run can win
    L = V.outcome_label
    assert L(60, 50) == 'TARGET_MET' and L(75, 40) == 'TARGET_MET' and L(60, 51) == 'PROMISING_BELOW_TARGET' and L(59, 30) == 'PROMISING_BELOW_TARGET' and L(50, 10) == 'PROMISING_BELOW_TARGET'
    assert L(49, 10) == 'BELOW_GATE' and L(49, 60) == 'BELOW_GATE' and L(70, 65) == 'PROMISING_BELOW_TARGET'
    with pytest.raises(ValueError): L(30, 10, n=50)                                                            # no label from a partial set
    new = {s: int(s % 3 == 0) for s in range(100)}; old = {s: int(s % 2 == 0) for s in range(100)}; c = V.paired_counts(new, old)
    assert sum(c.values()) == 100 and c['both_succeed'] == 17 and c['new_only'] == 17 and c['legacy_only'] == 33 and c['both_fail'] == 33
    with pytest.raises(ValueError): V.paired_counts(new, {s: 0 for s in range(99)})


def test_qualification_is_single_use_and_execution_needs_authorisation(tmp_path):
    run = tmp_path / 'runs' / 'vision_v1_001'; run.mkdir(parents=True); plan = run / 'plan.json'; write_json(plan, {'x': 1})
    with pytest.raises(SystemExit): S.require_authorization(run, plan)
    for bad in ({'approved_by': 'agent', 'paid_compute_authorized': True, 'cap_usd': 3.0, 'plan_sha256': sha256(plan)}, {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 0, 'plan_sha256': sha256(plan)},
                {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 3.0, 'plan_sha256': 'the H1 plan'}, {'approved_by': 'user', 'cap_usd': 3.0, 'plan_sha256': sha256(plan)}):
        write_json(run / 'authorization.json', bad, overwrite=True)
        with pytest.raises(SystemExit): S.require_authorization(run, plan)
    write_json(run / 'authorization.json', {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 3.0, 'plan_sha256': sha256(plan)}, overwrite=True); assert S.require_authorization(run, plan)['cap_usd'] == 3.0
    for name in S.EXECUTION: assert 'require_authorization(' in inspect.getsource(getattr(S, 'cmd_' + name))
    for script in ('27_vision_v1_train.py', '28_vision_v1_export.py'):                                         # the trainer and the exporter ask too, even when called directly
        text = (ROOT / 'scripts' / script).read_text(encoding='utf-8'); assert 'V.require_authorization(a.auth_run, a.plan)' in text and 'V.require_plan_code(a.plan)' in text
    t = (ROOT / 'scripts' / '27_vision_v1_train.py').read_text(encoding='utf-8'); assert t.index('V.require_authorization(') < t.index('runpy.run_path(') and t.index('V.require_authorization(') < t.index('out.mkdir(')
    assert S.require_authorization is V.require_authorization
    for name in ('plan', 'dry_run'): assert 'require_authorization' not in inspect.getsource(getattr(S, 'cmd_' + name)) and 'runtime(' not in inspect.getsource(getattr(S, 'cmd_' + name))
    q = inspect.getsource(S.cmd_qualify_paired); ledger = q.index("'vision_v1_qualify.json'"); first_rollout = q.index('rollout_table(')
    assert ledger < first_rollout and 'overwrite' not in q and 'never resumed' in q                            # the set is marked consumed before any rollout; no overwrite, no resume
    led = tmp_path / 'ledger'; write_json(led / 'vision_v1_qualify.json', {'set': 'vision_v1_qualify'})
    with pytest.raises(FileExistsError): write_json(led / 'vision_v1_qualify.json', {'set': 'vision_v1_qualify'})      # the exclusive create that makes a second session impossible
    c = inspect.getsource(S.cmd_choose); assert 'qualification_ledger_exists' in c and c.index('qualification_ledger_exists') < c.index("write_json(run / 'selection.json'")      # no selection after the qualification


def test_budget_projection_blocks_instead_of_shrinking():
    ok = S.projected_cost(updates_per_second=12., usd_per_hour=0.25, spent_usd=0.10, cap_usd=3.0, reserve_usd=0.30, eval_seconds_per_episode=8.)
    assert ok['fits_cap'] and abs(ok['train_hours_both_runs'] - (2 * 100000 / 12 + 2 * 11 * 5 * 36) / 3600) < 1e-9 and abs(ok['evaluation_hours'] - 400 * 8 / 3600) < 1e-9
    slow = S.projected_cost(updates_per_second=3., usd_per_hour=0.25, spent_usd=0.10, cap_usd=3.0, reserve_usd=0.30, eval_seconds_per_episode=8.); assert not slow['fits_cap'] and slow['projected_total_usd'] > 3.0
    assert 'not reduced' in slow['rule'] and set(inspect.signature(S.projected_cost).parameters) == {'updates_per_second', 'usd_per_hour', 'spent_usd', 'cap_usd', 'reserve_usd', 'eval_seconds_per_episode', 'monitor_seconds_per_episode', 'fixed_seconds', 'margin'}
    assert abs(ok['monitoring_hours_bound'] - 2 * 11 * 5 * 36 / 3600) < 1e-9                                  # 11 upstream evaluations of 5 episodes per run, charged at a high bound
    assert S.TRAIN == {'seeds': [1, 2], 'num_demos': 500, 'total_iters': 100000, 'batch_size': 256, 'eval_freq': 10000, 'num_eval_episodes': 5, 'candidate_iteration': 80000, 'candidate_weights': 'EMA'}
