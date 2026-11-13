"""Full training-state checkpoint and its evaluate() hook. CPU only; does not run the upstream trainer."""
import random, sys, types
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from hci.training import save_full_checkpoint, install_checkpoint_hook, pack_rng, unpack_rng, source_fingerprint


class FakeEMA:
    def __init__(self, params): self.shadow = [p.detach().clone() for p in params]; self.optimization_step = 0
    def state_dict(self): return {'decay': .9999, 'optimization_step': self.optimization_step, 'power': .75, 'shadow_params': self.shadow}


def trainer(steps=3):
    agent = torch.nn.Linear(4, 2); ema_agent = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(agent.parameters(), lr=1e-3); sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 / (1 + i))
    ema = FakeEMA(agent.parameters())
    for _ in range(steps):
        opt.zero_grad(); agent(torch.randn(8, 4)).pow(2).mean().backward(); opt.step(); sch.step(); ema.optimization_step += 1
    return {'agent': agent, 'ema_agent': ema_agent, 'ema': ema, 'optimizer': opt, 'lr_scheduler': sch}


def test_full_checkpoint_contains_training_state_and_loads_weights_only(tmp_path):
    g = trainer(3); path = save_full_checkpoint(g, 2, tmp_path, {'args': {'batch_size': 256}, 'demo_sha256': 'x'})
    assert path.name == 'full_000002.pt'
    s = torch.load(path, map_location='cpu', weights_only=True)      # the export script loads with weights_only=True
    assert {'agent', 'ema_agent', 'ema', 'optimizer', 'lr_scheduler', 'rng', 'meta', 'resume_scope'} <= set(s)
    assert s['iteration_tag'] == 2 and s['updates_completed'] == 3    # tag i means i+1 updates in the 0-based upstream loop
    assert s['optimizer']['state'] and s['lr_scheduler']['last_epoch'] == 3 and len(s['ema']['shadow_params']) == 2
    assert 'sample order non-identical' in s['resume_scope']
    for k, v in g['agent'].state_dict().items(): assert torch.equal(s['agent'][k], v)


def test_full_checkpoint_never_overwrites_and_requires_all_objects(tmp_path):
    g = trainer(1); save_full_checkpoint(g, 10, tmp_path, {})
    with pytest.raises(FileExistsError): save_full_checkpoint(g, 10, tmp_path, {})
    g.pop('optimizer')
    with pytest.raises(KeyError): save_full_checkpoint(g, 20, tmp_path, {})


def test_rng_streams_round_trip():
    random.seed(3); np.random.seed(4); torch.manual_seed(5); s = pack_rng()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(9); np.random.seed(9); torch.manual_seed(9); unpack_rng(s)
    assert a == (random.random(), np.random.rand(), torch.rand(1).item())


def test_hook_saves_at_multiples_and_end_then_runs_the_original_evaluator(tmp_path, monkeypatch):
    calls = []; fake = types.ModuleType('diffusion_policy.evaluate')
    fake.evaluate = lambda n, agent, envs, device, backend: calls.append(n) or {'success_once': [0.]}
    pkg = types.ModuleType('diffusion_policy'); pkg.evaluate = fake
    monkeypatch.setitem(sys.modules, 'diffusion_policy', pkg); monkeypatch.setitem(sys.modules, 'diffusion_policy.evaluate', fake)
    install_checkpoint_hook(tmp_path, {'k': 1}, every=20)
    # Same shape as the upstream script: module globals hold the trainer objects, a closure passes `iteration`.
    g = trainer(1); g['evaluate'] = fake.evaluate
    exec("def evaluate_and_save_best(iteration):\n    return evaluate(5, ema_agent, None, 'cpu', 'physx_cpu')", g)
    for i in (0, 10, 20, 30, 40): g['evaluate_and_save_best'](i)
    assert calls == [5] * 5                                           # the upstream evaluation always still runs
    assert sorted(p.name for p in tmp_path.glob('*.pt')) == ['full_000020.pt', 'full_000040.pt']
    exec("def other(iteration):\n    return evaluate(5, agent, None, 'cpu', 'x')", g)
    with pytest.raises(RuntimeError): g['other'](60)                 # refuses to checkpoint from an unexpected call site


def test_source_fingerprint_is_stable_and_content_sensitive(tmp_path):
    (tmp_path / 'src/hci').mkdir(parents=True); f = tmp_path / 'src/hci/a.py'; f.write_text('x=1\n')
    a = source_fingerprint(tmp_path); assert a == source_fingerprint(tmp_path)
    f.write_bytes(b'x=1\r\n'); assert a == source_fingerprint(tmp_path)   # line endings do not change the fingerprint
    f.write_text('x=2\n'); assert a != source_fingerprint(tmp_path)
