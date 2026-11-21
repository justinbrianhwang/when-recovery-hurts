"""Phase C1 Option A: checks of the logging-only spatial probe (hci.spatial_probe) on the REAL pinned simulator (ManiSkill
a4a4f92, SAPIEN 3.0.3, PhysX CPU), without policy, cameras or GPU. Free, local; no new scene: only already-used Phase B
development scenes, driven by the actions their stored traces recorded.

  invariance   for each scene: the trunk and every recorded arm branch, executed with the probe OFF and ON in the order
               OFF, ON, OFF, ON on one environment. After every control step the full simulator state (get_state_dict),
               the observation, reward, terminated/truncated and info must be bit-identical between all four runs, and
               every random-number state (environment RNGs, numpy, torch, python) must be the same before and after each
               probe call. Tolerance 0: no tolerance is used anywhere.
  timing       no logger / the earlier logger / the spatial logger on the same branches, rotating order: the added time
  replay       the trunk and the arm branches of the representative cases, executed once with the probe ON; writes trace
               files in the new record format, labelled LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS (not a C1 record, not the
               original run, no RGB), and the largest difference of the replayed tcp / cube positions to the stored trace.
               Stored Phase B records are only read, never changed or extended.

The environment is the one of hci.simulation.make_env except obs_mode='state' and render_mode=None (no renderer is
available here; rendering does not enter the physics step). What this does NOT show: the GPU host, the policy and its
inference, rendering, and the full C1 driver; those are covered by the mock tests and by the server probe check.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, random, sys, time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PHASE_B = REPO / 'results' / 'vision_center_v2_001' / 'execution'
CASES = {50632: 'shadow', 51004: 'pilot', 51009: 'pilot', 50892: 'shadow', 50959: 'pilot', 50702: 'shadow'}
LABEL = 'LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS'


def make_env():
    import gymnasium as gym
    import mani_skill.envs  # registration only
    from hci.io import load_config
    e = load_config(str(REPO / 'configs' / 'stackcube.yaml'))['env']
    env = gym.make(e['id'], num_envs=1, sim_backend='physx_cpu', obs_mode='state', control_mode=e['control_mode'], robot_uids=e['robot_uid'], reward_mode='sparse',
                   render_mode=None, reconfiguration_freq=1).unwrapped
    make_env.last = env; return env


def stored(seed):
    part = CASES.get(seed)
    if part is None:
        part = next(p for p in ('shadow', 'pilot') if (PHASE_B / p / 'scenes' / f'{seed}.json').exists())
    rec = json.loads((PHASE_B / part / 'scenes' / f'{seed}.json').read_text(encoding='utf-8'))
    with gzip.open(PHASE_B / part / rec['trace'], 'rt', encoding='utf-8') as f: tr = json.load(f)['traces']
    return part, rec, tr


def branches(rec, tr):
    """[(name, actions)]: the trunk, and each arm = trunk prefix up to the trigger + the arm's own logged actions."""
    trunk = [r['action'] for r in tr['trunk']]; out = [('trunk', trunk)]
    if rec.get('triggered'):
        t = int(rec['trigger']['step'])
        for c in rec['cells']:
            rows = tr.get(c['arm'])
            if rows: out.append((c['arm'], trunk[:t] + [r['action'] for r in rows]))
    return out


def rng_states(env):
    import torch
    from hci.simulation import rng_state
    s = {'numpy': np.random.get_state(), 'python': random.getstate(), 'torch': torch.get_rng_state().numpy().tobytes()}
    if torch.cuda.is_available(): s['torch_cuda'] = [x.numpy().tobytes() for x in torch.cuda.get_rng_state_all()]      # absent on the CPU-only local machine
    for k in ('_episode_rng', '_batched_episode_rng', '_main_rng', '_batched_main_rng'):
        if hasattr(env, k): s[k] = rng_state(getattr(env, k))
    return s


def same_rng(a, b):
    import pickle
    return pickle.dumps(a) == pickle.dumps(b)


def flat(x, pre=''):
    """Every array in a nested dict as bytes, so equality is bitwise (NaN == NaN only if the bytes are the same)."""
    import torch
    if isinstance(x, dict):
        out = {}
        for k in sorted(x): out.update(flat(x[k], f'{pre}/{k}'))
        return out
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    a = np.asarray(x); return {pre: (a.dtype.str, a.shape, a.tobytes())}


def run_branch(env, seed, actions, probe_on, checks=True):
    """Execute one branch; return per-step digests of state/obs/reward/info, the probe rows, and RNG violations.
    checks=False: no digests and no RNG comparison (a timing run: env.step and the probe only)."""
    import torch
    from hci import spatial_probe as SP
    obs, info = env.reset(seed=int(seed)); steps = []; rows = []; rng_bad = []
    def digest(*parts):
        h = hashlib.sha256()
        for p in parts:
            for k, (dt, sh, b) in sorted(flat(p).items()): h.update(k.encode()); h.update(dt.encode()); h.update(str(sh).encode()); h.update(b)
        return h.hexdigest()
    if checks: steps.append(digest(env.get_state_dict(), obs, info))
    if probe_on:
        r0 = rng_states(env) if checks else None; rows.append(SP.probe_spatial(env, 0, None))
        if checks and not same_rng(r0, rng_states(env)): rng_bad.append(0)
    for i, a in enumerate(actions, 1):
        obs, rew, term, trunc, info = env.step(torch.as_tensor(np.asarray(a, np.float32)[None], device=env.device))
        if probe_on:
            r0 = rng_states(env) if checks else None; rows.append(SP.probe_spatial(env, i, [float(z) for z in np.asarray(a, np.float32)]))
            if checks and not same_rng(r0, rng_states(env)): rng_bad.append(i)
        if checks: steps.append(digest(env.get_state_dict(), obs, {'rew': rew, 'term': term, 'trunc': trunc}, info))
    return steps, rows, rng_bad


def guard(out):
    out = Path(out)
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    if REPO / 'results' in out.resolve().parents and 'vision_center_v2_001' in out.resolve().parts: raise SystemExit('Refused: output inside the stored Phase B records')
    out.mkdir(parents=True); return out


def cmd_invariance(a):
    out = guard(a.out); env = make_env(); report = {'what': 'probe OFF/ON/OFF/ON on the real pinned simulator; bitwise; tolerance 0', 'scenes': {}, 'label': LABEL}; ok = True; started = time.time()
    try:
        for seed in a.seeds:
            part, rec, tr = stored(seed); res = {}
            for name, acts in branches(rec, tr):
                runs = [run_branch(env, seed, acts, on) for on in (False, True, False, True)]
                eq = all(r[0] == runs[0][0] for r in runs); first = None
                if not eq:
                    first = next(i for i in range(len(runs[0][0])) if len({r[0][i] for r in runs}) > 1)
                rng = runs[1][2] + runs[3][2]
                res[name] = {'steps': len(acts), 'state_obs_reward_info_bitwise_equal_all_4_runs': eq, 'first_differing_step': first, 'rng_changed_by_probe_at_steps': rng,
                             'probe_rows_equal_between_the_two_on_runs': json.dumps(runs[1][1], sort_keys=True) == json.dumps(runs[3][1], sort_keys=True), 'final_state_digest': runs[0][0][-1]}
                ok &= eq and not rng and res[name]['probe_rows_equal_between_the_two_on_runs']
                print(json.dumps({'seed': seed, 'branch': name, **{k: v for k, v in res[name].items() if k != 'final_state_digest'}}), flush=True)
            report['scenes'][str(seed)] = {'part': part, 'branches': res}
    finally:
        env.close()
    report.update(pass_=ok, seconds=time.time() - started, runtime=runtime(), spatial_probe_sha256=hashlib.sha256((REPO / 'src' / 'hci' / 'spatial_probe.py').read_bytes()).hexdigest(), rng_states_compared=sorted(rng_states(make_env.last)) if getattr(make_env, 'last', None) else None)
    report['pass'] = report.pop('pass_'); (out / 'invariance.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({'pass': ok, 'seconds': report['seconds']}))
    if not ok: raise SystemExit(2)


def cmd_replay(a):
    out = guard(a.out); env = make_env(); summary = {'label': LABEL, 'note': 'stored Phase B actions replayed with the probe ON; not a C1 record, not the original run; no RGB; Phase B records unchanged', 'scenes': {}}
    (out / 'traces').mkdir(); t_on = t_off = 0.; n = 0
    try:
        for seed in a.seeds:
            part, rec, tr = stored(seed); traces = {}; fid = {}
            for name, acts in branches(rec, tr):
                _, rows, rng = run_branch(env, seed, acts, True)                    # the record, with the RNG check
                t0 = time.perf_counter(); run_branch(env, seed, acts, False, checks=False); t_off += time.perf_counter() - t0      # timing runs: no checks, OFF then ON
                t0 = time.perf_counter(); run_branch(env, seed, acts, True, checks=False); t_on += time.perf_counter() - t0; n += len(acts)
                if rng: raise RuntimeError(f'{seed} {name}: the probe changed an RNG state at steps {rng}')
                ref = tr['trunk'] if name == 'trunk' else tr['trunk'][:int(rec['trigger']['step'])] + tr[name]
                d = max(float(np.max(np.abs(np.subtract(x[q], y[q])))) for x, y in zip(rows[1:], ref) for q in ('tcp', 'cubeA', 'cubeB'))
                fid[name] = {'steps': len(acts), 'max_abs_position_difference_to_stored_trace_m': d}
                traces[name] = rows
            p = out / 'traces' / f'trace_{seed}.json.gz'
            with gzip.open(p, 'xt', encoding='utf-8') as f: json.dump({'seed': seed, 'label': LABEL, 'source_record': f'{part}/scenes/{seed}.json', 'trigger_step': rec['trigger']['step'] if rec.get('triggered') else None,
                                                                         'traces': traces}, f)
            summary['scenes'][str(seed)] = {'part': part, 'fidelity': fid, 'trace': p.name, 'trace_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'trace_bytes': p.stat().st_size}
            print(json.dumps({'seed': seed, 'fidelity': fid}), flush=True)
    finally:
        env.close()
    summary['timing'] = {'env_steps_each': n, 'seconds_probe_on': t_on, 'seconds_probe_off': t_off, 'ms_per_step_on': 1e3 * t_on / n, 'ms_per_step_off': 1e3 * t_off / n, 'ms_added_per_step': 1e3 * (t_on - t_off) / n,
                         'how': 'separate timing runs without state digests or RNG checks (reset + env.step [+ probe]); OFF then ON per branch',
                         'note': 'local WSL CPU, no rendering and no policy: the added milliseconds per step are the logging cost; its share of a C1 step is smaller, since a C1 step also renders two cameras and runs the policy'}
    summary['runtime'] = runtime(); (out / 'replay_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8'); print(json.dumps(summary['timing']))


def time_branch(env, seed, actions, probe):
    """Seconds for reset + every env.step, with `probe` (None, or a function(env, t, action)) after each step and at reset."""
    import torch
    t0 = time.perf_counter(); env.reset(seed=int(seed))
    if probe: probe(env, 0, None)
    for i, a in enumerate(actions, 1):
        env.step(torch.as_tensor(np.asarray(a, np.float32)[None], device=env.device))
        if probe: probe(env, i, [float(z) for z in np.asarray(a, np.float32)])
    return time.perf_counter() - t0


def cmd_timing(a):
    """No logger / the earlier logger (hci.recovery_v1.probe, what C1 logs under plan dd3af27e) / the spatial logger, on the
    same branches, in rotating order, `--repeats` times. The cost that Option A adds is spatial minus earlier."""
    from hci import recovery_v1 as RV
    from hci import spatial_probe as SP
    out = guard(a.out); env = make_env(); kinds = {'none': None, 'earlier_logger': RV.probe, 'spatial_logger': SP.probe_spatial}; sec = {k: 0. for k in kinds}; n = 0; per = []
    try:
        for seed in a.seeds:
            part, rec, tr = stored(seed)
            for name, acts in branches(rec, tr):
                for rep in range(a.repeats):
                    order = list(kinds)[rep % 3:] + list(kinds)[:rep % 3]
                    for k in order: dt = time_branch(env, seed, acts, kinds[k]); sec[k] += dt; per.append({'seed': seed, 'branch': name, 'rep': rep, 'kind': k, 'steps': len(acts), 'seconds': dt})
                    n += len(acts)
    finally:
        env.close()
    ms = {k: 1e3 * v / n for k, v in sec.items()}
    rep = {'env_steps_per_kind': n, 'ms_per_step': ms, 'ms_added_by_spatial_over_earlier_logger': ms['spatial_logger'] - ms['earlier_logger'], 'ms_earlier_logger_over_none': ms['earlier_logger'] - ms['none'],
           'how': 'reset + env.step (+ logger at reset and after every step); no digests, no RNG checks; rotating order of the three kinds', 'runs': per, 'runtime': runtime(),
           'note': 'local WSL CPU, state observations, no rendering, no policy: absolute numbers are for this machine; the server probe check measures the real ratio'}
    (out / 'timing.json').write_text(json.dumps(rep, indent=1), encoding='utf-8'); print(json.dumps({k: v for k, v in rep.items() if k not in ('runs', 'runtime')}))


def runtime():
    import platform, sapien, torch, mani_skill
    return {'python': sys.version.split()[0], 'platform': platform.platform(), 'sapien': sapien.__version__, 'torch': torch.__version__, 'numpy': np.__version__, 'mani_skill_path': str(Path(mani_skill.__file__).parent)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    for n in ('invariance', 'replay', 'timing'):
        q = s.add_parser(n); q.add_argument('--out', required=True); q.add_argument('--seeds', type=int, nargs='+', default=list(CASES))
        if n == 'timing': q.add_argument('--repeats', type=int, default=3)
    a = p.parse_args(); {'invariance': cmd_invariance, 'replay': cmd_replay, 'timing': cmd_timing}[a.cmd](a)


if __name__ == '__main__':
    main()
