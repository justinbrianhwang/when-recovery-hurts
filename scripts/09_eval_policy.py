"""Untouched base-policy success on a declared development seed set, through the project Runtime (deployment path).

Development only: refuses any seed that belongs to the calibration or Test split. A `policy_qualify*` set can be
evaluated once; a second attempt is refused so that a qualification result cannot be re-drawn for another candidate.
Primary metric: success_once (absorbing), horizon from the config. `--success-at-end` additionally keeps executing
the policy to the horizon after the first success and reports success at the last step as a secondary metric.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
from hci.io import load_config, read_json, write_json, new_run, finish_run, runtime_info, sha256, digest
from hci.statistics import wilson_interval


def resolve_seeds(manifest, name, main_splits, demo_seeds=(), limit=None):
    """Seeds of `name`, after checking that no policy-eval seed collides with any other declared use."""
    if name not in manifest['splits']: raise KeyError(f'Unknown policy-eval set: {name}')
    seen = {}
    for source, table in (('splits', main_splits['splits']), ('policy_eval', manifest['splits'])):
        for split, ids in table.items():
            if len(ids) != len(set(ids)): raise ValueError(f'Duplicate seed inside {source}/{split}')
            for i in ids:
                if i in seen: raise ValueError(f'Seed {i} is declared in both {seen[i]} and {source}/{split}')
                seen[i] = f'{source}/{split}'
    clash = sorted(set(demo_seeds) & {i for t in manifest['splits'].values() for i in t})
    if clash: raise ValueError(f'Policy-eval seeds are also demonstration scenes: {clash[:10]}')
    ids = list(manifest['splits'][name])
    if limit is not None:
        if limit <= 0: raise ValueError('episodes must be positive')
        ids = ids[:limit]
    return ids


def summarize(rows, horizon):
    n = len(rows); k = int(sum(r['success_once'] for r in rows)); lo, hi = wilson_interval(k, n)
    calls = sum(r['policy_calls'] for r in rows); sec = sum(r['policy_inference_seconds'] for r in rows)
    out = {'n': n, 'success_once': k, 'success_once_rate': k / n if n else None, 'wilson95': [lo, hi],
           'mean_steps_to_completion': float(np.mean([r['steps'] for r in rows])) if n else None,
           'mean_steps_successful': float(np.mean([r['steps'] for r in rows if r['success_once']])) if k else None,
           'policy_calls': int(calls), 'policy_inference_seconds': sec,
           'seconds_per_policy_call': sec / calls if calls else None,
           'wall_seconds_per_episode': float(np.mean([r['wall_seconds'] for r in rows])) if n else None, 'horizon': horizon}
    if rows and 'success_at_end' in rows[0]:
        e = int(sum(r['success_at_end'] for r in rows)); lo, hi = wilson_interval(e, n)
        out.update(success_at_end=e, success_at_end_rate=e / n, success_at_end_wilson95=[lo, hi])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--policy', required=True); p.add_argument('--out', required=True)
    p.add_argument('--set', required=True, dest='name'); p.add_argument('--manifest', default='configs/policy_eval_splits.json')
    p.add_argument('--splits', default='configs/splits.json'); p.add_argument('--upstream', default='third_party/ManiSkill')
    p.add_argument('--demo-manifest', help='demo manifest JSON; its scene seeds must not overlap the evaluation seeds')
    p.add_argument('--episodes', type=int); p.add_argument('--success-at-end', action='store_true')
    p.add_argument('--ledger', default='artifacts/ledger'); p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    a = p.parse_args(); c = load_config(a.config); manifest = read_json(a.manifest); main_splits = read_json(a.splits)
    demo_seeds = read_json(a.demo_manifest)['all_scene_seeds'] if a.demo_manifest else ()
    ids = resolve_seeds(manifest, a.name, main_splits, demo_seeds, a.episodes)
    from hci.experiment import runtime, fingerprint
    fp = fingerprint(c, a.policy)
    if a.name.startswith('policy_qualify'):
        if a.episodes is not None: p.error('A qualification set is evaluated in full')
        # Exclusive create: a second qualification on the same seeds (e.g. for another candidate) is refused.
        write_json(Path(a.ledger) / f'{a.name}.json', {'set': a.name, 'out': a.out, **fp, 'policy': a.policy,
                   'note': 'Qualification seeds are single-use. This file marks them as consumed.'})
    out = new_run(a.out, 'policy-eval-development', c); r = runtime(c, a.policy, a.upstream, a.device); rows = []
    started = time.perf_counter()
    try:
        with (out / 'episodes.jsonl').open('x', encoding='utf-8') as f:
            for seed in ids:
                tick = time.perf_counter(); r.reset(seed); res = r.complete()
                row = {'seed': seed, 'success_once': int(res['success']), 'steps': res['steps'], 'reason': res['reason'],
                       'policy_calls': res['policy_calls'], 'policy_inference_seconds': res['policy_inference_seconds']}
                if a.success_at_end:
                    # Same convention as the upstream evaluator (ignore_terminations): keep acting until the horizon.
                    while r.t < r.horizon: r.done = False; r.step_base()
                    row['success_at_end'] = int(r.y); row['steps_executed_total'] = r.t
                row['wall_seconds'] = time.perf_counter() - tick; rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
                print(f"{a.name} seed={seed} success_once={row['success_once']} steps={row['steps']} "
                      f"({sum(x['success_once'] for x in rows)}/{len(rows)})", flush=True)
        s = summarize(rows, c['env']['horizon'])
        report = {'set': a.name, 'seeds': ids, **s, **fp, 'policy': a.policy, 'policy_eval_manifest_sha256': sha256(a.manifest),
                  'split_manifest_sha256': digest(main_splits), 'success_definition': c['protocol']['success_definition'],
                  'source_type': 'real_sim', 'wall_seconds': time.perf_counter() - started, 'runtime': runtime_info(),
                  'scope': 'Development measurement of one fixed policy without intervention. Not calibration, not the final Test. '
                           'The interval is descriptive; it is not a guarantee about the population success rate.'}
        write_json(out / 'policy_eval.json', report); finish_run(out, n=s['n'], success_once=s['success_once'])
        print(json.dumps({k: report[k] for k in ('set', 'n', 'success_once', 'wilson95', 'mean_steps_to_completion', 'seconds_per_policy_call')}, indent=2))
    finally:
        r.env.close()


if __name__ == '__main__':
    main()
