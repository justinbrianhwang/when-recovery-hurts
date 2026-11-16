"""Recovery-V1 diagnostic: mechanism labels on synthetic traces, seed gates, the server file list, cost projection.

CPU only, hand-written traces. These tests say nothing about how the real policy fails.
"""
import importlib.util, json, re
from pathlib import Path
import pytest
from hci import recovery_v1 as RV

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery_v1_diag', ROOT / 'scripts' / '30_recovery_v1_diag.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
B = [0.10, 0.0, 0.02]                    # cubeB rests on the table
A0 = [0.0, 0.0, 0.02]
PLACE = [0.10, 0.0, 0.06]                # cubeA resting on cubeB


def row(t, A, *, grasped=False, grip=-1.0, tcp=None, static=True, success=False, Bpos=B, q=(1, 0, 0, 0), v=(0, 0, 0), fingers=0.0):
    tcp = list(A) if tcp is None else tcp; xy = ((A[0] - Bpos[0]) ** 2 + (A[1] - Bpos[1]) ** 2) ** .5
    on = xy <= RV.XY_TOL_M and abs(A[2] - Bpos[2] - RV.CUBE_M) <= RV.Z_TOL_M
    return {'t': t, 'tcp': tcp, 'cubeA': list(A), 'cubeB': list(Bpos), 'tcp_to_cubeA': sum((a - b) ** 2 for a, b in zip(tcp, A)) ** .5, 'cubeA_to_cubeB_xy': xy, 'is_cubeA_grasped': grasped,
            'is_cubeA_on_cubeB': on, 'is_cubeA_static': static, 'success': success, 'action': None if t == 0 else [0., 0., 0., grip], 'tcp_q': [0, 1, 0, 0], 'finger_qpos': [0.02, 0.02],
            'cubeA_q': list(q), 'cubeB_q': [1, 0, 0, 0], 'cubeA_v': list(v), 'cubeA_w': [0, 0, 0], 'cubeB_v': [0, 0, 0], 'cubeB_w': [0, 0, 0], 'f_A_B': 0., 'f_A_left': fingers, 'f_A_right': fingers,
            'f_A_table': 0., 'f_B_left': 0., 'f_B_right': 0.}


def trace(*parts):
    """parts: (number of steps, cubeA position, keyword arguments of row)"""
    rows = []
    for n, A, kw in parts:
        for _ in range(n): rows.append(row(len(rows) + 1, A, **kw))
    return row(0, A0, tcp=[0, 0, 0.2]), rows


CARRY = [(5, A0, dict(tcp=[0, 0, 0.2], grip=1.0)), (5, A0, dict(grasped=True)), (10, [0.05, 0, 0.10], dict(grasped=True))]
CASES = {
    'NO_APPROACH': [(30, A0, dict(tcp=[0, 0, 0.2], grip=1.0))],
    'NO_GRASP': [(30, A0, dict(grip=-1.0))],
    'GRASP_NOT_LIFTED': [(5, A0, dict(grip=1.0)), (25, A0, dict(grasped=True))],
    'NOT_RELEASED_AT_PLACE': CARRY + [(20, PLACE, dict(grasped=True))],
    'HOLDING_NOT_PLACED': CARRY + [(20, [0.10, 0, 0.10], dict(grasped=True))],
    'GRASP_LOST_WITHOUT_OPEN_COMMAND': CARRY + [(10, [0.05, 0, 0.02], dict(grip=-1.0))],
    'RELEASE_OFF_TARGET_XY': CARRY + [(3, [0.15, 0, 0.06], dict(grasped=True)), (10, [0.15, 0, 0.02], dict(grip=1.0))],
    'EARLY_RELEASE_ABOVE': CARRY + [(3, [0.10, 0, 0.09], dict(grasped=True)), (10, [0.13, 0.03, 0.02], dict(grip=1.0))],
    'RELEASE_BELOW_REST_HEIGHT': CARRY + [(3, [0.10, 0.03, 0.045], dict(grasped=True)), (10, [0.13, 0.03, 0.02], dict(grip=1.0))],
    'POST_RELEASE_UNSTABLE': CARRY + [(3, PLACE, dict(grasped=True)), (2, PLACE, dict(grip=1.0, static=False)), (10, [0.14, 0, 0.02], dict(grip=1.0))],
    'IN_PLACE_NOT_STATIC': CARRY + [(3, PLACE, dict(grasped=True)), (10, PLACE, dict(grip=1.0, static=False))],
    'UNKNOWN': CARRY + [(3, PLACE, dict(grasped=True)), (10, PLACE, dict(grip=1.0, static=True))],
    'SUCCESS': CARRY + [(3, PLACE, dict(grasped=True)), (1, PLACE, dict(grip=1.0, success=True))],
}


@pytest.mark.parametrize('label', list(CASES))
def test_each_rule_on_a_trace_built_for_it(label):
    v = RV.classify(*trace(*CASES[label])); assert v['label'] == label and v['meaning'] == RV.LABELS[label]
    json.dumps(v, allow_nan=False)                                        # every verdict and its evidence can be written to a file


def test_every_label_has_a_case_and_an_order():
    assert set(CASES) == set(RV.LABELS) == set(RV.RULE_ORDER) and len(RV.RULE_ORDER) == len(RV.LABELS)


def test_evidence_carries_the_numbers_of_the_verdict():
    e = RV.classify(*trace(*CASES['EARLY_RELEASE_ABOVE']))['evidence']
    assert e['release_step'] == 23 and e['height_gap_at_release_m'] == pytest.approx(0.03) and e['xy_offset_at_release_m'] == pytest.approx(0.0) and 1.0 in e['gripper_commands_around_release']
    e = RV.classify(*trace(*CASES['NOT_RELEASED_AT_PLACE']))['evidence']
    assert e['in_place_steps_while_grasped'] == 20 and e['open_command_steps_in_hold'] == 0 and e['final_gripper_command'] == -1.0


def test_open_command_without_release_is_still_not_released():
    v = RV.classify(*trace(*CARRY, (10, PLACE, dict(grasped=True)), (10, PLACE, dict(grasped=True, grip=1.0))))
    assert v['label'] == 'NOT_RELEASED_AT_PLACE' and v['evidence']['open_command_steps_in_hold'] == 10       # the label states the fact; who failed (policy or gripper) is in the evidence


def test_short_flicker_of_the_grasp_flag_is_not_a_release():
    v = RV.classify(*trace(*CARRY, (2, [0.05, 0, 0.10], dict(grasped=False)), (20, [0.10, 0, 0.10], dict(grasped=True))))
    assert v['label'] == 'HOLDING_NOT_PLACED' and v['flags']['holds'] == 1


def test_last_carry_decides_after_a_second_attempt():
    v = RV.classify(*trace(*CARRY, (5, [0.05, 0, 0.02], dict(grip=1.0)), (5, [0.05, 0, 0.02], dict(grasped=True)), (5, [0.05, 0, 0.10], dict(grasped=True)), (3, PLACE, dict(grasped=True)),
                           (10, [0.14, 0, 0.02], dict(grip=1.0))))
    assert v['label'] == 'POST_RELEASE_UNSTABLE' and v['flags']['carried_holds'] == 2


def test_cube_b_displacement_is_flagged_not_labelled():
    moved = [0.13, 0.0, 0.02]
    v = RV.classify(*trace(*CARRY, (20, [0.10, 0, 0.10], dict(grasped=True, Bpos=moved))))
    assert v['flags']['cubeB_moved'] and v['label'] == 'HOLDING_NOT_PLACED'


def test_grasp_lost_in_the_last_steps_is_unknown_not_a_hold():
    v = RV.classify(*trace(*CARRY, (20, PLACE, dict(grasped=True)), (2, PLACE, dict(grip=1.0, static=False))))
    assert v['label'] == 'UNKNOWN' and v['evidence']['steps_without_grasp_at_end'] == 2


def test_release_evidence_names_both_steps():
    e = RV.classify(*trace(*CASES['EARLY_RELEASE_ABOVE']))['evidence']
    assert (e['release_step'], e['first_step_without_grasp'], e['release_measured_at']) == (23, 24, 'last grasped step') and e['height_gap_at_first_step_without_grasp_m'] == pytest.approx(-0.04)


def test_run_needs_a_budget_gate_of_the_same_plan():
    src = (ROOT / 'scripts' / '30_recovery_v1_diag.py').read_text(encoding='utf-8')
    assert "bg.get('plan_sha256') != sha256(a.plan)" in src and "plan_sha256=sha256(a.plan), seconds_per_step" in src


def test_empty_trace_is_unknown():
    assert RV.classify(row(0, A0), [])['label'] == 'UNKNOWN'


def test_tilt():
    assert RV.tilt_deg([1, 0, 0, 0]) == pytest.approx(0) and RV.tilt_deg([2 ** -.5, 2 ** -.5, 0, 0]) == pytest.approx(0, abs=1e-5)           # a quarter turn puts another face up
    assert RV.tilt_deg([0.9238795, 0.3826834, 0, 0]) == pytest.approx(45, abs=1e-3)


def test_summary_counts_and_tie():
    eps = [{'seed': i, 'success_once': 0, 'label': l} for i, l in enumerate(['UNKNOWN', 'EARLY_RELEASE_ABOVE', 'EARLY_RELEASE_ABOVE', 'NO_GRASP', 'NO_GRASP'])] + [{'seed': 9, 'success_once': 1, 'label': 'SUCCESS'}]
    s = RV.summarize(eps); assert (s['n'], s['success_once'], s['failures'], s['unknown']) == (6, 1, 5, 1) and s['largest_is_tied'] and s['failure_labels']['NO_GRASP']['seeds'] == [3, 4]
    assert not RV.summarize(eps[1:4])['largest_is_tied']


# ---------------------------------------------------------------- seeds
def manifest():
    return {'splits': {k: list(range(lo, hi + 1)) for k, (lo, hi) in RV.SETS.items()}}


def test_diag_seeds_and_reserved_range():
    assert RV.diag_seeds(manifest(), 'recovery_v1_diag', {'x': [50599, 50800]}) == list(range(50600, 50700))
    with pytest.raises(KeyError): RV.diag_seeds(manifest(), 'recovery_v1_pilot_reserved', {})
    with pytest.raises(ValueError, match='recorded use'): RV.diag_seeds(manifest(), 'recovery_v1_diag', {'earlier': [50650]})
    with pytest.raises(ValueError, match='recorded use'): RV.diag_seeds(manifest(), 'recovery_v1_diag', {'earlier': [50750]})          # the reserved range must be unused as well
    m = manifest(); m['splits']['recovery_v1_diag'][0] = 50599
    with pytest.raises(ValueError, match='frozen range'): RV.diag_seeds(m, 'recovery_v1_diag', {})


def test_committed_manifest_has_no_recorded_use():
    need = [S.MANIFEST, *S.SEED_SOURCES.values(), S.DEMO_MANIFEST, *S.PRIOR_SCENES]
    if not all((ROOT / f).exists() for f in need): pytest.skip('seed records of earlier stages are not on this machine')
    import os; cwd = os.getcwd(); os.chdir(ROOT)
    try: assert RV.diag_seeds(json.loads((ROOT / S.MANIFEST).read_text()), 'recovery_v1_diag', S.used_ledger()) == list(range(50600, 50700))
    finally: os.chdir(cwd)


# ---------------------------------------------------------------- server file list, ledger, cost
def test_every_repository_path_in_the_driver_is_in_the_server_list_or_is_local_only():
    """Vision-V1 run 2 stopped once because a file read on the server was not uploaded. Every path constant of the driver is accounted for here."""
    src = (ROOT / 'scripts' / '30_recovery_v1_diag.py').read_text(encoding='utf-8')
    paths = set(re.findall(r"'((?:results|runs|configs|artifacts|src|scripts)/[\w./-]+\.(?:json|jsonl|yaml|py|sh|txt|pt|tgz))'", src))
    local_only = {S.BACKUP['archive'], S.BACKUP['file_list'], S.BACKUP['source_checkpoint'], S.VISION['plan'], S.VISION['selection'], S.VISION['paired'], S.LEDGER}       # read by `plan` only, or written
    assert paths - local_only <= set(S.SERVER_READS), sorted(paths - local_only - set(S.SERVER_READS))
    assert not set(S.SERVER_READS) & {S.BACKUP['archive'], S.LEDGER}


def test_probe_seeds_are_used_select_scenes_and_not_diagnostic_ones():
    assert all(50400 <= s <= 50499 for s in S.PROBE_SEEDS) and not set(S.PROBE_SEEDS) & set(range(50600, 50800))


def test_run_refuses_without_authorization(tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    with pytest.raises(SystemExit, match='no authorization'): S.require_authorization(tmp_path, tmp_path / 'plan.json')


def test_cost_projection():
    d = RV.projected_cost(seconds_per_episode=30, usd_per_hour=0.45, spent_usd=0.30, cap_usd=2.0)
    assert d['projected_total_usd'] == pytest.approx(0.30 + (100 * 30 + 900) * 1.25 / 3600 * 0.45 + 0.20) and d['fits_cap']
    assert not RV.projected_cost(seconds_per_episode=30, usd_per_hour=0.45, spent_usd=1.6, cap_usd=2.0)['fits_cap']
