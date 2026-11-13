import importlib.util, pathlib, pytest
pytest.importorskip('imageio')
spec = importlib.util.spec_from_file_location('ft', pathlib.Path(__file__).parents[1] / 'scripts' / '12_failure_trace.py')
ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft)

def row(t, d=0.2, grasp=False, z=0.02, xy=0.2, on=False, ok=False):
    return {'t': t, 'tcp_to_cubeA': d, 'is_cubeA_grasped': grasp, 'cubeA': [0, 0, z], 'cubeB': [xy, 0, 0.02],
            'cubeA_to_cubeB_xy': xy, 'is_cubeA_on_cubeB': on, 'success': ok, 'action': [0, 0, 0, 1]}

def test_stage_labels():
    assert ft.classify([])[0] == 'unreadable'
    assert ft.classify([row(1), row(2)])[0] == 'approach'
    assert ft.classify([row(1), row(2, d=0.01)])[0] == 'grasp'
    assert ft.classify([row(1), row(2, d=0.01, grasp=True)])[0] == 'grasp'  # closed on it but never lifted
    assert ft.classify([row(1), row(2, d=0.01, grasp=True, z=0.06)])[0] == 'transport_align'
    assert ft.classify([row(1), row(2, d=0.01, grasp=True, z=0.06, xy=0.01)])[0] == 'place_stack'
    label, facts = ft.classify([row(1), row(2, d=0.01, grasp=True, z=0.06, xy=0.01, on=True, ok=True)])
    assert label == 'success' and facts['first_grasp_step'] == 2 and facts['first_on_cubeB_step'] == 2
