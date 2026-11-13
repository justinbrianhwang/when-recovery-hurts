"""Regression: install_training_patches must import on the pinned ManiSkill install.

Found on the first real GPU host (train stack_plumbing_s1): `import mani_skill.utils.wrappers.flatten as flatten`
raised ImportError because `mani_skill.utils` is shadowed by `mani_skill.envs.utils`.
Skipped where ManiSkill / the official diffusion_policy baseline are not installed.
"""
import importlib
import pytest


def test_training_patches_install_and_target_upstream_modules():
    pytest.importorskip('gymnasium')
    pytest.importorskip('mani_skill')
    # The baseline package is importable through the pinned source tree, exactly as hci.policies.load_upstream does.
    import sys
    from pathlib import Path
    baseline = Path(__file__).resolve().parents[1] / 'third_party/ManiSkill/examples/baselines/diffusion_policy'
    if baseline.is_dir() and str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    pytest.importorskip('diffusion_policy')
    import mani_skill.envs  # same import state as the training launcher
    from hci import observations as o
    flatten = importlib.import_module('mani_skill.utils.wrappers.flatten')
    u = importlib.import_module('diffusion_policy.utils')
    saved = (flatten.FlattenRGBDObservationWrapper, u.build_state_obs_extractor, u.convert_obs)
    try:
        o.install_training_patches()
        assert flatten.FlattenRGBDObservationWrapper is o.StrictRGBObservationWrapper
        assert u.build_state_obs_extractor is o.build_strict_extractor
        assert u.convert_obs is not saved[2]
    finally:
        flatten.FlattenRGBDObservationWrapper, u.build_state_obs_extractor, u.convert_obs = saved
