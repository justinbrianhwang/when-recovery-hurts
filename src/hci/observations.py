import gymnasium as gym
import torch
from .schema import state_parts, sensor_names

class StrictRGBObservationWrapper(gym.ObservationWrapper):
    """Same observation whitelist during official DP training and our evaluation."""
    def __init__(self,env,*args,**kwargs):
        super().__init__(env)
        self.base_env=env.unwrapped
        self.names=sensor_names(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))
    def observation(self,obs):
        return {'state':torch.cat(state_parts(obs),dim=-1).float(),
                'rgb':torch.cat([obs['sensor_data'][n]['rgb'] for n in self.names],dim=-1)}

def build_strict_extractor(env_id):
    return state_parts

def install_training_patches():
    """Explicitly replace only the observation paths; do not modify upstream files."""
    # Not `import mani_skill.utils.wrappers.flatten as flatten`: at the pinned commit the attribute
    # `mani_skill.utils` is shadowed by `mani_skill.envs.utils`, so attribute traversal raises ImportError.
    import importlib
    flatten=importlib.import_module('mani_skill.utils.wrappers.flatten')
    u=importlib.import_module('diffusion_policy.utils')
    flatten.FlattenRGBDObservationWrapper=StrictRGBObservationWrapper
    u.build_state_obs_extractor=build_strict_extractor
    original=u.convert_obs
    def ordered_convert(obs,*args,**kwargs):
        obs=dict(obs);obs['sensor_data']={n:obs['sensor_data'][n] for n in sorted(obs['sensor_data'])}
        return original(obs,*args,**kwargs)
    u.convert_obs=ordered_convert
