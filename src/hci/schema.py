"""A fixed sensor whitelist. Never use task-object poses as online features."""
STATE_PATHS=(('agent','qpos'),('agent','qvel'),('extra','tcp_pose'))

def state_parts(obs):
    out=[]
    for parent,key in STATE_PATHS:
        if parent not in obs or key not in obs[parent]:
            raise KeyError(f'Missing required sensor field {parent}/{key}')
        out.append(obs[parent][key])
    return out

def sensor_names(obs):
    names=sorted(obs['sensor_data'])
    if not names: raise ValueError('No camera found')
    for n in names:
        if 'rgb' not in obs['sensor_data'][n]: raise ValueError(f'No RGB in {n}')
    return names
