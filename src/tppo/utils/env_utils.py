"""
Generalizes environments to allow agent interaction with 
continual-foragax - a research environment written in 
gymnax - and the rest of gymnax. This includes generalizing
continuous and discrete state spaces and action dimensions. 
"""

import gymnax
from foragax import registry 

def make_env(env_id:str, **kwargs): 
    """
    Makes the environment depending on whether it is from foragax or gymnax. 

    Returns a tuple: (env, env_params).
    """
    foragax_envs = registry.ENV_CONFIGS.keys()
    gymnax_envs = gymnax.registered_envs

    if env_id in foragax_envs: 
        env = registry.make(env_id=env_id, **kwargs)
        env_params = env.default_params

    elif env_id in gymnax_envs: 
        env, env_params = gymnax.make(env_id=env_id, **kwargs)
       
    else: 
        raise ValueError(f"Env id {env_id} is not a gymnax or foragax environment.")

    return env, env_params

def get_env_specs(env, env_params):
    """
    Retrieves information of the environment. 

    Returns 
        - env_type: "discrete" or "continuous" 
        - action_dim
    """ 
    action_space = env.action_space(env_params)

    if isinstance(action_space, gymnax.environments.spaces.Discrete): 
        env_type = "discrete"
        action_dim = action_space.n 
    elif isinstance(action_space, gymnax.environments.spaces.Box): 
        env_type = "continuous"
        action_dim = action_space.shape[0]
    else:
        raise ValueError("Environment is not supported by gymnax.")

    return env_type, action_dim