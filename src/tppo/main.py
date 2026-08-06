import jax 
import jax.numpy as jnp
import gymnax 
from flax import nnx

from tppo.utils import load_configs 
from tppo.utils import env_utils

from tppo.algorithms.base_tppo import TransformerPPO


def main():
    env_name = "Pendulum-v1"
    env, env_params = env_utils.make_env(env_id=env_name)
    env_type, action_dim = env_utils.get_env_specs(env, env_params) 

    MODEL_NAME = "example_model"
    FILE_TYPE = ".toml"
    cfg = load_configs.load_config(MODEL_NAME+FILE_TYPE)[MODEL_NAME]

    SEED = 0 
    rng = nnx.Rngs(SEED)
    model = TransformerPPO(**cfg, env_type=env_type, actor_out=action_dim, rngs=rng)


if __name__ == "__main__": 
    main() 