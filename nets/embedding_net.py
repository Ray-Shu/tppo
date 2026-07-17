# %%
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

# %%
class EmbeddingNet(nnx.Module):
    linear: nnx.Linear | None 

    def __init__(self, d_hidden, T, rngs):
        self.rngs = rngs
        self.d_hidden = d_hidden
        self.T = T

        self.is_initialized = False
        self.linear = nnx.data(None)

        # positional embedding matrix (P) 
        p_init = jax.random.normal(self.rngs.params(), (self.T, self.d_hidden))
        self.P = nnx.Param(p_init)

    def __call__(self, hidden:Any, obs):
        """
        hidden: Any
        obs: ((T, obs_dim), (T, action_dim), (T, 1)), where T is the context window. 
        """

        # U matrix, a concatenation of observation, last action and last reward
        obs, last_action, last_reward = obs
        obs = jnp.reshape(obs, (obs.shape[0], -1))
        U = jnp.concatenate(
            (obs, last_action, last_reward),
            axis = -1
        )

        if not self.is_initialized: 
            # trainable embedding matrix (E) 
            self.linear = nnx.Linear(
                in_features = U.shape[-1],
                out_features = self.d_hidden, 
                rngs = self.rngs 
            )            
            self.is_initialized = True

        E = self.linear(U)
        X = E + self.P

        return X 

# %%
# rngs = nnx.Rngs(42)
# hidden = 9
# T = 10
# enet = EmbeddingNet(d_hidden=4, T=T, rngs=rngs)


# shapes = ((T, 9*9*3), (T, 4), (T, 1))
# keys = jax.random.split(rngs.params(), len(shapes))
# obs = tuple(jax.random.normal(key, shape) for key,shape in zip(keys, shapes))
# e = enet(hidden, obs)
# print(e.shape)

# %%
