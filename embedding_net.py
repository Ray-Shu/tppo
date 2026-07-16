# %%
from typing import Any

from flax.typing import RNGSequences
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# %%
class EmbeddingNet(nnx.Module):
    def __init__(self, d_hidden, rngs):
        self.rngs = rngs
        self.d_hidden = d_hidden
        self.linear = nnx.data(None)  # lazy initialization of the linear module to account for dynamic input features

    def __call__(self, hidden, obs):
        """
        hidden: Any
        obs: ((batch_size, obs_dim), (batch_size, action_dim), (batch_size, 1))
        """

        # U matrix, a concatenation of observation, last action and last reward
        obs, last_action, last_reward = obs
        obs = jnp.reshape(obs, (obs.shape[0], -1))
        U = jnp.concatenate(
            (obs, last_action, last_reward),
            axis = -1
        )

        # E matrix, a linear projection of U
        if self.linear is None:
            self.linear = nnx.data(nnx.Linear(
                in_features = U.shape[-1],
                out_features = self.d_hidden,
                rngs = self.rngs
            ))
        E = self.linear(U)
        return E

# %%
enet = EmbeddingNet(d_hidden=4, rngs=nnx.Rngs(42))

key = jax.random.PRNGKey(0)
hidden = 9
shapes = ((10, 9*9*3), (10, 4), (10, 1))
keys = jax.random.split(key, len(shapes))
obs = tuple(jax.random.normal(key, shape) for key,shape in zip(keys, shapes))
e = enet(hidden, obs)
print(e.shape)
