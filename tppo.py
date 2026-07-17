# %%
from typing import Any
import numpy

import jax
import jax.numpy as jnp
import distrax
from flax import nnx

from nets.actor_critics import ActorHead, CriticHead
from nets.embedding_net import EmbeddingNet
from nets.transformer import TransformerBlock

# %%
class tppo(nnx.Module):
    def __init__(self, hidden:Any, d_hidden, rngs):
        # params
        self.d_hidden = d_hidden
        self.hidden = hidden

        # nets
        self.embed = EmbeddingNet(self.d_hidden, rngs)
        self.actor = ActorHead(self.d_hidden, rngs)
        self.critic = CriticHead(self.d_hidden, rngs)


    def __call__(self, obs):
        x = self.embed(self.hidden, obs)
        pass


# %%
T = 5
key = jax.random.PRNGKey(0)
hidden = 9
shapes = ((10, 9*9*3), (10, 4), (10, 1))
keys = jax.random.split(key, len(shapes))
obs = tuple(jax.random.normal(key, shape) for key,shape in zip(keys, shapes))
