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
class TransformerPPO(nnx.Module):
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff: int, rngs:nnx.Rngs):
        """
        T: size of the context window 
        d_hidden: hidden dim 
        d_keys, d_vals: dimensions for queries, keys and values for the attention mechanism  
        d_ff: dimension of the feedforward network in transformer block
        """
        # params
        self.T = T
        self.d_hidden = d_hidden
        self.d_keys = d_keys 
        self.d_vals = d_vals 
        self.d_ff = d_ff

        # nets
        self.embed = EmbeddingNet(self.d_hidden, self.T, rngs)
        self.transformer = TransformerBlock(self.T, self.d_hidden, self.d_keys, self.d_vals, self.d_ff, rngs)
        self.actor = ActorHead(self.d_hidden, rngs)
        self.critic = CriticHead(self.d_hidden, rngs)


    def __call__(self, hidden:Any, obs):
        """
        hidden: Any
        obs: ((T, obs_dim), (T, action_dim), (T, 1)), where T is the context window. 
        """
        x = self.embed(hidden, obs)
        out = self.transformer(x)
        
        print(x.shape)
        print(out.shape)


# %%
T = 5
rngs = nnx.Rngs(42) 
key = jax.random.PRNGKey(0)
hidden = 9
d_hidden = 4
d_keys = 3
d_vals = 6 
d_ff = 10
shapes = ((T, 9*9*3), (T, d_hidden), (T, 1))
keys = jax.random.split(rngs.params(), len(shapes))
obs = tuple(jax.random.normal(key, shape) for key,shape in zip(keys, shapes))

tppo =TransformerPPO(T, d_hidden, d_keys, d_vals, d_ff, rngs)

tppo(obs)

