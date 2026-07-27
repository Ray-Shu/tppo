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
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff: int, rngs:nnx.Rngs, band:int|None = None, num_layers=1):
        """
        T: size of the context window 
        d_hidden: hidden dim 
        d_keys, d_vals: dimensions for queries, keys and values for the attention mechanism  
        d_ff: dimension of the feedforward network in transformer block
        band: the "receptive field" of the transformer. Similar to context length, but works with stacks of transformer blocks
        """
        # params
        self.T = T
        self.d_hidden = d_hidden
        self.d_keys = d_keys 
        self.d_vals = d_vals 
        self.d_ff = d_ff

        # nets
        self.embed = EmbeddingNet(self.d_hidden, rngs)
        self.actor = ActorHead(self.d_hidden, rngs)
        self.critic = CriticHead(self.d_hidden, rngs)

        self.blocks = nnx.List([
            TransformerBlock(self.T, self.d_hidden, self.d_keys, self.d_vals, self.d_ff, rngs, band=band)
            for _ in range(num_layers)
        ])


    def __call__(self, hidden:Any, obs, read_from=None):
        """
        hidden: Any
        obs: ((B, M, obs_dim), (B, M, action_dim), (B, M, 1)) where:
            - B is the batch size
            - M is the sequence length: T-1 + N, where T is the context window and N is the rollout buffer size

        read_from: which positions become decisions.
            None: last position only, (B, .)
            int: many decisions per sequence
        """
        x = self.embed(obs)

        for block in self.blocks:
            x = block(x)

        h = x[:, -1, :] if read_from is None else x[:, read_from:, :]
        pi, value = self.actor(h), self.critic(h)

        return pi, value        


# %%
if __name__ == "__main__":
    B = 2
    T = 5
    rngs = nnx.Rngs(42)
    d_hidden = 4
    d_keys = 3
    d_vals = 6
    d_ff = 10
    shapes = ((B, T, 9 * 9 * 3), (B, T, 4), (B, T, 1))
    keys = jax.random.split(rngs.params(), len(shapes))
    obs = tuple(jax.random.normal(key, shape) for key, shape in zip(keys, shapes))

    tppo = TransformerPPO(T, d_hidden, d_keys, d_vals, d_ff, rngs)

    pi, value = tppo(None, obs)

    print(pi.probs)
    print(value)

