from typing import Any
import numpy

import jax
import jax.numpy as jnp
import distrax
from flax import nnx

from tppo.algorithms.nets.actor_critics import DiscreteActorHead, ContinuousActorHead, CriticHead
from tppo.algorithms.nets.embedding_net import EmbeddingNet
from tppo.algorithms.nets.transformer import TransformerBlock

class TransformerPPO(nnx.Module):
    def __init__(
            self, 
            T:int, 
            d_hidden:int, 
            d_keys:int, 
            d_vals:int, 
            d_ff:int, 
            band:int|None, 
            num_blocks:int,
            actor_out:int, 
            env_type:str,
            rngs:nnx.Rngs,
            concat_embedding:bool=True):
        """
        T: size of the context window 
        d_hidden: hidden dim 
        d_keys, d_vals: dimensions for queries, keys and values for the attention mechanism  
        d_ff: dimension of the feedforward network in transformer block
        band: the "receptive field" of the transformer. Similar to context length, but works with stacks of transformer blocks
        num_blocks: the total number of transformer blocks
        actor_out: the output dimension of the actor head 
        critic_out: the output dimension of the critic head
        env_type: whether or not the environment wants 'discrete' or 'continuous' outputs 
        concat_embedding: whether or not to concatenate last action and last reward information to an observation; default is True
        """
        # transformer params
        self.T = T
        self.d_hidden = d_hidden
        self.d_keys = d_keys 
        self.d_vals = d_vals 
        self.d_ff = d_ff
        self.num_blocks = num_blocks
        if band is None: 
            band = self.T / self.num_blocks

        # nets
        self.embed = EmbeddingNet(self.d_hidden, rngs, concat_embedding)
        self.critic = CriticHead(self.d_hidden, rngs)

        # actor 
        if env_type.lower() == "discrete": 
            self.actor = DiscreteActorHead(self.d_hidden, actor_out, rngs)
        elif env_type.lower() == "continuous":
            self.actor = ContinuousActorHead(self.d_hidden, actor_out, rngs)
        else: 
            raise ValueError(f"Environment type {env_type} doesn't work. An environment type is either 'discrete' or 'continuous'.")

        # transformer blocks
        self.blocks = nnx.List([
            TransformerBlock(self.T, self.d_hidden, self.d_keys, self.d_vals, self.d_ff, rngs, band=band)
            for _ in range(num_blocks)
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

        Returns: 
            pi (distrax), value 
        """
        x = self.embed(obs)

        for block in self.blocks:
            x = block(x)

        h = x[:, -1, :] if read_from is None else x[:, read_from:, :]
        pi, value = self.actor(h), self.critic(h)

        return pi, value        

