import distrax
import jax.numpy as jnp
from flax import nnx

class DiscreteActorHead(nnx.Module):
    def __init__(self, d_hidden, d_out, rngs):
        self.linear = nnx.Linear(in_features=d_hidden, out_features=d_out, rngs=rngs)

    def __call__(self, x):
        """
        Returns a distrax categorical distribution that can be sampled from.
        """
        logits = self.linear(x)
        actions = distrax.Categorical(logits=logits)
        return actions

class ContinuousActorHead(nnx.Module): 
    def __init__(self, d_hidden, d_out, rngs): 
        self.linear = nnx.Linear(in_features=d_hidden, out_features=d_out, rngs=rngs)
        self.log_stddev = nnx.Param(jnp.zeros(d_out))

    def __call__(self, x):
        "Returns a distrax normal distribution"
        mu = self.linear(x)
        std = jnp.exp(self.log_stddev)

        return distrax.Normal(loc=mu, scale=std)

class CriticHead(nnx.Module):
    def __init__(self, d_hidden, rngs, d_out=1):
        self.linear = nnx.Linear(in_features=d_hidden, out_features=d_out, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)