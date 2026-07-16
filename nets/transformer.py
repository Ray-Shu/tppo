# %%%
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# %%
"""
A single transformer block.

Takes the embedding token X = E + P, where X has shape [T x d_hidden].
T: size of the context window
E: embedding matrix
P: position matrix
"""
class TransformerBlock(nnx.Module):
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff:int, rngs:nnx.Rngs):
        self.context_window = T
        self.d_hidden = d_hidden
        self.d_queries = d_keys  # d_queries must be the same as d_keys
        self.d_keys = d_keys
        self.d_values = d_vals
        self.d_ff = d_ff
        self.rngs = rngs

        self.layernorm1 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)

        self.queries_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_queries, rngs=self.rngs)
        self.keys_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_keys, rngs=self.rngs)
        self.values_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_values, rngs=self.rngs)
        self.output_linear = nnx.Linear(in_features=self.d_values, out_features=self.d_hidden, rngs=self.rngs)

        self.ffn = nnx.Sequential(
            nnx.Linear(in_features=self.d_hidden, out_features=self.d_ff, rngs=self.rngs),
            nnx.relu,
            nnx.Linear(in_features=self.d_ff, out_features=self.d_hidden, rngs=self.rngs)
        )

    def __call__(self, x):
        x_ln = self.layernorm1(x)

        # attn block
        Q = self.queries_linear(x_ln)
        K = self.keys_linear(x_ln)
        V = self.values_linear(x_ln)

        mask = jnp.triu(jnp.ones([self.context_window, self.context_window]), k=1)
        M = jnp.where(mask==1, -jnp.inf, 0)

        scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(self.d_keys) + M
        out = nnx.softmax(scores, axis=-1)
        attn_out = jnp.matmul(out, V)
        x2 = self.output_linear(attn_out) + x

        x2_ln = self.layernorm2(x2)
        return self.ffn(x2_ln) + x2


# %%
rngs = nnx.Rngs(0)
T = 10
d_h = 5
d_k = 3 # same as d_q
d_v = 4
d_ff = 7
transformer = TransformerBlock(T, d_h, d_k, d_v, d_ff, rngs)
x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(10, 5))

y = transformer(x)
print(y.shape)
print(y)
