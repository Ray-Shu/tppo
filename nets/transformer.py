# %%
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
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff:int, rngs:nnx.Rngs, band:int|None=None):
        self.context_window = T
        self.band = T if band is None else band  # band is how much the model sees (ie. receptive field)
        self.d_hidden = d_hidden
        self.d_queries = d_keys  # d_queries must be the same as d_keys
        self.d_keys = d_keys
        self.d_values = d_vals
        self.d_ff = d_ff
        self.rngs = rngs
        self.relative_bias = nnx.Param(jnp.zeros((self.band))) # positional encodings relative to each transformer block

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
        """
        
        """
        M = x.shape[1]
        x_ln = self.layernorm1(x)

        # attn block
        Q = self.queries_linear(x_ln)
        K = self.keys_linear(x_ln)
        V = self.values_linear(x_ln)

        # create a context-window aware mask
        i = jnp.arange(M)[:, None]
        j = jnp.arange(M)[None, :]
        causal = (j <= i) & (j > i - self.band)

        dist = jnp.clip(i - j, 0, self.band - 1)   # 0 = self, band-1 = oldest in window

        scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(self.d_keys)
        scores = scores + self.relative_bias[dist]
        scores = jnp.where(causal, scores, -jnp.inf)

        attn_out = jnp.matmul(nnx.softmax(scores, axis=-1), V)
        x2 = self.output_linear(attn_out) + x

        x2_ln = self.layernorm2(x2)
        return self.ffn(x2_ln) + x2



# %%
if __name__ == "__main__": 
    rngs = nnx.Rngs(0)
    T = 3
    N = 6
    d_h = 5
    d_k = 3 # same as d_q
    d_v = 4
    d_ff = 7
    transformer = TransformerBlock(T, d_h, d_k, d_v, d_ff, rngs)
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(1, T-1+N, d_h))

    y = transformer(x)
    print(y)
    print(y.shape)

# %%
