from typing import Tuple, Optional
import jax
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array

"""
    TODO:
    - change data_dim -> input_dim in all layers
"""

class MultiheadAttentionBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    ln0: eqx.nn.LayerNorm
    ln1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, data_dim, n_heads, hidden_dim, *, key):
        keys = jr.split(key)
        self.attention = eqx.nn.MultiheadAttention(
            n_heads, data_dim, key=keys[0]
        )
        self.ln0 = eqx.nn.LayerNorm((data_dim,))
        self.ln1 = eqx.nn.LayerNorm((data_dim,))
        self.mlp = eqx.nn.MLP(
            data_dim, 
            data_dim, 
            width_size=hidden_dim, 
            depth=1, 
            activation=jax.nn.gelu,
            key=keys[1]
        )
    
    def __call__(self, x, y):
        # if x is y:
        #     h = jax.vmap(self.ln0)(x + self.attention(x, x, x))
        # else:
        h = jax.vmap(self.ln0)(x + self.attention(x, y, y))
        return jax.vmap(self.ln1)(h + jax.vmap(self.mlp)(h))



class Encoder(eqx.Module):
    sabs: Tuple[MultiheadAttentionBlock]

    def __init__(self, data_dim, n_layers, n_heads, hidden_dim, *, key):
        sabs = []
        keys = jr.split(key, n_layers)
        for key in keys:
            sabs.append(
                MultiheadAttentionBlock(
                    data_dim, n_heads, hidden_dim, key=key
                )
            )
        keys = jr.split(keys[-1], 3)
        self.sabs = tuple(sabs) # Encoder layers

    def __call__(self, x):
        z = x
        for sab in self.sabs:
            z = sab(z, z)
        return z


class MultiheadAttentionPooling(eqx.Module):
    mlp: eqx.nn.MLP
    S: jax.Array
    mab: MultiheadAttentionBlock

    def __init__(self, data_dim, n_heads, n_seeds, hidden_dim, *, key):
        keys = jr.split(key, 3)
        self.mlp = eqx.nn.MLP(
            data_dim, 
            data_dim, 
            depth=1, 
            width_size=hidden_dim, 
            activation=jax.nn.gelu,
            key=keys[0]
        )
        self.S = jr.normal(keys[1], (n_seeds, data_dim))
        self.mab = MultiheadAttentionBlock(
            data_dim, n_heads, hidden_dim, key=keys[2]
        )

    def __call__(self, z):
        return self.mab(self.S, jax.vmap(self.mlp)(z))


class Decoder(eqx.Module):
    sab: MultiheadAttentionBlock
    pma: MultiheadAttentionPooling
    mlp: eqx.nn.MLP

    def __init__(self, data_dim, n_heads, n_seeds, hidden_dim, *, key):
        keys = jr.split(key, 3)
        self.sab = MultiheadAttentionBlock(
            data_dim, n_heads, hidden_dim, key=keys[0]
        )
        self.pma = MultiheadAttentionPooling(
            data_dim, 
            n_heads, 
            n_seeds, 
            hidden_dim,
            key=keys[1]
        )
        self.mlp = eqx.nn.MLP(
            data_dim, 
            data_dim, 
            depth=1, 
            width_size=hidden_dim, 
            activation=jax.nn.gelu, 
            key=keys[2]
        )
    
    def __call__(self, z):
        p = self.pma(z)
        return jax.vmap(self.mlp)(self.sab(p, p))
        

class SetTransformer(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    mlp: eqx.nn.MLP
    embedder: Optional[eqx.Module] = None

    def __init__(
        self, 
        data_dim: int, 
        out_dim: int, 
        n_layers: int, 
        n_heads: int, 
        n_seeds: int, 
        hidden_dim: int, 
        embed_dim: int, 
        *, 
        key: Key
    ):
        keys = jr.split(key, 4)
        input_dim = embed_dim if embed_dim is not None else data_dim
        self.encoder = Encoder(
            input_dim, 
            n_layers, 
            n_heads, 
            hidden_dim, 
            key=keys[0]
        )
        self.decoder = Decoder(
            input_dim, 
            n_heads, 
            n_seeds, 
            hidden_dim, 
            key=keys[1]
        )
        self.mlp = eqx.nn.MLP(
            n_seeds * input_dim, 
            out_dim, 
            width_size=hidden_dim, 
            depth=1, 
            activation=jax.nn.gelu,
            key=keys[2]
        )
        if embed_dim is not None:
            self.embedder = eqx.nn.Linear(data_dim, embed_dim, key=keys[3])
    
    def __call__(self, x: Array) -> Array:
        if self.embedder is not None:
            x = jax.vmap(self.embedder)(x)
        y = self.decoder(self.encoder(x)) # No encoder here?! it is within PMA that is within 
        return self.mlp(y.flatten()) # This right?, this array always the same shape