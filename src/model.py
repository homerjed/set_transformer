from typing import Tuple, Optional
import jax
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array


class MultiheadAttentionBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    _q: eqx.nn.Linear
    _k: eqx.nn.Linear
    _v: eqx.nn.Linear
    ln0: eqx.nn.LayerNorm
    ln1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(
        self, 
        dim_Q: int, 
        dim_K: int, 
        dim_V: int, 
        n_heads: int, 
        hidden_dim: int, 
        *, 
        key: Key
    ):
        keys = jr.split(key, 6)
        self.attention = eqx.nn.MultiheadAttention(
            n_heads, dim_V, key=keys[0]
        )
        self._q = eqx.nn.Linear(dim_Q, dim_V, key=keys[1])
        self._k = eqx.nn.Linear(dim_K, dim_V, key=keys[2])
        self._v = eqx.nn.Linear(dim_K, dim_V, key=keys[3])
        self.ln0 = eqx.nn.LayerNorm((dim_V,))
        self.ln1 = eqx.nn.LayerNorm((dim_V,))
        self.mlp = eqx.nn.MLP(
            dim_V, 
            dim_V, 
            width_size=hidden_dim, 
            depth=0, 
            activation=jax.nn.gelu,
            key=keys[5]
        )
    
    def __call__(self, x: Array, y: Array) -> Array:
        q = jax.vmap(self._q)(x)
        k = jax.vmap(self._k)(y)
        v = jax.vmap(self._v)(y)
        h = jax.vmap(self.ln0)(q + self.attention(q, k, v))
        return jax.vmap(self.ln1)(h + jax.vmap(self.mlp)(h))


class SelfAttentionBlock(eqx.Module):
    mab: MultiheadAttentionBlock

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        n_heads: int, 
        hidden_dim: int, 
        *, 
        key: Key
    ):
        self.mab = MultiheadAttentionBlock(
            in_size, 
            in_size, 
            out_size, 
            n_heads, 
            hidden_dim=hidden_dim, 
            key=key
        )

    def __call__(self, x: Array) -> Array:
        return self.mab(x, x)


class InducedSelfAttentionBlock(eqx.Module):
    I: Array
    mab0: MultiheadAttentionBlock
    mab1: MultiheadAttentionBlock

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        n_heads: int, 
        n_inducing_points: int, 
        hidden_dim: int, 
        *, 
        key: Key
    ):
        keys = jr.split(key, 3)
        self.I = jr.normal(keys[0], (n_inducing_points, out_size))
        self.mab0 = MultiheadAttentionBlock(
            out_size, in_size, out_size, n_heads, hidden_dim, key=keys[0]
        )
        self.mab1 = MultiheadAttentionBlock(
            in_size, out_size, out_size, n_heads, hidden_dim, key=keys[1]
        )
    
    def __call__(self, x: Array) -> Array:
        H = self.mab0(self.I, x)
        return self.mab1(x, H)


class MultiheadAttentionPooling(eqx.Module):
    S: Array
    mab: MultiheadAttentionBlock

    def __init__(
        self, 
        in_size: int, 
        n_heads: int, 
        n_seeds: int, 
        hidden_dim: int, 
        *, 
        key: Key
    ):
        keys = jr.split(key)
        self.S = jr.normal(keys[0], (n_seeds, in_size))
        self.mab = MultiheadAttentionBlock(
            in_size, 
            in_size, 
            in_size, 
            n_heads, 
            hidden_dim, 
            key=keys[1]
        )

    def __call__(self, z: Array) -> Array:
        return self.mab(self.S, z)


# class Encoder(eqx.Module):
#     sabs: Tuple[MultiheadAttentionBlock]

#     def __init__(self, in_size, out_size, n_layers, n_heads, hidden_dim, *, key):
#         keys = jr.split(key, n_layers)
#         sabs = [
#             SelfAttentionBlock(
#                 in_size, hidden_dim, n_heads, hidden_dim, key=keys[0]
#             )
#         ]
#         for key in jr.split(key, n_layers):
#             sabs.append(
#                 SelfAttentionBlock(
#                     hidden_dim, hidden_dim, n_heads, hidden_dim, key=key
#                 )
#             )
#         sabs += [
#             SelfAttentionBlock(
#                 hidden_dim, out_size, n_heads, hidden_dim, key=keys[-1]
#             )
#         ]
#         self.sabs = tuple(sabs) # Encoder layers

#     def __call__(self, x):
#         z = x
#         for sab in self.sabs:
#             z = sab(z)
#         return z


# class Decoder(eqx.Module):
#     sabs: MultiheadAttentionBlock
#     pma: MultiheadAttentionPooling
#     mlp: eqx.nn.MLP

#     def __init__(self, in_dim, out_dim, n_layers, n_heads, n_seeds, hidden_dim, *, key):
#         keys = jr.split(key, 3)
#         self.pma = MultiheadAttentionPooling(
#             in_dim, 
#             n_heads, 
#             n_seeds, 
#             hidden_dim,
#             key=keys[0]
#         )
#         sabs = []
#         dims = [in_dim] + [hidden_dim] * n_layers + [hidden_dim]
#         for (_key, _in, _out) in zip(
#             jr.split(keys[1], n_layers + 2), 
#             dims[:-1], 
#             dims[1:]
#         ):
#             sabs.append(
#                 SelfAttentionBlock(
#                     _in, _out, n_heads, hidden_dim, key=_key
#                 )
#             )
#         self.sabs = tuple(sabs)
#         self.mlp = eqx.nn.MLP(
#             hidden_dim, 
#             out_dim, 
#             depth=0, 
#             width_size=hidden_dim, 
#             activation=jax.nn.gelu, 
#             key=keys[2]
#         )
    
#     def __call__(self, z):
#         p = self.pma(z)
#         x = p
#         for sab in self.sabs:
#             x = sab(x)
#         return jax.vmap(self.mlp)(x)
        

# class SetTransformer(eqx.Module):
#     encoder: Encoder
#     decoder: Decoder
#     mlp: eqx.nn.MLP
#     embedder: Optional[eqx.Module] = None

#     def __init__(
#         self, 
#         data_dim: int, 
#         out_dim: int, 
#         n_layers: int, 
#         n_heads: int, 
#         n_seeds: int, 
#         hidden_dim: int, 
#         embed_dim: int, 
#         *, 
#         key: Key
#     ):
#         keys = jr.split(key, 4)
#         input_dim = embed_dim if embed_dim is not None else data_dim
#         self.encoder = Encoder(
#             input_dim, 
#             hidden_dim,
#             n_layers, 
#             n_heads, 
#             hidden_dim, 
#             key=keys[0]
#         )
#         self.decoder = Decoder(
#             hidden_dim, 
#             out_dim,
#             n_heads, 
#             n_seeds, 
#             hidden_dim, 
#             key=keys[1]
#         )
#         self.mlp = eqx.nn.MLP(
#             n_seeds * input_dim, 
#             out_dim, 
#             width_size=hidden_dim, 
#             depth=1, 
#             activation=jax.nn.gelu,
#             key=keys[2]
#         )
#         if embed_dim is not None:
#             self.embedder = eqx.nn.Linear(data_dim, embed_dim, key=keys[3])
    
#     def __call__(self, x: Array) -> Array:
#         if self.embedder is not None:
#             x = jax.vmap(self.embedder)(x)
#         y = self.decoder(self.encoder(x)) 
#         return self.mlp(y.flatten()) 