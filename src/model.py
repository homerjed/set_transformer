from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array


class MultiheadAttentionBlock(eqx.Module):
    dim_V: int
    n_heads: int
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
        mlp_kwargs: Optional[dict] = None,
        key: Key
    ):
        self.dim_V = dim_V
        self.n_heads = n_heads

        keys = jr.split(key, 3) 

        if mlp_kwargs is None:
            mlp_kwargs = dict(
                activation=jax.nn.gelu, 
                width_size=hidden_dim,
                depth=0
            )

        self._q = eqx.nn.Linear(dim_Q, dim_V, key=keys[0])
        self._k = eqx.nn.Linear(dim_K, dim_V, key=keys[1])
        self._v = eqx.nn.Linear(dim_K, dim_V, key=keys[2])
        self.ln0 = eqx.nn.LayerNorm((dim_V,))
        self.ln1 = eqx.nn.LayerNorm((dim_V,))
        self.mlp = eqx.nn.MLP(
            dim_V, 
            dim_V, 
            **mlp_kwargs,
            key=keys[3]
        )
    
    def __call__(self, x: Array, y: Array) -> Array:
        q = jax.vmap(self._q)(x)
        k = jax.vmap(self._k)(y)
        v = jax.vmap(self._v)(y)

        dim_split = self.dim_V // self.n_heads

        q_ = jnp.hstack(jnp.split(q, dim_split, axis=1))
        k_ = jnp.hstack(jnp.split(k, dim_split, axis=1))
        v_ = jnp.hstack(jnp.split(v, dim_split, axis=1))

        attention = jax.nn.softmax(q_ @ k_.T / jnp.sqrt(self.dim_V), axis=1)

        o = jnp.concatenate([q_ + attention @ v_], axis=1)
        o = jax.vmap(self.ln0)(o)
        o = o + jax.nn.gelu(jax.vmap(self.mlp)(o))
        o = jax.vmap(self.ln1)(o)
        return o


class SelfAttentionBlock(eqx.Module):
    mab: MultiheadAttentionBlock

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        n_heads: int, 
        hidden_dim: int, 
        *, 
        mlp_kwargs: Optional[dict] = None,
        key: Key
    ):
        self.mab = MultiheadAttentionBlock(
            in_size, 
            in_size, 
            out_size, 
            n_heads, 
            hidden_dim=hidden_dim, 
            mlp_kwargs=mlp_kwargs,
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
        mlp_kwargs: Optional[dict] = None,
        key: Key
    ):
        keys = jr.split(key, 3)
        self.I = jr.normal(keys[0], (n_inducing_points, out_size))
        self.mab0 = MultiheadAttentionBlock(
            out_size, 
            in_size, 
            out_size, 
            n_heads, 
            hidden_dim, 
            mlp_kwargs=mlp_kwargs,
            key=keys[0]
        )
        self.mab1 = MultiheadAttentionBlock(
            in_size, 
            out_size, 
            out_size, 
            n_heads, 
            hidden_dim, 
            mlp_kwargs=mlp_kwargs,
            key=keys[1]
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
        mlp_kwargs: Optional[dict] = None,
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
            mlp_kwargs=mlp_kwargs,
            key=keys[1]
        )

    def __call__(self, z: Array) -> Array:
        return self.mab(self.S, z)