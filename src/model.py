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
    lq: eqx.nn.Linear
    lk: eqx.nn.Linear
    lv: eqx.nn.Linear
    ln0: eqx.nn.LayerNorm
    ln1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, dim_Q, dim_K, dim_V, n_heads, hidden_dim, *, key):
        keys = jr.split(key, 6)
        self.attention = eqx.nn.MultiheadAttention(
            n_heads, dim_V, key=keys[0]
        )
        self.lq = eqx.nn.Linear(dim_Q, dim_V, key=keys[1])
        self.lk = eqx.nn.Linear(dim_K, dim_V, key=keys[2])
        self.lv = eqx.nn.Linear(dim_K, dim_V, key=keys[3])
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
    
    def __call__(self, x, y):
        q = jax.vmap(self.lq)(x)
        k = jax.vmap(self.lk)(y)
        v = jax.vmap(self.lv)(y)
        h = jax.vmap(self.ln0)(q + self.attention(q, k, v)) # q was x?
        return jax.vmap(self.ln1)(h + jax.vmap(self.mlp)(h))


class SelfAttentionBlock(eqx.Module):
    mab: MultiheadAttentionBlock

    def __init__(self, in_size, out_size, n_heads, hidden_dim, *, key):
        self.mab = MultiheadAttentionBlock(
            in_size, in_size, out_size, n_heads, hidden_dim=hidden_dim, key=key
        )

    def __call__(self, x):
        return self.mab(x, x)


class InducedSelfAttentionBlock(eqx.Module):
    I: jax.Array
    mab0: MultiheadAttentionBlock
    mab1: MultiheadAttentionBlock

    def __init__(self, in_size, out_size, n_heads, n_inducing_points, *, key):
        keys = jr.split(key, 3)
        self.I = jr.normal(keys[0], (n_inducing_points, out_size))
        self.mab0 = MultiheadAttentionBlock(
            out_size, in_size, out_size, n_heads, key=keys[0]
        )
        self.mab1 = MultiheadAttentionBlock(
            in_size, out_size, out_size, n_heads, key=keys[1]
        )
    
    def __call__(self, x):
        H = self.mab0(self.I, x)
        return self.mab1(x, H)


class Encoder(eqx.Module):
    sabs: Tuple[MultiheadAttentionBlock]

    def __init__(self, in_size, out_size, n_layers, n_heads, hidden_dim, *, key):
        keys = jr.split(key, n_layers)
        sabs = [
            SelfAttentionBlock(
                in_size, hidden_dim, n_heads, hidden_dim, key=keys[0]
            )
        ]
        for key in jr.split(key, n_layers):
            sabs.append(
                SelfAttentionBlock(
                    hidden_dim, hidden_dim, n_heads, hidden_dim, key=key
                )
            )
        sabs += [
            SelfAttentionBlock(
                hidden_dim, out_size, n_heads, hidden_dim, key=keys[-1]
            )
        ]
        self.sabs = tuple(sabs) # Encoder layers

    def __call__(self, x):
        z = x
        for sab in self.sabs:
            z = sab(z)
        return z


class MultiheadAttentionPooling(eqx.Module):
    # mlp: eqx.nn.MLP
    S: jax.Array
    mab: MultiheadAttentionBlock

    def __init__(self, in_size, n_heads, n_seeds, hidden_dim, *, key):
        keys = jr.split(key, 3)
        # self.mlp = eqx.nn.MLP(
        #     data_dim, 
        #     data_dim, 
        #     depth=1, 
        #     width_size=hidden_dim, 
        #     activation=jax.nn.gelu,
        #     key=keys[0]
        # )
        self.S = jr.normal(keys[1], (n_seeds, in_size))
        self.mab = MultiheadAttentionBlock(
            in_size, in_size, in_size, n_heads, hidden_dim, key=keys[2]
        )

    def __call__(self, z):
        # print("S z", self.S.shape, z.shape)
        return self.mab(self.S, z)# jax.vmap(self.mlp)(z))


class Decoder(eqx.Module):
    sabs: MultiheadAttentionBlock
    pma: MultiheadAttentionPooling
    mlp: eqx.nn.MLP

    def __init__(self, in_dim, out_dim, n_layers, n_heads, n_seeds, hidden_dim, *, key):
        keys = jr.split(key, 3)
        # self.sab = SelfAttentionBlock(
        #     in_dim, hidden_dim, n_heads, hidden_dim, key=keys[0]
        # )
        self.pma = MultiheadAttentionPooling(
            in_dim, 
            n_heads, 
            n_seeds, 
            hidden_dim,
            key=keys[1]
        )
        sabs = []
        dims = [in_dim] + [hidden_dim] * n_layers + [hidden_dim]
        for (key, _in, _out) in zip(
            jr.split(key, 1), dims[:-1], dims[1:]
        ):
            sabs.append(
                SelfAttentionBlock(
                    _in, _out, n_heads, hidden_dim, key=key
                )
            )
        self.sabs = tuple(sabs)
        self.mlp = eqx.nn.MLP(
            hidden_dim, #in_dim,
            out_dim, 
            depth=1, 
            width_size=hidden_dim, 
            activation=jax.nn.gelu, 
            key=keys[2]
        )
    
    def __call__(self, z):
        # print("z", z.shape)
        p = self.pma(z)
        # print("p", p.shape)
        # a = self.sab(p)
        x = p
        for sab in self.sabs:
            x = sab(x)
        # print("a", a.shape)
        return jax.vmap(self.mlp)(x)
        

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
            hidden_dim,
            n_layers, 
            n_heads, 
            hidden_dim, 
            key=keys[0]
        )
        self.decoder = Decoder(
            hidden_dim, #input_dim, 
            out_dim,
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


if __name__ == "__main__":
    import jax.numpy as jnp
    key = jr.key(0)
    x = jnp.ones((8, 9, 1)) # set

    # m = SelfAttentionBlock(1, 8, n_heads=1, hidden_dim=8, key=key)
    # print(jax.vmap(m)(x).shape)

    class SmallSetTransformer(eqx.Module):
        blocks: eqx.nn.Sequential
        pooling: MultiheadAttentionPooling
        out: eqx.nn.Linear

        def __init__(self, *, key):
            keys = jr.split(key, 4)
            self.blocks = [
                SelfAttentionBlock(
                    in_size=1, out_size=64, n_heads=4, hidden_dim=64, key=keys[0]
                ),
                SelfAttentionBlock(
                    in_size=64, out_size=64, n_heads=4, hidden_dim=64, key=keys[1]
                )
            ]
            # Decoder 
            self.pooling = MultiheadAttentionPooling(
                data_dim=64, n_heads=4, n_seeds=3, hidden_dim=64, key=keys[2]
            )
            self.out = eqx.nn.Linear(in_features=64, out_features=1, key=keys[3])

        def __call__(self, x):
            print("e")
            for b in self.blocks:
                x = b(x)
            print("d")
            x = self.pooling(x)
            x = jax.vmap(self.out)(x)
            return x.squeeze(-1)

    model = SmallSetTransformer(key=key)
    print(jax.vmap(model)(x).shape)