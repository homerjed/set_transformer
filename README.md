# Set Transformer

Implementation of [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825) (Lee++ 2019) in `jax` and `equinox`.

Uses MNIST (loaded with `torch`) converted into point-clouds.

To do:
* ISAB blocks
* AdaNorm Layer normalisation
* Dataloader for mixed-cardinality sets
* Dropout
* LR schedule

![alt text](figs/data.png?raw=true)