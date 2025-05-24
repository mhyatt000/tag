from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Space
import jax
import numpy as np
import torch


def spec(tree: dict[str, np.ndarray]):
    return jax.tree.map(lambda x: x.shape, tree)


def batch_space(space: Space, n: int) -> Space:
    """Wrap a Gymnasium space with an extra batch dimension of size n."""
    if isinstance(space, Box):
        low = np.broadcast_to(space.low, (n,) + space.shape).copy()
        high = np.broadcast_to(space.high, (n,) + space.shape).copy()
        return Box(low=low, high=high, dtype=space.dtype)

    elif isinstance(space, Dict):
        return Dict({key: batch_space(subsp, n) for key, subsp in space.spaces.items()})

    else:
        raise TypeError(f"batch_space() only supports Box and Dict, got {type(space)}")


def space2spec(space: spaces.Space):
    """Recursively convert a ``gymnasium.Space`` tree to a nested spec of arrays.

    Each leaf space is replaced with a NumPy array of zeros with the same shape
    and dtype as the space. ``gymnasium.spaces.Dict`` instances are returned as
    normal Python dictionaries.
    """
    if isinstance(space, spaces.Dict):
        return {k: space2spec(s) for k, s in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(space2spec(s) for s in space.spaces)
    if isinstance(space, spaces.Discrete):
        return np.zeros((1,), dtype=np.int64)
    if isinstance(space, spaces.MultiBinary):
        return np.zeros(space.n, dtype=np.int8)
    if isinstance(space, spaces.MultiDiscrete):
        return np.zeros(space.nvec.shape, dtype=np.int64)
    if isinstance(space, spaces.Box):
        dtype = space.dtype if space.dtype is not None else np.float32
        return np.zeros(space.shape, dtype=dtype)
    raise TypeError(f"Unsupported space type: {type(space)}")


def spec2batch_spec(spec, n_envs: int):
    """Stack ``spec`` along a new leading dimension of size ``n_envs``.

    ``spec`` should be a nested tree of arrays as produced by :func:`space2spec`.
    The returned structure mirrors ``spec`` with each array stacked along axis 0
    ``n_envs`` times using ``jax.numpy``.
    """

    def _stack(x):
        return torch.stack([x] * n_envs)

    return jax.tree.map(_stack, spec)
