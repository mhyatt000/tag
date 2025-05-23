from gymnasium.spaces import Box, Dict
import jax.numpy as jnp
import numpy as np

from tag.utils import space2spec, spec2batchspec


def test_space2spec_box():
    space = Box(low=0, high=1, shape=(3, 2), dtype=np.float32)
    spec = space2spec(space)
    assert isinstance(spec, np.ndarray)
    assert spec.shape == (3, 2)
    assert spec.dtype == np.float32


def test_space2spec_dict_and_batch():
    space = Dict(
        {
            "a": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "b": Box(low=0, high=1, shape=(1,), dtype=np.int32),
        }
    )
    spec = space2spec(space)
    assert set(spec.keys()) == {"a", "b"}
    batch = spec2batchspec(spec, 4)
    assert batch["a"].shape == (4, 2)
    assert batch["b"].shape == (4, 1)
    assert isinstance(batch["a"], jnp.ndarray)
