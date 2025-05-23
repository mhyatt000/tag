from __future__ import annotations


class Space:
    def sample(self):
        raise NotImplementedError

    def contains(self, x) -> bool:
        return True


class Box(Space):
    def __init__(self, low, high, shape=None, dtype=float):
        self.low = low
        self.high = high
        self.shape = tuple(shape) if shape is not None else None
        self.dtype = dtype

    def sample(self):
        if self.shape is None:
            return []
        return [0 for _ in range(self._numel())]

    def contains(self, x) -> bool:
        return True

    def _numel(self) -> int:
        n = 1
        if self.shape is not None:
            for dim in self.shape:
                n *= dim
        return n


class Dict(Space):
    def __init__(self, spaces: dict[str, Space]):
        self.spaces = spaces

    def sample(self):
        return {k: s.sample() for k, s in self.spaces.items()}

    def contains(self, x) -> bool:
        if not isinstance(x, dict):
            return False
        for k, s in self.spaces.items():
            if k not in x or not s.contains(x[k]):
                return False
        return True
