import numpy as np


def tile_xyz(n, z, spacing: float = 1.0):
    """Return ``(x, y, z)`` coordinates tiled in a grid.

    The previous implementation iterated over each coordinate and appended the
    result to a list.  This version performs the operation using ``numpy``
    vectorised operations which is both shorter and typically faster.

    Parameters
    ----------
    n:
        Number of coordinates to generate.
    z:
        Height value for all coordinates.
    spacing:
        Distance between adjacent grid points.
    """

    side = int(np.ceil(np.sqrt(n)))  # number of tiles per row/column
    row, col = np.divmod(np.arange(n), side)
    offset = spacing * (side - 1) / 2.0  # to centre grid at (0, 0)

    x = row * spacing - offset
    y = col * spacing - offset
    z = np.full_like(x, z, dtype=np.float32)

    return np.stack([x, y, z], axis=1)
