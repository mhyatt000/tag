import numpy as np


def tile_xyz(n, z, spacing=1.0):
    side = int(np.ceil(np.sqrt(n)))  # number of tiles per row/column
    coords = []

    offset = spacing * (side - 1) / 2.0  # to center grid at (0,0)

    for i in range(n):
        row = i // side
        col = i % side
        x = (row * spacing) - offset
        y = (col * spacing) - offset
        coords.append([x, y, z])

    return np.array(coords)
