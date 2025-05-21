from dataclasses import dataclass, field
import genesis as gs
import numpy as np

def default(x):
    return field(default_factory=lambda: x)

@dataclass
class Terrain:
    '''
    makes terrain
    size = int
    pos = tuple (xpos, ypos, zpos)
    tiles = 2D arr of terrain tiles
    '''
    size: float = 5
    pos: tuple = default((0, 0))
    tiles: List[List[TerrainTile]] = default([[TerrainTile(heightfield=HeightField())]])

    def __post_init__(self):
        #things

    # def __init__(self, size, pos, tiles):
    #     # TODO: add guard statements
        
    #     # TODO: 
    #     self.size = size
    

    def set_in_scene(self, scene):
        
@dataclass
class TerrainTile:
    heightfield: HeightField = HeightField()
    vertical_scale: float = 
    x_res: int = 5
    y_res: int = 5

    def __post_init__(self):
        quit()

    def generate(self, scene):
        scene.add_entity(
            gs.morphs.Terrain(height_field=heightfield.grid, )
        )

@dataclass
class HeightField:
    size: float = 1

    def __post_init__(self):
        self.grid = np.zeros((size, size))