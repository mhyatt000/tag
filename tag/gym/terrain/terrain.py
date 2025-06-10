from dataclasses import dataclass, field
from typing import Any, Tuple

import genesis as gs
import numpy as np

import genesis.utils.misc as mu
from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils

# TODO(mbt): Implement Class
# NOTE(dle): Placeholder Class until terrain is implemented
class Terrain:
    def __init__(self, scene: gs.Scene):
        self.terrain = scene.add_entity(gs.morphs.Plane())


# NOTE(dle): I believe this class should have the Obstacle object, we should discuss

@dataclass
class TerrainFactoryConfig:
    # terrain generation vars
    n:int = 1
    size:float = 5.0
    z_offset:float = 0.0
    horizontal_scale:float = 0.05
    vertical_scale:float = 0.01
    subterrain_types:list = field(default_factory=lambda: ['flat_terrain'])

    # wall vars
    wall_count:int = -1
    wall_count_max:int = -1
    wall_length:int = 40
    wall_length_max:int = -1
    wall_width:int = 3
    wall_width_max:int = -1
    wall_height:int = 80
    wall_height_max:int = -1
    walls_horizontal_only:bool = False
    walls_vertical_only:bool = False

    # perimeter wall vars
    perimeter_walls:bool = False
    perimeter_thickness:int = 1
    perimeter_height:int = 100
    perimeter_height_max:int = -1

class TerrainFactory:
    def __init__(self, n:int, size:float, z_offset:float, horizontal_scale:float, vertical_scale:float, subterrain_types:list):
        self.n = n
        self.size = size
        self.z_offset = z_offset
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.subterrain_types = subterrain_types

        pos = -1/2 * size * n
        self.pos = (pos, pos, z_offset)
        self.generate_terrain()

    def generate_terrain(self):
        st_types_2d_arr = [[None for _ in range(self.n)] for _ in range(self.n)]

        for i in range(self.n):
            for j in range(self.n):
                random_terrain = self.subterrain_types[np.random.randint(len(self.subterrain_types))] # grab a random subterrain from the list of subterrains

                st_types_2d_arr[i][j] = random_terrain

        self.result = WalledTerrain(
            n_subterrains = (self.n, self.n),
            subterrain_size = (self.size, self.size),
            horizontal_scale = self.horizontal_scale,
            vertical_scale = self.vertical_scale,
            pos = self.pos,
            subterrain_types = st_types_2d_arr
        )
        return self.terrain()

    def add_walls(
        self,
        n:int=3,
        max_n:int=-1,
        length:int=40,
        max_length:int=-1,
        width:int=3,
        max_width:int=-1,
        height:int=80,
        max_height:int=-1,
        horizontal_only:bool=False,
        vertical_only:bool=False
    ): 
        if (not (horizontal_only and vertical_only)) and (not length < 0) and (not width < 0) and (not height < 0) and (not n < 0):
            self.result.randomize_walls(
                n=n,
                max_n=max_n,
                length=length,
                max_length=max_length,
                width=width,
                max_width=max_width,
                height=height,
                max_height=max_height,
                horizontal_only=horizontal_only,
                vertical_only=vertical_only
            )
    
    def add_perimeter_walls(
        self,
        thickness:int=1,
        height:int=100,
        max_height:int=-1
    ):
        if not (thickness < 1 or height < 0):
            if height < max_height:
                height = np.random.randint(height, max_height)
            
            self.result.add_perimeter_walls(thickness, height)
    
    def terrain(self):
        return self.result.terrain()

class WalledTerrain:
    def __init__(self,
        randomize: bool = False,  # whether to randomize the terrain
        n_subterrains: Tuple[int, int] = (3, 3),  # number of subterrains in x and y directions
        subterrain_size: Tuple[float, float] = (12.0, 12.0),  # meter
        horizontal_scale: float = 0.25,  # meter size of each cell in the subterrain
        vertical_scale: float = 0.005,  # meter height of each step in the subterrain
        pos: Tuple[float, float, float] = (0, 0, 0),
        subterrain_types: Any = "flat_terrain"
    ):
        self.randomize = randomize
        self.n_subterrains = n_subterrains
        self.subterrain_size = subterrain_size
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.pos = pos
        self.subterrain_types = subterrain_types
        self._post_init()

    def _post_init(self):
        ################## Validate args ##################
        supported_subterrain_types = [
            "flat_terrain",
            "fractal_terrain",
            "random_uniform_terrain",
            "sloped_terrain",
            "pyramid_sloped_terrain",
            "discrete_obstacles_terrain",
            "wave_terrain",
            "stairs_terrain",
            "pyramid_stairs_terrain",
            "stepping_stones_terrain",
        ]

        if isinstance(self.subterrain_types, str):
            subterrain_types = []
            for i in range(self.n_subterrains[0]):
                row = []
                for j in range(self.n_subterrains[1]):
                    row.append(self.subterrain_types)
                subterrain_types.append(row)
            self.subterrain_types = subterrain_types
        else:
            if np.array(self.subterrain_types).shape != (self.n_subterrains[0], self.n_subterrains[1]):
                gs.raise_exception(
                    "`subterrain_types` should be either a string or a 2D list of strings with the same shape as `n_subterrains`."
                )

        for row in self.subterrain_types:
            for subterrain_type in row:
                if subterrain_type not in supported_subterrain_types:
                    gs.raise_exception(
                        f"Unsupported subterrain type: {subterrain_type}, should be one of {supported_subterrain_types}"
                    )

        if not mu.is_approx_multiple(self.subterrain_size[0], self.horizontal_scale) or not mu.is_approx_multiple(
            self.subterrain_size[1], self.horizontal_scale
        ):
            gs.raise_exception("`subterrain_size` should be divisible by `horizontal_scale`.")


        ################## Initialization ##################
        self._init_walls()
        self._make_heightfield()
        self._add_walls()
        self._generate_terrain()

    def _init_walls(self):
        subterrain_rows = int(self.subterrain_size[0] / self.horizontal_scale)
        subterrain_cols = int(self.subterrain_size[1] / self.horizontal_scale)

        self.walls = np.zeros(
            np.array(self.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
        )

    def _make_heightfield(self):
        subterrain_rows = int(self.subterrain_size[0] / self.horizontal_scale)
        subterrain_cols = int(self.subterrain_size[1] / self.horizontal_scale)

        heightfield = np.zeros(
            np.array(self.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
        )

        for i in range(self.n_subterrains[0]):
            for j in range(self.n_subterrains[1]):
                subterrain_type = self.subterrain_types[i][j]

                new_subterrain = isaacgym_terrain_utils.SubTerrain(
                    width=subterrain_rows,
                    length=subterrain_cols,
                    vertical_scale= self.vertical_scale,
                    horizontal_scale= self.horizontal_scale,
                )
                if not self.randomize:
                    saved_state = np.random.get_state()
                    np.random.seed(0)

                if subterrain_type == "flat_terrain":
                    subterrain_height_field = np.zeros((subterrain_rows, subterrain_cols), dtype=np.int16)

                elif subterrain_type == "fractal_terrain":
                    subterrain_height_field = fractal_terrain(new_subterrain, levels=8, scale=5.0).height_field_raw

                elif subterrain_type == "random_uniform_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.random_uniform_terrain(
                        new_subterrain,
                        min_height=-0.1,
                        max_height=0.1,
                        step=0.1,
                        downsampled_scale=0.5,
                    ).height_field_raw

                elif subterrain_type == "sloped_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.sloped_terrain(
                        new_subterrain,
                        slope=-0.5,
                    ).height_field_raw

                elif subterrain_type == "pyramid_sloped_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_sloped_terrain(
                        new_subterrain,
                        slope=-0.1,
                    ).height_field_raw

                elif subterrain_type == "discrete_obstacles_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.discrete_obstacles_terrain(
                        new_subterrain,
                        max_height=0.05,
                        min_size=1.0,
                        max_size=5.0,
                        num_rects=20,
                    ).height_field_raw

                elif subterrain_type == "wave_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.wave_terrain(
                        new_subterrain,
                        num_waves=2.0,
                        amplitude=0.1,
                    ).height_field_raw

                elif subterrain_type == "stairs_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.stairs_terrain(
                        new_subterrain,
                        step_width=0.75,
                        step_height=-0.1,
                    ).height_field_raw

                elif subterrain_type == "pyramid_stairs_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_stairs_terrain(
                        new_subterrain,
                        step_width=0.75,
                        step_height=-0.1,
                    ).height_field_raw

                elif subterrain_type == "stepping_stones_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.stepping_stones_terrain(
                        new_subterrain,
                        stone_size=1.0,
                        stone_distance=0.25,
                        max_height=0.2,
                        platform_size=0.0,
                    ).height_field_raw

                else:
                    gs.raise_exception(f"Unsupported subterrain type: {subterrain_type}")

                if not self.randomize:
                    np.random.set_state(saved_state)

                heightfield[
                    i * subterrain_rows : (i + 1) * subterrain_rows, j * subterrain_cols : (j + 1) * subterrain_cols
                ] = subterrain_height_field

        self.base_heightfield = heightfield
    
    def _add_walls(self):
        subterrain_rows = int(self.subterrain_size[0] / self.horizontal_scale)
        subterrain_cols = int(self.subterrain_size[1] / self.horizontal_scale)

        heightfield = np.zeros(
            np.array(self.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
        )

        for x in range(len(heightfield)):
            for y in range(len(heightfield[x])):
                heightfield[x,y] = self.base_heightfield[x,y] + self.walls[x,y]

        self.heightfield = heightfield

    def _generate_terrain(self):
        self.result = gs.morphs.Terrain(
            horizontal_scale= self.horizontal_scale,
            vertical_scale= self.vertical_scale,
            pos= self.pos,
            height_field= self.heightfield
        )

    def randomize_walls(self,
        n,
        max_n,
        length,
        max_length,
        width,
        max_width,
        height,
        max_height,
        horizontal_only,
        vertical_only,
    ):
        ################## Validate inputs ##################
        if n < max_n:
            n = np.random.randint(n, max_n)
        
        if vertical_only: # flip horizontal and vertical
            temp = width
            width = length
            length = temp

            temp = max_width
            max_width = max_length
            max_length = temp
            randdirection = False
        elif horizontal_only:
            randdirection = False
        else:
            randdirection = True

        ################## Generate walls ##################
        for num in range(n):
            if width < max_width:
                curr_width = np.random.randint(width, max_width)
            else:
                curr_width = width
            
            if length < max_length:
                curr_length = np.random.randint(length, max_length)
            else:
                curr_length = length
            
            if height < max_height:
                curr_height = np.random.randint(height, max_height)
            else:
                curr_height = height

            subterrain_rows = len(self.walls)
            subterrain_cols = len(self.walls[0])
            
            if randdirection and np.random.randint(2)==0:
                x_pos = np.random.randint(subterrain_rows - curr_length)
                y_pos = np.random.randint(subterrain_cols - curr_width)
                self.place_wall(x_pos, y_pos, x_pos+curr_length, y_pos+curr_width, curr_height)
            else:
                x_pos = np.random.randint(subterrain_rows - curr_width)
                y_pos = np.random.randint(subterrain_cols - curr_length)
                self.place_wall(x_pos, y_pos, x_pos+curr_width, y_pos+curr_length, curr_height)

        self._add_walls()
        self._generate_terrain()
    
    def add_perimeter_walls(self, thickness:int, height:int):
        subterrain_rows = len(self.walls)
        subterrain_cols = len(self.walls[0])

        ################## Define key points ##################
        p1_topleft = (0,0)
        p1_botright = (thickness, subterrain_cols-thickness)
        
        p2_topleft = (0, subterrain_cols-thickness)
        p2_botright = (subterrain_rows-thickness, subterrain_cols)
        
        p3_topleft = (subterrain_rows-thickness, thickness)
        p3_botright = (subterrain_rows, subterrain_cols)

        p4_topleft = (thickness, 0)
        p4_botright = (subterrain_rows, thickness)

        ################## Find max height under wall area ##################
        height += max(
            self._find_highest_point(p1_topleft[0], p1_topleft[1], p1_botright[0], p1_botright[1]),
            self._find_highest_point(p2_topleft[0], p2_topleft[1], p2_botright[0], p2_botright[1]),
            self._find_highest_point(p3_topleft[0], p3_topleft[1], p3_botright[0], p3_botright[1]),
            self._find_highest_point(p4_topleft[0], p4_topleft[1], p4_botright[0], p4_botright[1])
        )

        ################## Generate walls ##################
        self._place_wall_at_height(p1_topleft[0], p1_topleft[1], p1_botright[0], p1_botright[1], height),
        self._place_wall_at_height(p2_topleft[0], p2_topleft[1], p2_botright[0], p2_botright[1], height),
        self._place_wall_at_height(p3_topleft[0], p3_topleft[1], p3_botright[0], p3_botright[1], height),
        self._place_wall_at_height(p4_topleft[0], p4_topleft[1], p4_botright[0], p4_botright[1], height)

        self._add_walls()
        self._generate_terrain()

    def place_wall(self, top_left_x_pos:int, top_left_y_pos:int, bot_right_x_pos:int, bot_right_y_pos:int, height:int):
        height += self._find_highest_point(top_left_x_pos, top_left_y_pos, bot_right_x_pos, bot_right_y_pos)
        self._place_wall_at_height(
            top_left_x_pos,
            top_left_y_pos,
            bot_right_x_pos,
            bot_right_y_pos,
            height
        )

    def _place_wall_at_height(self, top_left_x_pos:int, top_left_y_pos:int, bot_right_x_pos:int, bot_right_y_pos:int, height:int):
        for x in range(top_left_x_pos, bot_right_x_pos):
            for y in range(top_left_y_pos, bot_right_y_pos):
                self.walls[x,y] = max(self.walls[x,y], height) - self.base_heightfield[x,y]

    def _find_highest_point(self, top_left_x_pos:int, top_left_y_pos:int, bot_right_x_pos:int, bot_right_y_pos:int):
        result = self.base_heightfield[0,0]
        for x in range(top_left_x_pos, bot_right_x_pos):
            for y in range(top_left_y_pos, bot_right_y_pos):
                result = max(result, self.base_heightfield[x,y])
        return result

    def terrain(self):
        return self.result