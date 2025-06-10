from dataclasses import dataclass
from enum import Enum

from genesis import gs
import numpy as np
import torch

import tag.gym.terrain.terrain as terrain

class TerrainType(str, Enum):
    NONE = "none"
    PLANE = "plane"
    HEIGHTFIELD = "heightfield"


@dataclass
class Terrain:
    type: TerrainType = TerrainType.PLANE  # none, plane, heightfield
    friction: float = 1.0
    restitution: float = 0.0

    plane_length = 200.0  # [m]. plane size is 200x200x10 by default
    horizontal_scale = 0.1  # [m]
    vertical_scale = 0.005  # [m]
    border_size = 5  # [m]

    curriculum = False

    # rough terrain only:
    measure_heights = False
    # 1mx1.6m rectangle (without center line)
    measured_points_x = [float(i) / 10 for i in range(-8, 9)]
    measured_points_y = [float(i) / 10 for i in range(-5, 6)]

    selected = False  # select a unique terrain type and pass all arguments
    terrain_kwargs = None  # Dict of arguments for selected terrain

    max_init_terrain_level = 1  # starting curriculum state

    terrain_length = 6.0
    terrain_width = 6.0
    num_rows = 4  # number of terrain rows (levels)
    num_cols = 4  # number of terrain cols (types)
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    # trimesh only:
    # slopes above this threshold will be corrected to vertical surfaces
    slope_treshold = 0.75

    @property
    def typ(self):
        return self.type.value


class Arena:
    """terrain + obstacles"""


class TerrainMixin:
    def validate(self):
        # TODO this should go somewhere else ? like terrain config
        if self.cfg.terrain.typ not in ["heightfield"]:
            self.cfg.terrain.curriculum = False

    def _init_terrain(self, tcfg:terrain.TerrainFactoryConfig):
        if self.cfg.terrain.typ in ["plane"]:
            self.terrain = self.scene.add_entity(gs.morphs.Plane(collision=True, fixed=True))
        else:
            factory = terrain.TerrainFactory(
                tcfg.n,
                tcfg.size,
                tcfg.z_offset,
                tcfg.horizontal_scale,
                tcfg.vertical_scale,
                tcfg.subterrain_types
            )

            factory.add_walls(
                n = tcfg.wall_count,
                max_n = tcfg.wall_count_max,
                length = tcfg.wall_length,
                max_length = tcfg.wall_length_max,
                width = tcfg.wall_width,
                max_width = tcfg,
                height = tcfg.wall_height,
                max_height = tcfg.wall_height_max,
                vertical_only = tcfg.walls_vertical_only,
                horizontal_only = tcfg.walls_horizontal_only
            )

            if tcfg.perimeter_walls:
                factory.add_perimeter_walls(
                    thickness = tcfg.perimeter_thickness,
                    height = tcfg.perimeter_height,
                    max_height = tcfg.perimeter_height_max
                )

            self.terrain = self.scene.add_entity(factory.terrain())

        self.terrain.set_friction(self.cfg.terrain.friction)
        # TODO(mbt): Implement Terrain System
        # TODO(mbt): Obstacle System

    def _init_heightfield_terrain(self):
        # add terrain
        typ = self.cfg.terrain.typ
        if typ == "plane":
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif typ == "heightfield":
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif typ is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.typ == "heightfield":
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0  # give a small margin(1.0m)
            self.terrain_x_range[1] = (
                self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            )
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = (
                self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
            )
        elif self.cfg.terrain.typ == "plane":  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length / 2 + 1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length / 2 - 1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length / 2 + 1  # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length / 2 - 1

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        self.scene.clear_debug_objects()
        height_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points)
        height_points[0, :, 0] += self.base_pos[0, 0]
        height_points[0, :, 1] += self.base_pos[0, 1]
        height_points[0, :, 2] = self.measured_heights[0, :]
        # print(f"shape of height_points: ", height_points.shape) # (n_envs, num_points, 3)
        self.scene.draw_debug_spheres(
            height_points[0, :], radius=0.03, color=(0, 0, 1, 0.7)
        )  # only draw for the first env

    def _update_terrain_curriculum(self, env_ids):
        if not self.cfg.terrain.curriculum:
            return

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                height_field=self.utils_terrain.height_field_raw,
            )
        )
        self.height_samples = (
            torch.tensor(self.utils_terrain.heightsamples)
            .view(self.utils_terrain.tot_rows, self.utils_terrain.tot_cols)
            .to(self.device)
        )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.typ in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.n_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.n_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.n_envs, device=self.device),
                (self.n_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.utils_terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.n_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.n_envs))
            num_rows = np.ceil(self.n_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            # plane has limited size, we need to specify spacing base on n_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = min(
                (self.cfg.terrain.plane_length / 2) / (num_rows - 1),
                (self.cfg.terrain.plane_length / 2) / (num_cols - 1),
            )
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.n_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.n_envs]
            self.env_origins[:, 2] = 0.0
            self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (n_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.n_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.typ == "plane":
            return torch.zeros(
                self.n_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.typ == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.base_pos[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.base_pos[:, :3]
            ).unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points / self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.n_envs, -1) * self.cfg.terrain.vertical_scale
