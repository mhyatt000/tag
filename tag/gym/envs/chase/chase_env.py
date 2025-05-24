"""Chase environment implementation."""

from typing import Tuple

import genesis as gs
from gymnasium.spaces import Dict
import torch

from tag.gym.base.env import BaseEnv
from tag.gym.envs.terrain_mixin import TerrainEnvMixin
from tag.gym.robots.multi import MultiRobot
from tag.gym.terrain.terrain import Terrain

from .chase_config import ChaseEnvConfig
from .utils import create_camera, create_scene


class Chase(BaseEnv, TerrainEnvMixin):
    """Simple two-robot chase environment."""

    def __init__(self, cfg: ChaseEnvConfig = ChaseEnvConfig()):
        """Create a new environment instance."""
        super().__init__(cfg)
        self.cfg: ChaseEnvConfig = cfg
        # TODO(dle): Have these pull from arguments/config
        self.n_envs = 1
        self.n_rendered = 1
        self.n_rendered = min(self.n_envs, self.n_rendered)

        self.env_spacing = (2.5, 2.5)
        self.n_robots = 2  # TODO(dle): Place Default Value Somewhere in Task or Config

        # Scene
        self.scene: gs.Scene = create_scene(cfg, self.n_rendered)

        # Entities

        # TODO(mbt): Implement Terrain System
        # FEATURE: Obstacle System
        # NOTE(dle): Terrain Class Placeholder
        self.terrain = Terrain(self.scene)

        # TODO(mbt): Implement Color System
        self.robots = MultiRobot(
            self.scene,
            self.cfg.robot,
            self.n_robots,
            self.n_envs,  # NOTE(dle):  Temp Fix
            [  ### NOTE(dle): Placeholder Color System
                (1, 0.5, 0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (1.0, 0.0, 0.0, 1.0),
                None,
            ],  ###
        )

        self.cam = create_camera(self.scene, self.cfg.vis.visualized)

        self._init_spaces()

        # self._init_buffers()

        self.build()

    def build(self):
        self.scene.build(
            n_envs=self.n_envs,
            env_spacing=self.env_spacing if self.n_rendered > 1 else [0, 0],
        )
        if self.cam is not None:
            self.cam.start_recording()

    # TODO: Implement Method - Should this method be in another class?
    def set_control_gains(self):
        pass

    # TODO: Properly Implement Step Method - Actions, Updates, etc.
    def step(self, actions: dict) -> Tuple[dict, None, None, None, None]:
        """Advance the simulation by one step."""
        # Execute actions

        self.robots.act(actions)

        self.scene.step()

        # Visualization
        if self.cfg.vis.visualized:
            self.cam.render()

        obs = self.compute_observations()
        # TODO(dle): Implement Dummy Reward System
        # reward = self.get_reward()
        # return obs, reward, term, trunc, info
        return obs, None, None, None, None

    # TODO: Implement Reset Method
    def reset(self) -> Tuple[dict, None]:
        """Reset the environment state."""
        return self.action_space.sample(), None

    # NOTE(dle): Do we need?
    def get_observations(self) -> Tuple[torch.Tensor, dict]:
        return self._obs

    def compute_observations(self) -> dict:
        """Collect observations from robots and environment."""
        robot_obs = self.robots.compute_observations()
        env_obs = {}
        terrain_obs = {}

        obs = {"robots": robot_obs} | env_obs | terrain_obs
        self._obs = obs
        return self.get_observations()

    def _init_spaces(self):
        """Define observation and action spaces."""
        self.observation_space = Dict(
            {
                # TODO: This needs to be properly tiled, solving by passing n_envs through robots
                "robots": self.robots.observation_space,
                "terrain": Dict({}),
                "obstacles": Dict({}),
            }
        )

        self.action_space = self.robots.action_space
