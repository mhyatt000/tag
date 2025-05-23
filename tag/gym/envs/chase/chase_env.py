"""Chase environment implementation."""

from tag.gym.robots.go2 import Go2Config, Go2Robot
from typing import Dict as TDict
from typing import Tuple

import genesis as gs
from gymnasium.spaces import Dict
import torch

from tag.gym.base.env import BaseEnv
from tag.gym.envs.terrain_mixin import TerrainEnvMixin
from tag.gym.robots.multi import MultiRobot

from .chase_config import ChaseEnvConfig
from .utils import create_camera, create_robots, create_scene


class Chase(BaseEnv,TerrainEnvMixin):
    """Simple two-robot chase environment."""

    def __init__(self, cfg: ChaseEnvConfig = ChaseEnvConfig()):
                 # , args: TDict | None = None, cfg: ChaseEnvConfig = ChaseEnvConfig()):
        """Create a new environment instance."""
        super().__init__(cfg)
        self.cfg: ChaseEnvConfig = cfg
        self.n_envs = 4
        self.n_rendered = 4
        self.env_spacing = (2.5, 2.5)

        # Scene
        self.scene: gs.Scene = create_scene(cfg, self.n_rendered)

        # TODO fix
        # self._init_terrain()

        # Entities
        # self.robots: MultiRobot = create_robots(self.scene, self.cfg.robotCfg)
        self.robots = Go2Robot(self.scene, self.cfg.robot, 'r1')

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

    # TODO: Implement Method - Input should be changed to Robot class when completed
    def set_control_gains(self):
        pass

    # TODO: Properly Implement Step Method - Actions, Updates, etc.
    def step(self, actions: TDict) -> Tuple[TDict, None, None, None, None]:
        """Advance the simulation by one step."""
        # Execute actions

        self.robots.act(actions)

        self.scene.step()

        # Check termination and reset
        # Compute weward
        # Compute observations
        # Create extras

        # Visualization
        if self.cfg.vis.visualized:
            self.cam.render()

        obs = self.compute_observations()
        # reward = self.get_reward()
        # return obs, reward, term, trunc, info
        return obs, None, None, None, None

    # TODO: Implement Reset Method
    def reset(self) -> Tuple[TDict, None]:
        """Reset the environment state."""
        return self.action_space.sample(), None

    # TODO: Review
    def get_observations(self) -> Tuple[torch.Tensor, TDict]:
        """Get observation buffer data
        Returns:
            Tuple[torch.Tensor, Dict]: A Tuple of the Observation Buffer and any Extras
        """
        return self._obs

    def compute_observations(self) -> TDict:
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
                "robots": self.robots.observation_space,
                "terrain": Dict({}),
                "obstacles": Dict({}),
            }
        )

        self.action_space = self.robots.action_space
