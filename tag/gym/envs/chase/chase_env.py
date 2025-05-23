"""Chase environment implementation."""

from typing import Dict, Tuple

import genesis as gs
from gymnasium.spaces import Box, Dict
import torch

from tag.gym.base.env import BaseEnv
from tag.gym.envs.terrain_mixin import TerrainEnvMixin
from tag.gym.robots.multi import MultiRobot

from .chase_config import ChaseEnvConfig
from .utils import create_camera, create_robots, create_scene


class Chase(TerrainEnvMixin, BaseEnv):
    """Simple two-robot chase environment."""

    def __init__(self, args: Dict | None = None, cfg: ChaseEnvConfig = ChaseEnvConfig()):
        """Create a new environment instance."""
        self.cfg: ChaseEnvConfig = cfg
        # if args is not None:
        #     self.n_envs = args.n_envs
        #     self.n_rendered = args.n_rendered
        #     self.n_envs = args.n_envs
        #     self.n_rendered = args.n_rendered
        # else:
        #     self.n_envs = self.cfg.sim.num_envs
        #     self.n_rendered = self.cfg.vis.n_rendered_envs
        self.n_envs = 4
        self.n_rendered = 4
        self.env_spacing = (2.5, 2.5)
        self.device = gs.gpu  # CPU Support maybe?
        # self.episode_length = self.cfg.sim.episode_length
        # self.num_obs = self.cfg.sim.num_obs
        # self.num_privileged_obs = self.cfg.sim.num_privileged_obs
        # self.max_episode_length = self.cfg.sim.max_episode_length

        # Init
        gs.init(
            logging_level=args.logging_level if args is not None else "info",
            backend=self.device,
        )

        # Scene
        self.scene: gs.Scene = create_scene(cfg, self.n_rendered)

        # self._init_buffers()
        self._init_spaces()

        # Entities
        self.robots: MultiRobot = create_robots(self.scene, self.cfg.robotCfg)
        self.cam = create_camera(self.scene, self.cfg.vis.visualized)

        self.build()

    def build(self):
        self.scene.build(
            n_envs=self.n_envs,
            env_spacing=self.env_spacing if self.n_rendered > 1 else [0, 0],
        )
        self.build_terrain()
        if self.cam is not None:
            self.cam.start_recording()

    # TODO: Implement Method - Input should be changed to Robot class when completed
    def set_control_gains(self):
        pass

    # TODO: Properly Implement Step Method - Actions, Updates, etc.
    def step(self, actions: Dict) -> Tuple[Dict, None, None, None, None]:
        """Advance the simulation by one step."""
        # Execute actions

        self.robots.act(actions)

        self.scene.step()

        # Update Buffers - Implented for Testing
        # self._update_buffers()
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
    def reset(self) -> Tuple[Dict, None]:
        """Reset the environment state."""
        return self.action_space.sample(), None

    # TODO: Review
    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """Get observation buffer data
        Returns:
            Tuple[torch.Tensor, Dict]: A Tuple of the Observation Buffer and any Extras
        """
        return self._obs

    def compute_observations(self) -> Dict:
        """Collect observations from robots and environment."""
        robot_obs = {
            r: {
                "base_pos": r.robot.get_pos(),
                "base_quat": r.robot.get_quat(),
                "base_velo": r.robot.get_vel(),
                "base_ang": r.robot.get_ang(),
                "link_pos": r.robot.get_links_pos(),
                "link_quat": r.robot.get_links_quat(),
                "link_vel": r.robot.get_links_vel(),
                "link_links_ang": r.robot.get_links_ang(),
                "link_acc": r.robot.get_links_pos(),  # newer version of genesis, fake for now
            }
            for r in self.robots
        }
        env_obs = {}
        terrain_obs = {}

        obs = robot_obs | env_obs | terrain_obs
        self._obs = obs
        return self.get_observations()

    def record_data(self):
        """Finalize and save any visual recordings."""
        if self.cfg.vis.visualized:
            self.cam.stop_recording(save_to_filename="./tag/gym/mp4/tagV1_video.mp4", fps=60)

    def _init_buffers(self):
        """Allocate internal tensors used for stepping the environment."""
        self.obs_buf = torch.zeros((self.n_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.n_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.reset_buf = torch.ones((self.n_envs,), device=self.device, dtype=gs.tc_int)
        self.rew_buf = torch.zeros((self.n_envs,), device=self.device, dtype=gs.tc_float)
        self.episode_length_buf = torch.zeros((self.n_envs,), device=self.device, dtype=gs.tc_int)

        # TODO: Implement buffer initialization for Go2 Models once Robot class is implemented
        # self.base_lin_vel = torch.zeros((self.n_envs, 3), device=gs.device, dtype=gs.tc_float)
        # self.base_ang_vel = torch.zeros((self.n_envs, 3), device=gs.device, dtype=gs.tc_float)

    def _init_spaces(self):
        """Define observation and action spaces."""
        self.observation_space = Dict(
            {
                "Robots": Dict(
                    {
                        "r1": Dict(  # Low and High Need to be Fixed
                            {
                                "base_pos": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "base_quat": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "base_velo": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "base_ang": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "link_pos": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),
                                "link_quat": Box(low=-2, high=2, shape=(self.n_envs, 12, 4)),
                                "link_vel": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),
                                "link_links_ang": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),
                                "link_acc": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),  # fake for now
                            }
                        ),
                        "r2": Dict(
                            {
                                "base_pos": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "base_quat": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "base_velo": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "base_ang": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                                "link_pos": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),
                                "link_quat": Box(low=-2, high=2, shape=(self.n_envs, 12, 4)),
                                "link_vel": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),
                                "link_links_ang": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),
                                "link_acc": Box(low=-2, high=2, shape=(self.n_envs, 12, 3)),  # fake for now
                            }
                        ),
                    }
                ),
                "Terrain": Dict({}),
                "Obstacles": Dict({}),
            }
        )

        self.action_space = Dict(
            {
                "r1": Box(low=-2, high=2, shape=(self.n_envs, 12)),
                "r2": Box(low=-2, high=2, shape=(self.n_envs, 12)),
            }
        )

    def _update_buffers(self):
        """Example buffer update for testing."""
        self.episode_length_buf += 1
        self.obs_buf = torch.cat(  # Entire Observation
            [
                # Robot, Environment, Terrain, etc
                torch.cat(  # Testing Data for Robot Position
                    [
                        torch.rand(4),  # Arb position for Robot 1
                        torch.rand(4),
                    ]  # Arb position for Robot 2
                )
            ]
        )

        # TODO: Implement Reset Check
        # Fake Reward Step
        self.rew_buf[:] = 0.0
        values = [i for i in range(len(self.rew_buf))]
        for i in range(len(values)):
            rew = values[i]
            self.rew_buf += rew

    # Silent Todo - Implement Domain Randomization
