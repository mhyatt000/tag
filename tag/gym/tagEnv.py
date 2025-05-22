from typing import Dict, Tuple, Union

import genesis as gs
import torch

from tag.gym.baseClasses import BaseEnv
from tag.gym.tagConfig import TagConfig


class TagEnv(BaseEnv):
    """Version 1"""

    def __init__(self, args=None, cfg: TagConfig = TagConfig()):
        self.cfg = cfg
        if args is not None:
            self.n_envs = args.n_envs
            self.n_rendered = args.n_rendered
            self.n_envs = args.n_envs
            self.n_rendered = args.n_rendered
        else:
            self.n_envs = self.cfg.sim.num_envs
            self.n_rendered = self.cfg.vis.n_rendered_envs
        self.device = gs.gpu  # CPU Support maybe?
        self.episode_length = self.cfg.sim.episode_length
        self.num_obs = self.cfg.sim.num_obs
        self.num_privileged_obs = self.cfg.sim.num_privileged_obs
        self.max_episode_length = self.cfg.sim.max_episode_length
        self.extras = dict()

        # Init
        gs.init(
            logging_level=args.logging_level if args is not None else "info",
            backend=self.device,
        )

        # Scene
        self.scene = gs.Scene(
            show_viewer=self.cfg.viewer.show_viewer,
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=cfg.solver.joint_limit,
                dt=cfg.solver.dt,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=cfg.vis.show_world_frame,
                n_rendered_envs=args.n_rendered if args is not None else 1,
            ),
        )

        self._init_buffers()

        # Entites

        # Plane
        # TODO: Implement Terrain Class and Options/Implementation
        if cfg.terrain.mesh_type == "plane":
            self.terrain = self.scene.add_entity(
                gs.morphs.Plane(),
            )

        # Go2
        # TODO: Implement Robot Class
        self.robot_1 = self.scene.add_entity(
            gs.morphs.URDF(
                file=cfg.robotCfg.asset.file,
                pos=torch.tensor(cfg.robotCfg.init_state.pos) + torch.tensor([0.0, -0.5, 0.0]),
            )
        )

        self.robot_2 = self.scene.add_entity(
            gs.morphs.URDF(
                file=cfg.robotCfg.asset.file,
                pos=torch.tensor(cfg.robotCfg.init_state.pos) + torch.tensor([0.0, 0.5, 0.0]),
            )
        )

        # TODO: Implement Camera Class and Options
        if self.cfg.vis.visualized:
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(7, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=60,
                GUI=False,
            )

    def build(self):
        self.scene.build(
            n_envs=self.n_envs,
            env_spacing=self.cfg.vis.env_spacing if self.n_rendered > 1 else [0, 0],
        )
        if self.cfg.vis.visualized:
            self.cam.start_recording()

    # TODO: Implement Method - Input should be changed to Robot class when completed
    def set_control_gains(self):
        pass

    # TODO: Properly Implement Step Method - Actions, Updates, etc.
    def step(
        self, actions: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict]:
        """Iterate one step in timespace and updates environment

        Args:
            actions (torch.Tensor):

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict]:
            Returns the following values in a Tuple
                torch.Tensor:               Observation Buffer
                Union[torch.Tensor, None]:  Privileged Observation Buffer
                torch.Tensor:               Reward Buffer
                torch.Tensor:               Reset Buffer
                Dict:                       Extra Values
        """

        # Execute actions

        for k, act in actions.items():
            self.robots[k].act(act)

        self.scene.step()

        # Update Buffers - Implented for Testing
        self._update_buffers()
        # Check termination and reset
        # Compute weward
        # Compute observations
        # Create extras

        # Visualization
        if self.cfg.vis.visualized:
            self.cam.render()

        obs = self.get_observations()
        reward = self.get_reward()
        return obs, reward, term, trunc, info

    # TODO: Implement Reset Method
    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Resets All Environments

        Returns:
            Tuple[torch.Tensor, Dict]: A Tuple of the Reset Buffer and any Extras
        """
        self.reset_buf[:] = torch.ones((self.n_envs,), device=self.device, dtype=gs.tc_int)
        return self.reset_buf, None

    # TODO: Review
    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """Get observation buffer data
        Returns:
            Tuple[torch.Tensor, Dict]: A Tuple of the Observation Buffer and any Extras
        """
        return self._obs

    def compute_observations(self) -> Dict:
        robot_obs = {
            r: {
                "base_link",
                "quat",
                "dofs",
                "vel",
            }
            for r in [_r.uid for _r in self.robots]
        }
        env_obs = {}
        terrain_obs = {}

        obs = robot_obs | env_obs | terrain_obs
        self._obs = obs

    def record_data(self):
        if self.cfg.vis.visualized:
            self.cam.stop_recording(save_to_filename="./tag/gym/mp4/tagV1_video.mp4", fps=60)

    def _init_buffers(self):
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

    def _update_buffers(self):
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
        self.privileged_obs_buf = torch.cat(  # Entire Observation
            [
                # Whatever privileged data
                torch.cat(
                    [  # Testing Data for Robot DOF position
                        torch.rand(12),  # Arb dof for Robot 1
                        torch.rand(12),  # Arb dof for Robot 2
                    ]
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
