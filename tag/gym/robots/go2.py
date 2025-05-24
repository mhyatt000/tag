from dataclasses import dataclass
from typing import Dict, Tuple

import genesis as gs
from gymnasium import spaces
import numpy as np
import torch

from tag.gym.base.config import (
    Asset,
    Control,
    InitState,
    RobotConfig,
    default,
)
from tag.gym.robots.robot import Robot


@dataclass
class Go2Config(RobotConfig):
    control: Control = default(Control(kp=40.0, kd=2.0))
    asset: Asset = default(
        Asset(
            file="urdf/go2/urdf/go2.urdf",
            local_dofs=[6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17],
        )
    )

    init_state: InitState = default(
        InitState(
            # default_joint_angles={
            # "FL_hip_joint": 0.1,
            # "RL_hip_joint": 0.1,
            # "FR_hip_joint": -0.1,
            # "RR_hip_joint": -0.1,
            # "FL_thigh_joint": 0.8,
            # "RL_thigh_joint": 1.0,
            # "FR_thigh_joint": 0.8,
            # "RR_thigh_joint": 1.0,
            # "FL_calf_joint": -1.5,
            # "RL_calf_joint": -1.5,
            # "FR_calf_joint": -1.5,
            # "RR_calf_joint": -1.5,
            # },
            pos=[0.0, 0.0, 0.42],
        )
    )


class Go2Robot(Robot):
    def __init__(self, scene: gs.Scene, cfg: Go2Config, n_envs: int, color: Tuple | None):
        # TODO: Figure out spaces without passing n_envs through
        self.cfg = cfg
        self.robot = scene.add_entity(
            gs.morphs.URDF(
                file=cfg.asset.file,
                pos=cfg.init_state.pos,
            ),
            ### NOTE(dle): Placeholder Color System
            surface=gs.surfaces.Default(color=color),
            ###
        )

        # TODO(dle): Add correct min and max values
        self.observation_space = spaces.Dict(
            {
                "base_pos": spaces.Box(-np.inf, np.inf, shape=(n_envs, 3), dtype=np.float32),
                "base_quat": spaces.Box(-np.inf, np.inf, shape=(n_envs, 4), dtype=np.float32),
                "base_velo": spaces.Box(-np.inf, np.inf, shape=(n_envs, 3), dtype=np.float32),
                "base_ang": spaces.Box(-np.inf, np.inf, shape=(n_envs, 3), dtype=np.float32),
                "link_pos": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 3), dtype=np.float32),
                "link_quat": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 4), dtype=np.float32),
                "link_vel": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 3), dtype=np.float32),
                "link_links_ang": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 3), dtype=np.float32),
                # NOTE(dle): Requires Current Genesis Branch
                # "link_acc": spaces.Box(-np.inf, np.inf, shape=(12, 3), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,  # TODO(dle): Find Correct Joint Ranges
            high=1.0,
            shape=(n_envs, len(cfg.asset.local_dofs)),
            dtype=np.float32,
        )

    def reset(self):
        # TODO
        return self.observe_state()

    def reset_idx(self, idx):
        # TODO
        pass

    def act(self, action: torch.Tensor, mode: str = "position"):
        # FEATURE: Velocity/Force if needed
        # NOTE(dle): dofs_idx_local should import from Go2Config, needs to be fixed.
        if mode == "position":
            self.robot.control_dofs_position(
                position=action,
                dofs_idx_local=np.array([6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17]),
            )

    def observe_state(self) -> Dict:
        obs = {
            "base_pos": self.robot.get_pos(),
            "base_quat": self.robot.get_quat(),
            "base_velo": self.robot.get_vel(),
            "base_ang": self.robot.get_ang(),
            "link_pos": self.robot.get_links_pos(),
            "link_quat": self.robot.get_links_quat(),
            "link_vel": self.robot.get_links_vel(),
            "link_links_ang": self.robot.get_links_ang(),
            # NOTE(dle): Requires Current Genesis Branch
            # "link_acc": self.robot.get_links_acc(),
            # NOTE(dle): Requires Current Genesis Branch
            # "link_force": self.robot.get_links_net_contact_force()
        }
        return obs

    # FEATURE
    def randomize(self, cfg):
        pass

    def compute_observations(self) -> Dict:
        return self.observe_state()
