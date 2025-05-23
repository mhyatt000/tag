from dataclasses import dataclass
from typing import Dict

import genesis as gs
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
            local_dofs=default([6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17]),
        )
    )

    init_state: InitState = default(
        InitState(
            pos=default([0.0, 0.0, 0.42]),
            default_joint_angles=default(
                {
                    "FL_hip_joint": 0.1,
                    "RL_hip_joint": 0.1,
                    "FR_hip_joint": -0.1,
                    "RR_hip_joint": -0.1,
                    "FL_thigh_joint": 0.8,
                    "RL_thigh_joint": 1.0,
                    "FR_thigh_joint": 0.8,
                    "RR_thigh_joint": 1.0,
                    "FL_calf_joint": -1.5,
                    "RL_calf_joint": -1.5,
                    "FR_calf_joint": -1.5,
                    "RR_calf_joint": -1.5,
                }
            ),
        )
    )


class Go2Robot(Robot):
    def __init__(self, scene: gs.Scene, cfg: Go2Config, uid: str):
        self.cfg = cfg
        self.robot = scene.add_entity(gs.morphs.URDF(file=cfg.asset.file, pos=cfg.init_state.pos))

    def reset(self):
        """To Be implemented
        self.robot.set_pos()
        """
        return self.observe_state()

    def reset_idx(self, idx):
        pass

    def act(self, action: torch.Tensor, mode: str = "position"):
        # Velocity/Force if needed
        if mode == "position":
            self.robot.control_dofs_position(
                position=action,
                dofs_idx_local=np.array(
                    [6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17]  # Couldn't get config to work for some reason
                ),
            )

    def observe_state(self) -> Dict:
        obs = {
            "base_pos": self.robot.get_pos(),
            "base_quat": self.robot.get_quat(),
            "base_velo": self.robot.get_velo(),
            "base_ang": self.robot.get_ang(),
            "link_pos": self.robot.get_links_pos(),
            "link_quat": self.robot.get_links_quat(),
            "link_vel": self.robot.get_links_vel(),
            "link_links_ang": self.robot.get_links_ang(),
            "link_acc": self.robot.get_links_acc(),
            # :link_force": self.robot.get_links_net_contact_force() # Newer Genesis Version Needed
        }
        return obs

    def randomize(self, cfg):
        pass
