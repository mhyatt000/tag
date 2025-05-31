from dataclasses import dataclass
from typing import Dict

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity
from gymnasium import spaces
import numpy as np
import torch

from tag.gym.base.config import MJCF, Control
from tag.names import MENAGERIE
from tag.utils import default

from .robot import Robot, RobotConfig, RobotState

H1MJCF = MJCF(file=str(MENAGERIE / "unitree_h1" / "h1.xml"))


@dataclass
class H1State(RobotState):
    """State for the H1 robot."""

    joints: Dict[str, float] = default({})

    @property
    def dof_pos(self) -> torch.Tensor:
        return torch.tensor(list(self.joints.values()), device=gs.device)


@dataclass
class H1Config(RobotConfig):
    """Config for the H1 robot."""

    asset: MJCF = default(H1MJCF)
    control: Control = default(Control(kp=20.0, kd=0.5))
    state: H1State = default(H1State())

    foot_name: list[str] = default(["foot"])
    links_to_keep: list[str] = default([])
    self_collisions: bool = True

    def _create(self, scene: gs.Scene) -> RigidEntity:
        return self.asset.create(
            scene,
            pos=self.state.pos,
            quat=self.state.quat,
            links_to_keep=self.links_to_keep,
            collision=True,
        )

    def create(self, scene: gs.Scene) -> "H1Robot":
        return H1Robot(scene, self)


class H1Robot(Robot):
    def __init__(self, scene: gs.Scene, cfg: H1Config):
        super().__init__(scene, cfg)

    @property
    def action_space(self) -> spaces.Box:
        j = len(self.cfg.state.joints)
        return spaces.Box(low=-np.pi, high=np.pi, shape=(j,), dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Dict:
        def _float_inf(shape):
            return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

        return spaces.Dict(
            {
                "base_pos": _float_inf((3,)),
                "base_quat": _float_inf((4,)),
                "base_velo": _float_inf((3,)),
                "base_ang": _float_inf((3,)),
            }
        )

    def observe(self) -> Dict:
        return {
            "base": {
                "pos": self.robot.get_pos(),
                "quat": self.robot.get_quat(),
                "inv_quat": self.inv_quat,
                "vel": self.robot.get_vel(),
                "ang": self.robot.get_ang(),
            },
            "dof": {
                "pos": self.robot.get_dofs_position(self.dofs),
                "vel": self.robot.get_dofs_velocity(self.dofs),
            },
        }

