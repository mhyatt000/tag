from dataclasses import dataclass
from typing import Dict

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
from gymnasium import spaces
import jax
import numpy as np
import torch

from tag.gym.base.config import MJCF, URDF, Control
from tag.names import MENAGERIE
from tag.utils import default

from .robot import Robot, RobotConfig, RobotState

local_dofs = [6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17]
GO2MJCF = MJCF(file=str(MENAGERIE / "unitree_go2" / "go2.xml"))
GO2URDF = URDF(file="urdf/go2/urdf/go2.urdf")  # Genesis/resources

GO2_STAND_JOINTS = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}
GO2_CROUCH_JOINTS = {
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


@dataclass
class Go2State(RobotState):
    joints: Dict[str, float] = default(GO2_STAND_JOINTS)

    # link_pos: torch.Tensor
    # link_quat: torch.Tensor
    # link_vel: torch.Tensor
    # link_links_ang: torch.Tensor

    @property
    def dof_pos(self) -> torch.Tensor:
        """Get the joint positions as a tensor."""
        return torch.tensor(list(self.joints.values()), device=gs.device)

@dataclass
class PDControl:
    # PD Drive parameters:
    # control_type = 'P'
    stiffness = {"joint": 20.0}  # [N*m/rad]
    damping = {"joint": 0.5}  # [N*m*s/rad]
    #  target angle = actionScale * action + defaultAngle
    action_scale = 0.25
    # control frequency 50Hz     ■ E222 multiple spaces after operator
    dt = 0.02
    #  Number of control action updates @ sim DT per policy DT
    decimation = 4


@dataclass
class Go2Config(RobotConfig):
    control: Control = default(Control(kp=20.0, kd=0.5))
    asset: MJCF | URDF = default(GO2URDF)

    state: Go2State = default(
        Go2State(
            pos=[0.0, 0.0, 0.42],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
    )

    foot_name: list[str] = default(["foot"])
    penalize_contacts_on: list[str] = default(["thigh", "calf"])
    terminate_after_contacts_on: list[str] = default(["base"])
    links_to_keep: list[str] = default(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    self_collisions: bool = True

    def _create(self, scene: gs.Scene) -> RigidEntity:
        """Create the robot asset."""
        return self.asset.create(
            scene,
            pos=self.state.pos,
            quat=self.state.quat,
            links_to_keep=self.links_to_keep,
            collision=True,
        )

    def create(self, scene: gs.Scene) -> "Go2Robot":
        """Create a Go2 robot instance."""
        return Go2Robot(scene, self)


@dataclass
class PipeState:
    """Dynamic state that changes after every pipeline step.

    Attributes:
    q: (q_size,) joint position vector
    qd: (qd_size,) joint velocity vector
    x: (num_links,) link position in world frame
    xd: (num_links,) link velocity in world frame
    contact: calculated contacts
    """

    q: jax.Array
    qd: jax.Array
    # x: Transform
    # xd: Motion
    # contact: Optional[Contact]


class Go2Robot(Robot):
    def __init__(self, scene: gs.Scene, cfg: Go2Config):
        super().__init__(scene, cfg)
        # self.robot.set_dofs_stiffness(
        # self.robot.set_dofs_damping(


    @property
    def action_space(self) -> spaces.Box:
        """Define the action space for the Go2 robot."""
        # TODO(dle): Find Correct Joint Ranges
        j = len(self.cfg.state.joints)
        return spaces.Box(low=-np.pi, high=np.pi, shape=(j,), dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Dict:
        """Define the observation space for the Go2 robot."""

        # TODO(dle): Add correct min and max values
        # TODO make dynamic, read an observation to define the shape

        def _float_inf(shape):
            _s = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
            return _s

        ospace = spaces.Dict(
            {
                "base_pos": _float_inf((3,)),
                "base_quat": _float_inf((4,)),
                "base_velo": _float_inf((3,)),
                "base_ang": _float_inf((3,)),
                "link_pos": _float_inf((12, 3)),
                "link_quat": _float_inf((12, 4)),
                "link_vel": _float_inf((12, 3)),
                "link_links_ang": _float_inf((12, 3)),
            }
        )
        return ospace

        # NOTE(dle): Requires Current Genesis Branch
        # "link_acc": spaces.Box(-np.inf, np.inf, shape=(12, 3), dtype=np.float32),


    def observe(self) -> Dict:
        obs = {
            "base": {
                "pos": self.robot.get_pos(),
                "quat": self.robot.get_quat(),
                "inv_quat": self.inv_quat,
                "vel": self.robot.get_vel(),
                "ang": self.robot.get_ang(),
            },
            "link": {
                "pos": self.robot.get_links_pos(),
                "quat": self.robot.get_links_quat(),
                "velo": self.robot.get_links_vel(),
                "ang": self.robot.get_links_ang(),
            },
            "dof": {
                "pos": self.robot.get_dofs_position(self.dofs),
                "vel": self.robot.get_dofs_velocity(self.dofs),
            },
            # NOTE(dle): Requires Current Genesis Branch
            # "link_acc": self.robot.get_links_acc(),
            # NOTE(dle): Requires Current Genesis Branch
            # "link_force": self.robot.get_links_net_contact_force()
        }
        return obs

    # FEATURE
    def randomize(self, cfg):
        pass


    @property
    def feet(self):
        return self.find_link_indices(self.cfg.foot_name)

    @property
    def indices(self):
        idxs = {
            "terminate": self.find_link_indices(self.cfg.terminate_after_contacts_on),
            "penalize": self.find_link_indices(self.cfg.penalize_contacts_on),
            "links": self.find_link_indices(self.cfg.links_to_keep),
            "feet": self.feet,
        }
        return idxs

    def pos_limits(self, soft: int | None = None):
        lim = torch.stack(self.robot.get_dofs_limit(self.dofs), dim=1)
        if soft is not None:
            lim = self.soften_limits(lim, soft)
        return lim

    def torque_limits(self):
        return self.robot.get_dofs_force_range(self.dofs)[1]

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
            - self.batched_d_gains * self.dof_vel
        )
        return torques

    @property
    def contact_forces(self):
        link_contact_forces = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=gs.device,
            dtype=gs.tc_float,
        )
        return link_contact_forces

    def soften_limits(self, lim: np.ndarray, soft: int | None = None):
        m = (lim[:, 0] + lim[:, 1]) / 2
        r = lim[:, 1] - lim[:, 0]

        factor = 0.5 * r * soft
        lim[:, 0] = m - factor
        lim[:, 1] = m + factor
        return lim

    def _observe_walk(
        self,
        inv_base_init_quat,
        global_gravity,
        scales,
        commands,
        default_dof_pos,
        actions,
    ) -> None:
        # TODO(mhyatt) remove need for passed args
        # better handling of internal state

        # TODO(codex) clean for readability

        self.buf = {}
        base_pos = self.robot.get_pos()
        base_quat = self.robot.get_quat()

        base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(base_quat) * inv_base_init_quat,
                base_quat,
            ),
            rpy=True,
            degrees=True,
        )

        inv_base_quat = inv_quat(base_quat)
        base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        projected_gravity = transform_by_quat(global_gravity, inv_base_quat)
        dof_pos = self.robot.get_dofs_position(self.dofs)
        dof_vel = self.robot.get_dofs_velocity(self.dofs)

        # compute observations
        obs = torch.cat(
            [
                base_ang_vel * scales["ang_vel"],  # 3
                projected_gravity,  # 3
                commands * scales["cmd"],  # 3
                (dof_pos - default_dof_pos) * scales["dof_pos"],  # 12
                dof_vel * scales["dof_vel"],  # 12
                actions,  # 12
            ],
            axis=-1,
        )
        return obs

    def reset(self, envs_idx: list[int]):
        super().reset(envs_idx)
