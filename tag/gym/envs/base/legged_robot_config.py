from dataclasses import dataclass, field

import torch
from Typing import Optional, Tuple

from .base_config import BaseConfig


@dataclass
class LeggedRobotCfg(BaseConfig):
    @dataclass
    class env:
        num_envs: int = 1
        num_observations: int = 48
        num_privilege_obs: Optional[torch.Tensor] = None
        num_actions: int = 12
        send_timeouts: bool = True
        episode_length_s: int = 20
        debug: bool = False
        debug_viz: bool = False

    # TODO: Further complete terrain implementation
    @dataclass
    class terrain:
        mesh_type: str = "plane"
        friction: float = 1.0
        restitution: float = 0.0

    # TODO: Implement commands class in LeggedRobotCfg

    @dataclass
    class init_state:
        pos: list[float] = [0.0, 0.0, 1.0]
        quat: list[float] = [1.0, 0.0, 0.0, 0.0]
        lin_vel: list[float] = [0.0, 0.0, 0.0]
        ang_vel: list[float] = [0.0, 0.0, 0.0]
        randomize_angle: bool = False
        angle_range: list[float] = [0.0, 0.0]
        default_joint_angles: dict[str, float] = {
            "joint_1": 0.0,
        }

    @dataclass
    class control:
        control_type: str = "P"
        # PD Controller Settings
        kp: dict[str, float] = {"joint_1": 10.0}  # stiffness
        kv: dict[str, float] = {"joint_1": 1.0}  # dampening
        action_scale: float = 0.5
        dt: float = 0.02
        decimation: int = 4

    @dataclass
    class asset:
        dof_names: list[str] = field(default_factory=list)
        file: str = field(default_factory=str)
        links_to_keep: list[str] = field(default_factory=list)
        foot_name: list[str] = field(default_factory=list)
        penalize_contacts_on: list[str] = field(default_factory=list)
        terminate_after_contacts_on: list[str] = field(default_factory=list)
        self_collisions: bool = True
        fix_base_link: bool = False

    @dataclass
    class domain_rand:
        randomize_friction: bool = True
        friction_range: list[float] = [0.5, 1.25]
        randomize_mass: bool = True
        mass_range: list[float] = [-0.1, 0.1]
        push_robots: bool = True
        push_interval_s: int = 15
        max_push_vel_xy: float = 1.0
        randomize_com_displacement: bool = True  # center of mass
        com_displacement_range: list[float] = [-0.01, 0.01]
        latency: bool = True

    # TODO: Add rewards class
    # TODO: Add normalization/obs_scales class
    # TODO: Add noise class

    @dataclass
    class viewer:
        ref_env: int = 0
        show_viewer: bool = True
        pos: list[float] = [5, 0, 3.5]
        lookat: list[float] = [0.0, 0.0, 0.5]
        num_rendered_envs: int = 1
        GUI: bool = False
        add_camera: bool = False

    @dataclass
    class cam_settings:
        res: Tuple = Tuple(640, 480)
        pos: Tuple = Tuple(3.5, 0.0, 2.5)
        lookat: Tuple = Tuple(0, 0, 0.5)
        fov: int = 30
        GUI: bool = False

    @dataclass
    class sim:
        use_implicit_controller: bool = False
        gravity: list[float] = [0.0, 0.0, -9.81]


# TODO: Implement LeggedRobotCfg class for Algorithm
