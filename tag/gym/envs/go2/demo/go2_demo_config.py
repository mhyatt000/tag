from dataclasses import dataclass

from Typing import Tuple

from gym.envs.base.legged_robot_config import LeggedRobotCfg


@dataclass
class Go2DemoCfg(LeggedRobotCfg):
    @dataclass
    class env(LeggedRobotCfg.env):
        num_envs: int = 1

    @dataclass
    class terrain(LeggedRobotCfg.terrain):
        mesh_type: str = "plane"
        friction: float = 1.0
        restitution: float = 0.0

    @dataclass
    class init_state_1(LeggedRobotCfg.init_state):
        pos: list[float] = [0.0, -0.5, 0.42]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    @dataclass
    class init_state_2(LeggedRobotCfg.init_state):
        pos: list[float] = [0.0, 0.5, 0.42]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    @dataclass
    class control(LeggedRobotCfg.control):
        kp: dict[str, float] = {"joint": 20.0}  # stiffness
        kv: dict[str, float] = {"joint": 0.5}  # dampening
        action_scale: float = 0.25

    @dataclass
    class asset(LeggedRobotCfg.asset):
        file: str = "urdf/go2/urdf/go2.urdf"
        dof_names: list[str] = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        foot_name: list[str] = ["foot"]
        penalize_contacts_on: list[str] = ["thigh", "calf"]
        terminate_after_contacts_on: list[str] = ["base"]
        links_to_keep: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self_collisions: bool = True

    @dataclass
    class domain_rand(LeggedRobotCfg.domain_rand):
        # These are not implemented
        randomize_friction: bool = True
        friction_range: list[float] = [0.2, 1.7]
        randomize_mass: bool = True
        mass_range: list[float] = [-0.1, 0.1]
        push_robots: bool = True
        push_interval_s: int = 15
        max_push_vel_xy: float = 0.5
        randomize_com_displacement: bool = True  # center of mass
        com_displacement_range: list[float] = [-0.01, 0.01]
        latency: bool = True

    @dataclass
    class viewer(LeggedRobotCfg.viewer):
        ref_env: int = 0
        show_viewer: bool = False
        pos: list[float] = [5, 0, 3.5]
        lookat: list[float] = [0.0, 0.0, 0.5]
        num_rendered_envs: int = 1
        GUI: bool = False
        add_camera: bool = True

    @dataclass
    class cam_settings(LeggedRobotCfg.cam_settings):
        res: Tuple = Tuple(1280, 720)
        pos: Tuple = Tuple(5, 0.0, 3.5)
        lookat: Tuple = Tuple(0, 0, 0.5)
        fov: int = 65
        GUI: bool = False

    @dataclass
    class sim:
        use_implicit_controller: bool = False
        gravity: list[float] = [0.0, 0.0, -9.81]
