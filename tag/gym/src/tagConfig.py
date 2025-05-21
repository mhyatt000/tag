# TODO: Fix File Structure
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field

from baseConfig import EnvConfig, RobotConfig
from rich.pretty import pprint
import tyro


def default(x):
    return field(default_factory=lambda: x)


# Environment Arguments


@dataclass
class Terrain:
    mesh_type: str = "plane"
    friction: float = 1.0
    restitution: float = 1.0
    randomize_friction: bool = False
    friction_range: list[float] = default([0.5, 1.25])
    push_robots: bool = False
    push_interval_s: int = 15
    max_push_vel_xy: float = 1.0


@dataclass
class Viewer:
    show_viewer: bool = False


@dataclass
class Vis:
    visualized: bool = True
    show_world_frame: bool = True
    n_rendered_envs: int = 1
    env_spacing: list[int] = default([2.5, 2.5])


@dataclass
class Solver:
    collision: bool = True
    joint_limit: bool = True
    dt: float = 0.02


@dataclass
class Sim:
    dt: int = 0.01
    num_envs: int = 1
    num_actions: int = 12  # arb
    episode_length: int = 120  # for testing
    max_episode_length: int = 1000
    num_obs: int = 20  # arb
    num_privileged_obs: int = 10  # arb


# Robot Configs


@dataclass
class InitState:
    pos: list[float] = default([0.0, 0.0, 0.42])
    quat: list[float] = default([1.0, 0.0, 0.0, 0.0])
    default_joint_angles: dict[str, float] = default(
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
    )
    randomize_angle: bool = False
    angle_range: list[float] = default([0.05, 0.05])
    randomize_com_displacement: bool = False
    com_displacement_range: list[float] = default([-0.01, 0.01])
    randomize_mass: bool = False
    mass_range: list[float] = default([-0.1, 0.1])


@dataclass
class State:
    base_pos: list[float] = default([0.0, 0.0, 0.42])
    base_quat: list[float] = default([1.0, 0.0, 0.0, 0.0])
    lin_vel: list[float] = default([0.0, 0.0, 0.0])
    ang_vel: list[float] = default([0.0, 0.0, 0.0])


@dataclass
class Control:
    control_type: str = "P"
    kp: float = 40.0  # should be changed
    kd: float = 2.0  # should be changed
    action_scale: float = 0.5
    dt: float = 0.02
    decimation: float = 4
    latency: bool = False


@dataclass
class Asset:
    file: str = "urdf/go2/urdf/go2.urdf"
    local_dofs: list[int] = default([6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17])


@dataclass
class Go2Config(RobotConfig):
    init_state: InitState = default(InitState())
    state: State = default(State())
    control: Control = default(Control())
    asset: Asset = default(Asset())


# Environment Config Class
@dataclass
class TagConfig(EnvConfig):
    terrain: Terrain = default(Terrain())
    viewer: Viewer = default(Viewer())
    vis: Vis = default(Vis())
    solver: Solver = default(Solver())
    sim: Sim = default(Sim())
    robotCfg: Go2Config = default(Go2Config())


# IMPLEMENT: Configurations for Tasks/Rewards/Observations


def main(env: TagConfig):
    pprint(env)


if __name__ == "__main__":
    tyro.cli(main)
