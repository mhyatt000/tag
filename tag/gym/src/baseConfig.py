from dataclasses import dataclass, field

from rich.pretty import pprint
import tyro


def default(x):
    return field(default_factory=lambda: x)


# Environment Arguments


# TODO: Paintable/Randomized Terrain Implementation
@dataclass
class Terrain:
    mesh_type: str = "plane"
    friction: float = 1.0
    restitution: float = 1.0
    randomize_friction: bool = False  # DR - Terrain Friction
    friction_range: list[float] = default([0.5, 1.25])
    push_robots: bool = False  # DR - Randomized Robot Displacement
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
    env_spacing: list[int] = default([0, 0])


@dataclass
class Solver:
    collision: bool = True
    joint_limit: bool = True


@dataclass
class Sim:
    dt: int = 0.01
    num_envs: int = 1
    num_actions: int = 10
    episode_length: int = 500
    max_episode_length: int = 1000
    num_obs: int = 1  # arb
    num_privileged_obs: int = 1  # arb


# Robot Configs


@dataclass
class InitState:
    pos: list[float] = default([0.0, 0.0, 1.0])  # spawn position
    quat: list[float] = default([1.0, 0.0, 0.0, 0.0])  # spawn orientation
    default_joint_angles: dict[str, float] = default({"joint_1": 0.0})  # default pose
    randomize_angle: bool = False  # DR - Initial Angle Spawn
    angle_range: list[float] = default([0.0, 0.0])  # min/max angle randomization
    randomize_com_displacement: bool = False  # DR - Center of Mass
    com_displacement_range: list[float] = default([-0.01, 0.01])  # min/max com displacement
    randomize_mass: bool = False  # DR - Mass Range
    mass_range: list[float] = default([-0.1, 0.1])


@dataclass
class State:
    base_pos: list[float] = default([0.0, 0.0, 1.0])  # base link position
    base_quat: list[float] = default([1.0, 0.0, 0.0, 0.0])  # base link orientation
    lin_vel: list[float] = default([0.0, 0.0, 0.0])  # base link linear velocity
    ang_vel: list[float] = default([0.0, 0.0, 0.0])  # base link angular velocity


@dataclass
class Control:
    control_type: str = "P"
    kp: float = 10.0  # Can add DR
    kd: float = 1.0  # Can add DR
    action_scale: float = 0.5
    dt: float = 0.02
    decimation: float = 4
    latency: bool = False


@dataclass
class Asset:
    file: str = ""
    local_dofs: list[int] = default([1])


@dataclass
class RobotConfig:
    init_state: InitState = default(InitState())
    state: State = default(State())
    control: Control = default(Control())
    asset: Asset = default(Asset())


# Environment Config Class
@dataclass
class EnvConfig:
    terrain: Terrain = default(Terrain())
    viewer: Viewer = default(Viewer())
    vis: Vis = default(Vis())
    solver: Solver = default(Solver())
    sim: Sim = default(Sim())
    robotCfg: RobotConfig = default(RobotConfig())


# IMPLEMENT: Configurations for Tasks/Rewards/Observations


def main(env: EnvConfig):
    pprint(env)


if __name__ == "__main__":
    tyro.cli(main)
