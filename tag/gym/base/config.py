from dataclasses import dataclass, field


def default(x):
    return field(default_factory=lambda: x)


def defaultcls(cls):
    return field(default_factory=cls)


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
    env_spacing: list[float] = default([2.5, 2.5])

    def __post_init__(self):
        pass
        # if self.n_rendered_envs == 1:
        # self.env_spacing = [0.0, 0.0]


@dataclass
class Solver:
    collision: bool = True
    joint_limit: bool = True
    dt: float = 0.02  # 50hz robot step


@dataclass
class Sim:
    dt: int = 0.01  # 100hz sim step physics
    num_envs: int = 1


@dataclass
class Task:
    num_actions: int = 12  # arb
    episode_length: int = 120  # for testing
    max_episode_length: int = 1000
    num_obs: int = 20  # arb
    num_privileged_obs: int = 10  # arb


# Robot Configs


@dataclass
class InitState:
    # default_joint_angles: dict[str, float] # = default({"joint", 1.0})  # default pose

    pos: list[float] = default([0.0, 0.0, 1.0])  # spawn position
    quat: list[float] = default([1.0, 0.0, 0.0, 0.0])  # spawn orientation

    randomize_angle: bool = False  # DR - Initial Angle Spawn
    angle_range: list[float] = default([0.0, 0.0])  # min/max angle randomization
    randomize_com_displacement: bool = False  # DR - Center of Mass
    com_displacement_range: list[float] = default([-0.01, 0.01])  # min/max com displacement
    randomize_mass: bool = False  # DR - Mass Range
    mass_range: list[float] = default([-0.1, 0.1])

    # TODO(dle) random stiffness dampening


@dataclass
class State:
    base_pos: list[float] = default([0.0, 0.0, 0.42])  # base link position
    base_quat: list[float] = default([1.0, 0.0, 0.0, 0.0])  # base link orientation
    lin_vel: list[float] = default([0.0, 0.0, 0.0])  # base link linear velocity
    ang_vel: list[float] = default([0.0, 0.0, 0.0])  # base link angular velocity


@dataclass
class Control:
    kp: float = 1.0
    kd: float = 1.0
    control_type: str = "P"
    action_scale: float = 0.5  # TODO(dle) add example in docstring
    decimation: float = 4
    latency: bool = False


@dataclass
class Asset:
    file: str = default("DO NOT REMOVE PLEASE")
    local_dofs: list[int] = default([1, 2, 3])


@dataclass
class RobotConfig:
    init_state: InitState = defaultcls(InitState)
    state: State = default(State())
    control: Control = default(Control())
    asset: Asset = default(Asset())


# Environment Config Class
# TODO: Env config does not neccessarily need a robot.
#       Robot's joints must be known in order to config
@dataclass
class EnvConfig:
    terrain: Terrain = default(Terrain())
    viewer: Viewer = default(Viewer())
    vis: Vis = default(Vis())
    solver: Solver = default(Solver())
    sim: Sim = default(Sim())

    def __post__init__(self):
        if self.sim.num_envs < 1:
            raise ValueError("num_envs must be greater than 0")
        if self.sim.num_envs < self.vis.n_rendered_envs:
            raise ValueError("n_rendered_envs must be less than or equal to num_envs")


# IMPLEMENT: Configurations for Tasks/Rewards/Observations
