from dataclasses import dataclass, field

from rich.pretty import pprint
import tyro


def default(x):
    return field(default_factory=lambda: x)


@dataclass
class Terrain:
    mesh_type: str = "plane"
    friction: float = 1.0
    restitution: float = 0.0
    # TODO: Further complete terrain implementation


@dataclass
class State:
    pos: list[float] = default([0.0, 0.0, 1.0])
    quat: list[float] = default([1.0, 0.0, 0.0, 0.0])
    lin_vel: list[float] = default([0.0, 0.0, 0.0])
    ang_vel: list[float] = default([0.0, 0.0, 0.0])
    randomize_angle: bool = False  # TODO(dle) what is this
    angle_range: list[float] = default([0.0, 0.0])
    default_joint_angles: dict[str, float] = default({"joint_1": 0.0})


@dataclass
class Control:
    control_type: str = "P"
    # PD Controller Settings
    # TODO(dle) something got lost with types here dict vs float
    kp: dict[str, float] = 10.0  # stiffness
    kv: dict[str, float] = 1.0  # dampening
    action_scale: float = 0.5
    dt: float = 0.02
    decimation: int = 4


@dataclass
class Asset:
    file: str
    dof_names: list[str] = default([])
    links_to_keep: list[str] = default([])
    foot_name: list[str] = default([])
    penalize_contacts_on: list[str] = default([])
    terminate_after_contacts_on: list[str] = default([])
    self_collisions: bool = True
    fix_base_link: bool = False


@dataclass
class DomainRand:
    """domain randomization (DR)"""

    randomize_friction: bool = True
    friction_range: list[float] = default([0.5, 1.25])
    randomize_mass: bool = True
    mass_range: list[float] = default([-0.1, 0.1])
    push_robots: bool = True
    push_interval_s: int = 15
    max_push_vel_xy: float = 1.0
    randomize_com_displacement: bool = True  # center of mass
    com_displacement_range: list[float] = default([-0.01, 0.01])
    latency: bool = True


# TODO: Add rewards class
# TODO: Add normalization/obs_scales class
# TODO: Add noise class


@dataclass
class Viewer:
    ref_env: int = 0
    show_viewer: bool = True
    pos: list[float] = default([5, 0, 3.5])
    lookat: list[float] = default([0.0, 0.0, 0.5])
    num_rendered_envs: int = 1
    GUI: bool = False
    add_camera: bool = False


@dataclass
class Camera:
    res: tuple = (640, 480)
    pos: tuple = (3.5, 0.0, 2.5)
    lookat: tuple = (0, 0, 0.5)
    fov: int = 30
    GUI: bool = False


@dataclass
class Sim:
    use_implicit_controller: bool = False
    gravity: list[float] = default([0.0, 0.0, -9.81])


"""
@dataclass
class Env:
    num_envs: int = 1
    num_observations: int = 48
    num_privilege_obs: Optional[torch.Tensor] = None
    num_actions: int = 12
    send_timeouts: bool = True
    episode_length_s: int = 20
    debug: bool = False
    debug_viz: bool = False
"""


@dataclass
class Env:
    asset: Asset | None
    terrain: Terrain | None

    state: State = default(State())
    control: Control = default(Control())
    dr: DomainRand = default(DomainRand())
    viewer: Viewer = default(Viewer())
    camera: Camera = default(Camera())
    sim: Sim = default(Sim())

    # TODO: Implement commands class in LeggedRobotCfg


# TODO: Implement LeggedRobotCfg class for Algorithm


def main(cfg: Env):
    pprint(cfg)


if __name__ == "__main__":
    main(tyro.run(Env))
