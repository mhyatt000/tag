from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.utils.geom import inv_quat
from gymnasium import spaces
import torch

from tag.gym.base.config import Asset, Control, State
from tag.protocols import Wraps, _Robot
from tag.utils import defaultcls

_counter = itertools.count()


@dataclass
class RobotState(State):
    pass


@dataclass
class RobotConfig(ABC):
    asset: Asset = defaultcls(Asset)
    state: RobotState = defaultcls(RobotState)
    control: Control = defaultcls(Control)

    @property
    def dof_names(self) -> list[str]:
        return list(getattr(self.state, "joints", {}).keys())

    def _create(self, scene) -> RigidEntity:
        """create robot asset"""
        return self.asset.create(scene, pos=self.state.pos, quat=self.state.quat)

    @abstractmethod
    def create(self, scene) -> "Robot":
        """Create a robot instance."""
        return Robot(scene, self)


class Robot(_Robot, Wraps):
    REGISTRY: dict[str, "Robot"] = {}

    def __init__(self, scene, cfg: RobotConfig):
        """Initialize the robot with a scene and configuration."""
        self.cfg = cfg
        self.robot = cfg._create(scene)

        # Initialize observation and action spaces
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Dict({})

        # Register the robot
        self.register()

    #
    # Utilities
    #

    @property
    def B(self):
        return self._solver.n_envs

    @property
    def name(self) -> str:
        """Return the name of the robot."""
        return self.idx  # from Entity

        # if getattr(self, "_name", None) is None:
        # self._name = uuid.uuid4().hex
        # self._uid = str(next(_counter))
        # self._morph = self.__class__.__name__

    def register(self) -> None:
        """Register the robot obj in the registry."""
        if self.name in Robot.REGISTRY:
            raise ValueError(f"Robot with name {self.name} already registered.")
        Robot.REGISTRY[self.name] = self

    def find_link_indices(self, names):
        """Find the indices of the links in the robot."""
        return [
            link.idx - self.robot.link_start for link in self.robot.links if any(name in link.name for name in names)
        ]

    @property
    def link_names(self):
        return [link.name for link in self.robot.links]

    #
    # Common robot helpers
    #

    @property
    def wrapped(self) -> RigidEntity:
        return self.robot

    @property
    def pos(self) -> torch.Tensor:
        return torch.tensor(self.robot.get_pos(), device=gs.device, dtype=gs.tc_float)

    @property
    def quat(self) -> torch.Tensor:
        return torch.tensor(self.robot.get_quat(), device=gs.device, dtype=gs.tc_float)

    @property
    def inv_quat(self) -> torch.Tensor:
        return inv_quat(self.quat)

    @property
    def dofs(self) -> list[int]:
        return sum(
            [self.robot.get_joint(name).dofs_idx_local for name in self.cfg.dof_names],
            [],
        )

    def act(self, action: torch.Tensor, mode: str = "position") -> None:
        if mode == "position":
            self.robot.control_dofs_position(position=action, dofs_idx_local=self.dofs)

    def reset(self, envs_idx: list[int]) -> None:
        b = len(envs_idx)
        if b == 0:
            return

        def _tile(x):
            return torch.tensor(x, device=gs.device).tile((b, 1))

        kp = torch.tensor([self.cfg.control.kp] * len(self.dofs))
        kd = torch.tensor([self.cfg.control.kd] * len(self.dofs))
        self.robot.set_dofs_kp(kp, dofs_idx_local=self.dofs)
        self.robot.set_dofs_kv(kd, dofs_idx_local=self.dofs)
        self.robot.set_dofs_position(
            position=_tile(list(self.cfg.state.joints.values())),
            dofs_idx_local=self.dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_pos(_tile(self.cfg.state.pos), zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(_tile(self.cfg.state.quat), zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)
