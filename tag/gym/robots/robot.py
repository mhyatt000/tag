import itertools

from gymnasium import spaces
import torch

from tag.gym.base.config import RobotConfig

# import uuid

_counter = itertools.count()


class Robot:
    cfg: RobotConfig

    observation_space: spaces.Space
    action_space: spaces.Space
    REGISTRY: dict[str, "Robot"] = {}

    @property
    def name(self) -> str:
        """Return the name of the robot."""
        if getattr(self, "_name", None) is None:
            # self._name = uuid.uuid4().hex
            self._uid = str(next(_counter))
            self._morph = self.__class__.__name__
        return self._uid

    def register(self) -> None:
        """Register the robot obj in the registry."""
        if self.name in Robot.REGISTRY:
            raise ValueError(f"Robot with name {self.name} already registered.")
        Robot.REGISTRY[self.name] = self

    def reset(self) -> None: ...

    def act(self, action: torch.Tensor, mode: str = "control") -> None: ...
