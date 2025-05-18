from typing import Any, Dict, Iterable, Protocol, Tuple

import genesis as gs

from .env import EnvConfig


class BaseEnv(Protocol):
    cfg: EnvConfig
    scene: gs.Scene
    max_episode_length: int

    def build(self) -> None: ...

    def reset(self) -> Tuple[Any, Dict[str, Any]]: ...

    def step(self, *actions: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]: ...

    def get_observations(self) -> Tuple[Any, Dict[str, Any]]: ...

    def get_privileged_observations(self) -> Any: ...


class Robot(Protocol):
    robot: gs.Entity
    scene: gs.Scene

    def reset(self) -> None: ...

    def act(self, act: Any | None = None, mode: str = "control") -> None: ...

    def close(self) -> None: ...

    def open(self) -> None: ...


class CameraManager(Protocol):
    cams: Dict[str, gs.Camera]

    def render(self, *, names: Iterable[str], **kwargs: Any) -> Dict[str, Any]: ...
