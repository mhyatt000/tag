
# Quick Start

## via UV

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv run main.py

##  via Conda

    conda create -n tag python=3.11
    conda activate tag
    pip install -e .
    python main.py

## Protocol Interfaces

```mermaid
classDiagram
    direction TD

    class BaseEnv {
        +cfg: EnvConfig
        +scene: gs.Scene
        +max_episode_length: int
        +build()
        +reset()
        +step()
        +get_observations()
        +get_privileged_observations()
    }

    class Robot {
        +robot: gs.Entity
        +scene: gs.Scene
        +reset()
        +act(act, mode="control")
        +close()
        +open()
    }

    class CameraManager {
        +cams: Dict[str, gs.Camera]
        +render(names)
    }

    class RobotEnv {
        +cams: Dict[str, gs.Camera]
        +render(names)
    }

    BaseEnv <|.. RobotEnv
    CameraManager <|.. RobotEnv

    class FrankaEnv
    FrankaEnv --|> RobotEnv
    FrankaEnv ..> Franka : uses

    class XArm7Env
    XArm7Env --|> RobotEnv
    XArm7Env ..> XArm7 : uses

    class Franka
    Franka ..|> Robot

    class XArm7
    XArm7 ..|> Robot
```

