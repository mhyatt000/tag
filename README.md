# tag

A collection of experiments and helper scripts around the brax and mujoco environments.

## Table of Contents
- [Installation](#installation)
- [Development](#development)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Scripts](#scripts)
  - [Terrain](#terrain)

## Installation

The project uses the [uv](https://github.com/astral-sh/uv) package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra brax --extra genesis --extra gl --extra debug
uv run main.py
```

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


## Development

Install additional development tools and set up pre-commit hooks:

```bash
uv sync --dev
uvx pre-commit install
```

## Directory Structure

- `main.py` – example entry point for running environments.
- `tag/` – Python package with experimental code (e.g. `brax/barkour.py`).
- `extras/` – additional resources used in notebooks or experiments.
- `third-party/` – vendored dependencies or assets.
- `pyproject.toml` – project configuration and dependency list.

## Usage

Run experiments directly from the command line:

```bash
python main.py
```

### Scripts

`main.py` provides a simple script that demonstrates training or running Brax environments. Custom scripts can be added under the `tag` package as needed.

### Terrain

Example for creating a composite terrain in Genesis:

```python
import genesis as gs
gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
terrain = scene.add_entity(
    gs.morphs.Terrain(
        n_subterrains=(3, 3),
        subterrain_size=(12.0, 12.0),
        subterrain_types=[
            ['flat_terrain', 'random_uniform_terrain', 'stepping_stones_terrain'],
            ['pyramid_sloped_terrain', 'discrete_obstacles_terrain', 'wave_terrain'],
            ['random_uniform_terrain', 'pyramid_stairs_terrain', 'sloped_terrain']
        ],
        randomize=True
    )
)
```
