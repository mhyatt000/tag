# tag

A collection of experiments and helper scripts around the brax and mujoco environments.

## Table of Contents
- [Installation](#installation)
- [Development](#development)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Scripts](#scripts)

## Installation

The project uses the [uv](https://github.com/astral-sh/uv) package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra brax --extra genesis --extra gl --extra debug
uv run main.py
```

## Development

Install additional development tools and set up pre-commit hooks:

```bash
uv sync --extra dev
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
