"""Simple environment curriculum management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Curriculum:
    """Utility to adapt environment parameters over training.

    Parameters
    ----------
    env: object
        Environment instance exposing an ``apply_curriculum`` method.
    steps: int
        Number of update steps over which the curriculum progresses from
        0 to 1. The progress value is clamped to ``[0, 1]``.
    """

    env: object
    steps: int
    step: int = 0

    def update(self, step: Optional[int] = None) -> None:
        """Advance the curriculum and update the environment.

        Parameters
        ----------
        step: int | None
            If provided, sets the internal step counter to ``step`` before
            updating. Otherwise the counter is incremented by one.
        """

        if step is None:
            self.step += 1
        else:
            self.step = step

        progress = min(1.0, max(0.0, self.step / max(1, self.steps)))
        if hasattr(self.env, "apply_curriculum"):
            self.env.apply_curriculum(progress)

    def reset(self) -> None:
        """Reset the curriculum to the initial state."""
        self.step = 0
        if hasattr(self.env, "apply_curriculum"):
            self.env.apply_curriculum(0.0)
