"""
Agent base class — observe, act, learn interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """Base class for RL agents."""

    @abstractmethod
    def observe(self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict[str, Any]) -> None:
        """Process observation and optional learning step."""
        pass

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action given observation. Returns action index."""
        pass

    @abstractmethod
    def learn(self) -> dict[str, float] | None:
        """Perform learning step. Returns metrics dict or None."""
        pass

    def reset(self) -> None:
        """Called at episode start. Override if needed."""
        pass
