from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GameBackend(ABC):
    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """Return current frame as BGR uint8 image."""

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left", hold_s: float = 0.01) -> None:
        """Perform a mouse click."""

    @abstractmethod
    def reset(self) -> None:
        """Reset game to a new episode using scripted UI actions."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
