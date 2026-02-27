from __future__ import annotations

import math
from dataclasses import dataclass
import time

import cv2
import numpy as np

from irisu_blackbox.config import ResetMacroStep
from irisu_blackbox.backends.base import GameBackend


@dataclass(slots=True)
class MockBackendConfig:
    width: int = 640
    height: int = 480
    seed: int = 0


class MockGameBackend(GameBackend):
    """Deterministic mock backend for local pipeline checks without the real game."""

    def __init__(self, cfg: MockBackendConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.frame_idx = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self._target = np.array([cfg.width // 2, cfg.height // 2], dtype=np.float32)
        self._velocity = np.array([2.0, 1.0], dtype=np.float32)
        self._pulse = 0.0

    def _step_sim(self) -> None:
        self.frame_idx += 1
        self._pulse += 0.15

        noise = self.rng.normal(0.0, 0.3, size=2).astype(np.float32)
        self._velocity += noise
        self._velocity = np.clip(self._velocity, -4.0, 4.0)
        self._target += self._velocity

        if self._target[0] < 24 or self._target[0] > (self.cfg.width - 24):
            self._velocity[0] *= -1
        if self._target[1] < 24 or self._target[1] > (self.cfg.height - 24):
            self._velocity[1] *= -1

        self._target[0] = float(np.clip(self._target[0], 24, self.cfg.width - 24))
        self._target[1] = float(np.clip(self._target[1], 24, self.cfg.height - 24))

        if self.misses >= 25:
            self.game_over = True

    def _draw(self) -> np.ndarray:
        frame = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)

        for x in range(0, self.cfg.width, 40):
            cv2.line(frame, (x, 0), (x, self.cfg.height), (22, 22, 22), 1)
        for y in range(0, self.cfg.height, 40):
            cv2.line(frame, (0, y), (self.cfg.width, y), (22, 22, 22), 1)

        radius = int(18 + (4 * (1 + math.sin(self._pulse))))
        center = (int(self._target[0]), int(self._target[1]))
        cv2.circle(frame, center, radius, (40, 220, 220), -1)
        cv2.circle(frame, center, radius + 2, (0, 120, 240), 2)

        cv2.putText(
            frame,
            f"score:{self.score}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"misses:{self.misses}",
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return frame

    def capture_frame(self) -> np.ndarray:
        self._step_sim()
        return self._draw()

    def click(self, x: int, y: int, button: str = "left", hold_s: float = 0.01) -> None:
        if self.game_over:
            return

        cursor = np.array([x, y], dtype=np.float32)
        dist = float(np.linalg.norm(cursor - self._target))

        if dist <= 36:
            base = 10 if button == "left" else 14
            self.score += base
            self.misses = max(0, self.misses - 1)
        else:
            self.misses += 1

    def reset(self) -> None:
        self.frame_idx = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self._target = np.array([self.cfg.width // 2, self.cfg.height // 2], dtype=np.float32)
        self._velocity = np.array([2.0, 1.0], dtype=np.float32)
        self._pulse = 0.0

    def run_macro(self, steps: list[ResetMacroStep]) -> None:
        for step in steps:
            kind = step.kind.lower()
            if kind == "sleep":
                if step.duration_s > 0:
                    time.sleep(step.duration_s)
                continue
            if kind == "click":
                if step.x is not None and step.y is not None:
                    self.click(step.x, step.y, button=step.button, hold_s=step.duration_s)
                continue
            if kind == "key":
                continue

    def close(self) -> None:
        return
