from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class FrameProcessor:
    def __init__(self, width: int, height: int, stack: int) -> None:
        self.width = width
        self.height = height
        self.stack = stack
        self._frames: deque[np.ndarray] = deque(maxlen=stack)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0).copy()

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        for _ in range(self.stack):
            self._frames.append(initial_frame)
        return self.observation

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._frames.append(frame)
        return self.observation

    @property
    def observation(self) -> np.ndarray:
        if len(self._frames) < self.stack:
            raise RuntimeError("Frame stack not initialized")
        stacked = np.stack(list(self._frames), axis=0)
        return stacked.astype(np.float32, copy=False)
