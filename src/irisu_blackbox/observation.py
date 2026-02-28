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

    def _resize_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized_rgb = self._resize_rgb(frame_bgr)
        # Reward shaping uses normalized float tensors.
        return np.transpose(resized_rgb.astype(np.float32) / 255.0, (2, 0, 1)).copy()

    def preprocess_observation(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized_rgb = self._resize_rgb(frame_bgr)
        # SB3 image extractors are most reliable with uint8 channel-first tensors.
        return np.transpose(resized_rgb, (2, 0, 1)).copy()

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
        stacked = np.concatenate(list(self._frames), axis=0)
        return stacked
