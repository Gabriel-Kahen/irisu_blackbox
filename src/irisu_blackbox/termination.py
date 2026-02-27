from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class TemplateTerminationDetector:
    def __init__(self, template_path: str | None, threshold: float = 0.9) -> None:
        self.threshold = threshold
        self.template: np.ndarray | None = None
        if template_path:
            path = Path(template_path).expanduser().resolve()
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise RuntimeError(f"Could not read game-over template image: {path}")
            self.template = template

    def matches(self, frame_bgr: np.ndarray) -> bool:
        if self.template is None:
            return False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = self.template.shape[:2]
        if gray.shape[0] < h or gray.shape[1] < w:
            return False

        result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, _ = cv2.minMaxLoc(result)
        return bool(max_value >= self.threshold)

    def is_game_over(self, frame_bgr: np.ndarray) -> bool:
        return self.matches(frame_bgr)
