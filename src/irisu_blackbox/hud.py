from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from irisu_blackbox.config import HealthBarConfig, Rect, ScoreOCRConfig
from irisu_blackbox.score_ocr import extract_score


@dataclass(slots=True)
class HUDState:
    score: int | None
    health_percent: float | None
    health_visible: bool | None

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "score": self.score,
            "health_percent": self.health_percent,
            "health_visible": self.health_visible,
        }


class HUDReader:
    def __init__(self, score_cfg: ScoreOCRConfig, health_cfg: HealthBarConfig) -> None:
        self.score_cfg = score_cfg
        self.health_cfg = health_cfg

    @staticmethod
    def _crop(frame_bgr: np.ndarray, region: Rect) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        left = max(0, region.left)
        top = max(0, region.top)
        right = min(w, region.right)
        bottom = min(h, region.bottom)
        if right <= left or bottom <= top:
            return None
        return frame_bgr[top:bottom, left:right]

    def _read_score(self, frame_bgr: np.ndarray) -> int | None:
        if not self.score_cfg.enabled or self.score_cfg.region is None:
            return None
        return extract_score(
            frame_bgr=frame_bgr,
            region=self.score_cfg.region,
            tesseract_cmd=self.score_cfg.tesseract_cmd,
        )

    def _read_health(self, frame_bgr: np.ndarray) -> tuple[float | None, bool | None]:
        if not self.health_cfg.enabled or self.health_cfg.region is None:
            return None, None

        crop = self._crop(frame_bgr, self.health_cfg.region)
        if crop is None or crop.size == 0:
            return None, False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_1 = np.array(self.health_cfg.hsv_lower_1, dtype=np.uint8)
        upper_1 = np.array(self.health_cfg.hsv_upper_1, dtype=np.uint8)
        lower_2 = np.array(self.health_cfg.hsv_lower_2, dtype=np.uint8)
        upper_2 = np.array(self.health_cfg.hsv_upper_2, dtype=np.uint8)

        mask_1 = cv2.inRange(hsv, lower_1, upper_1)
        mask_2 = cv2.inRange(hsv, lower_2, upper_2)
        mask = cv2.bitwise_or(mask_1, mask_2)

        visible_pixels = int(np.count_nonzero(mask))
        visible = visible_pixels >= self.health_cfg.min_visible_pixels
        if not visible:
            return 0.0, False

        col_fill = (mask.astype(np.float32) / 255.0).mean(axis=0)
        filled_cols = np.flatnonzero(col_fill >= self.health_cfg.column_fill_threshold)
        if filled_cols.size == 0:
            return 0.0, True

        percent = float((filled_cols.max() + 1) / mask.shape[1])
        percent = max(0.0, min(1.0, percent))
        return percent, True

    def read(self, frame_bgr: np.ndarray) -> HUDState:
        score = self._read_score(frame_bgr)
        health_percent, health_visible = self._read_health(frame_bgr)
        return HUDState(
            score=score,
            health_percent=health_percent,
            health_visible=health_visible,
        )
