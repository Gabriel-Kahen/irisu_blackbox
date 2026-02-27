from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from irisu_blackbox.config import HealthBarConfig, Rect, ScoreOCRConfig
from irisu_blackbox.score_ocr import extract_score_reading


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
        self._committed_score: int | None = None
        self._health_smoothing_window = max(1, int(self.health_cfg.smoothing_window))
        self._health_history: deque[float] = deque(maxlen=self._health_smoothing_window)

    def reset(self) -> None:
        self._committed_score = None
        self._health_history.clear()

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
        reading = extract_score_reading(
            frame_bgr=frame_bgr,
            region=self.score_cfg.region,
            tesseract_cmd=self.score_cfg.tesseract_cmd,
        )
        if reading is None:
            return None
        if reading.confidence >= 0 and reading.confidence < self.score_cfg.min_confidence:
            return None
        return reading.score

    def _read_health(self, frame_bgr: np.ndarray) -> tuple[float | None, bool | None]:
        if not self.health_cfg.enabled:
            return None, None

        if self.health_cfg.method.lower() == "scanline":
            return self._read_health_scanline(frame_bgr)
        return self._read_health_profile(frame_bgr)

    def _read_health_profile(self, frame_bgr: np.ndarray) -> tuple[float | None, bool | None]:
        if self.health_cfg.region is None:
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

        # Estimate fill per column from a baseline-aware red strength profile.
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        red_strength = (0.7 * saturation) + (0.3 * value)
        red_strength = red_strength * (mask.astype(np.float32) / 255.0)
        col_strength = red_strength.mean(axis=0)
        if col_strength.size > 4:
            col_strength = cv2.GaussianBlur(col_strength[None, :], (1, 0), 1.2)[0]

        baseline = float(np.percentile(col_strength, 20))
        peak_strength = float(np.percentile(col_strength, 95))
        if peak_strength <= baseline:
            return 0.0, True

        adaptive_threshold = baseline + (
            (peak_strength - baseline) * self.health_cfg.adaptive_fill_peak_ratio
        )
        adaptive_threshold = max(self.health_cfg.column_fill_threshold, adaptive_threshold)
        col_mask = (col_strength >= adaptive_threshold).astype(np.uint8)
        if col_mask.size == 0:
            percent = 0.0
            if self.health_cfg.invert_percent:
                percent = 1.0 - percent
            return percent, True

        # Close tiny holes in the per-column signal.
        closed = cv2.morphologyEx(
            (col_mask[None, :] * 255),
            cv2.MORPH_CLOSE,
            np.ones((1, 5), dtype=np.uint8),
        )
        col_mask = (closed[0] > 0).astype(np.uint8)

        spans = _find_true_spans(col_mask)
        if not spans:
            percent = 0.0
            if self.health_cfg.invert_percent:
                percent = 1.0 - percent
            return percent, True

        # Use edge-consistent span based on configured fill direction.
        if self.health_cfg.fill_direction.lower() == "right_to_left":
            start, end = max(spans, key=lambda item: item[1])
        else:
            start, end = min(spans, key=lambda item: item[0])
        width = mask.shape[1]
        direction = self.health_cfg.fill_direction.lower()
        if direction == "right_to_left":
            percent = float((width - start) / width)
        else:
            percent = float((end + 1) / width)

        percent = max(0.0, min(1.0, percent))
        if self.health_cfg.invert_percent:
            percent = 1.0 - percent
            percent = max(0.0, min(1.0, percent))
        return percent, True

    def _read_health_scanline(self, frame_bgr: np.ndarray) -> tuple[float | None, bool | None]:
        if (
            self.health_cfg.scanline_start_x is None
            or self.health_cfg.scanline_end_x is None
            or self.health_cfg.scanline_y is None
        ):
            return None, None

        h, w = frame_bgr.shape[:2]
        x0 = int(max(0, min(w - 1, min(self.health_cfg.scanline_start_x, self.health_cfg.scanline_end_x))))
        x1 = int(max(0, min(w - 1, max(self.health_cfg.scanline_start_x, self.health_cfg.scanline_end_x))))
        y = int(max(0, min(h - 1, self.health_cfg.scanline_y)))
        half_h = max(0, int(self.health_cfg.scanline_half_height))

        if x1 <= x0:
            return 0.0, False

        y0 = max(0, y - half_h)
        y1 = min(h, y + half_h + 1)
        band = frame_bgr[y0:y1, x0 : x1 + 1]
        if band.size == 0:
            return 0.0, False

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
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

        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        red_strength = ((0.75 * saturation) + (0.25 * value)) * (mask.astype(np.float32) / 255.0)
        col_strength = red_strength.mean(axis=0)
        if col_strength.size > 4:
            col_strength = cv2.GaussianBlur(col_strength[None, :], (1, 0), 1.0)[0]

        if col_strength.size <= 1:
            return 0.0, True

        grad = np.diff(col_strength)
        abs_grad = np.abs(grad)
        boundary = int(np.argmax(abs_grad))
        contrast = float(abs_grad[boundary])

        direction = self.health_cfg.fill_direction.lower()
        length = col_strength.size

        if contrast < self.health_cfg.scanline_contrast_threshold:
            # Fallback to profile-style estimation when the edge is weak.
            baseline = float(np.percentile(col_strength, 20))
            peak_strength = float(np.percentile(col_strength, 95))
            if peak_strength <= baseline:
                return 0.0, True
            threshold = baseline + (
                (peak_strength - baseline) * self.health_cfg.adaptive_fill_peak_ratio
            )
            threshold = max(self.health_cfg.column_fill_threshold, threshold)
            col_mask = (col_strength >= threshold).astype(np.uint8)
            spans = _find_true_spans(col_mask)
            if not spans:
                return 0.0, True
            if direction == "right_to_left":
                start, _ = max(spans, key=lambda item: item[1])
                percent = float((length - start) / length)
            else:
                _, end = min(spans, key=lambda item: item[0])
                percent = float((end + 1) / length)
        else:
            if direction == "right_to_left":
                percent = float((length - (boundary + 1)) / length)
            else:
                percent = float((boundary + 1) / length)

        percent = max(0.0, min(1.0, percent))
        if self.health_cfg.invert_percent:
            percent = 1.0 - percent
            percent = max(0.0, min(1.0, percent))
        return percent, True

    def read(self, frame_bgr: np.ndarray) -> HUDState:
        raw_score = self._read_score(frame_bgr)
        score = self._stabilize_score(raw_score)
        health_percent, health_visible = self._read_health(frame_bgr)
        health_percent = self._smooth_health_percent(health_percent, health_visible)
        return HUDState(
            score=score,
            health_percent=health_percent,
            health_visible=health_visible,
        )

    def _stabilize_score(self, raw_score: int | None) -> int | None:
        if raw_score is None:
            if self.score_cfg.hold_last_value_when_missing:
                return self._committed_score
            return None

        candidate = int(raw_score)

        if self._committed_score is None:
            self._committed_score = candidate
            return self._committed_score

        max_step = int(self.score_cfg.max_step_increase)
        if max_step > 0 and candidate > (self._committed_score + max_step):
            # Likely OCR spike; ignore this step and keep current score.
            return self._committed_score

        if self.score_cfg.monotonic_non_decreasing:
            self._committed_score = max(self._committed_score, candidate)
        else:
            self._committed_score = candidate
        return self._committed_score

    def _smooth_health_percent(
        self,
        percent: float | None,
        visible: bool | None,
    ) -> float | None:
        if percent is None:
            self._health_history.clear()
            return None

        if visible is not True:
            self._health_history.clear()
            return percent

        self._health_history.append(float(percent))
        if self._health_smoothing_window <= 1:
            return float(percent)

        values = np.asarray(self._health_history, dtype=np.float32)
        return float(np.median(values))


def _find_true_spans(col_mask: np.ndarray) -> list[tuple[int, int]]:
    if col_mask.size == 0:
        return []
    padded = np.concatenate(([0], col_mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]
