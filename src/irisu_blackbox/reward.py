from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irisu_blackbox.config import RewardConfig, ScoreOCRConfig
from irisu_blackbox.score_ocr import extract_score


@dataclass(slots=True)
class RewardTerms:
    survival: float = 0.0
    activity: float = 0.0
    cascade: float = 0.0
    stale: float = 0.0
    score_delta: float = 0.0

    @property
    def total(self) -> float:
        return self.survival + self.activity + self.cascade + self.stale + self.score_delta

    def as_dict(self) -> dict[str, float]:
        return {
            "survival": self.survival,
            "activity": self.activity,
            "cascade": self.cascade,
            "stale": self.stale,
            "score_delta": self.score_delta,
            "total": self.total,
        }


class RewardShaper:
    def __init__(self, cfg: RewardConfig, score_cfg: ScoreOCRConfig | None = None) -> None:
        self.cfg = cfg
        self.score_cfg = score_cfg or ScoreOCRConfig()
        self._prev_processed: np.ndarray | None = None
        self._stale_steps = 0
        self._prev_score: int | None = None

    def reset(
        self,
        processed_frame: np.ndarray,
        raw_frame_bgr: np.ndarray,
        observed_score: int | None = None,
    ) -> None:
        self._prev_processed = processed_frame.copy()
        self._stale_steps = 0
        self._prev_score = observed_score if observed_score is not None else self._read_score(raw_frame_bgr)

    def _read_score(self, raw_frame_bgr: np.ndarray) -> int | None:
        if not self.score_cfg.enabled or self.score_cfg.region is None:
            return None
        return extract_score(
            frame_bgr=raw_frame_bgr,
            region=self.score_cfg.region,
            tesseract_cmd=self.score_cfg.tesseract_cmd,
            method=self.score_cfg.method,
            template_dir=self.score_cfg.template_dir,
            template_min_similarity=self.score_cfg.template_min_similarity,
            template_fallback_to_tesseract=self.score_cfg.template_fallback_to_tesseract,
            template_expected_digits=self.score_cfg.template_expected_digits,
            template_inner_left=self.score_cfg.template_inner_left,
            template_inner_right=self.score_cfg.template_inner_right,
        )

    def step(
        self,
        processed_frame: np.ndarray,
        raw_frame_bgr: np.ndarray,
        observed_score: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        if self._prev_processed is None:
            self.reset(processed_frame, raw_frame_bgr, observed_score=observed_score)

        terms = RewardTerms(survival=self.cfg.survival_reward)

        assert self._prev_processed is not None
        diff = np.abs(processed_frame - self._prev_processed)
        activity = float(diff.mean())
        terms.activity = activity * self.cfg.activity_reward_scale

        if activity >= self.cfg.cascade_threshold:
            terms.cascade = self.cfg.cascade_bonus

        if activity < self.cfg.stale_threshold:
            self._stale_steps += 1
        else:
            self._stale_steps = 0

        if self._stale_steps >= self.cfg.stale_patience:
            terms.stale = self.cfg.stale_penalty

        current_score = observed_score if observed_score is not None else self._read_score(raw_frame_bgr)
        if current_score is not None and self._prev_score is not None:
            delta = max(0, current_score - self._prev_score)
            terms.score_delta = delta * self.cfg.score_delta_scale
        if current_score is not None:
            self._prev_score = current_score

        self._prev_processed = processed_frame.copy()
        return terms.total, terms.as_dict()
