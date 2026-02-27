from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from irisu_blackbox.action_grid import decode_action
from irisu_blackbox.backends.base import GameBackend
from irisu_blackbox.config import EnvConfig
from irisu_blackbox.hud import HUDReader
from irisu_blackbox.observation import FrameProcessor
from irisu_blackbox.reward import RewardShaper
from irisu_blackbox.termination import TemplateTerminationDetector


class IrisuBlackBoxEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: EnvConfig, backend: GameBackend) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.frame_processor = FrameProcessor(cfg.obs_width, cfg.obs_height, cfg.frame_stack)
        self.reward_shaper = RewardShaper(cfg.reward, cfg.score_ocr)
        self.hud_reader = HUDReader(cfg.score_ocr, cfg.health_bar)
        self.termination_detector = TemplateTerminationDetector(
            cfg.game_over_template,
            threshold=cfg.game_over_threshold,
        )

        self.action_space = spaces.Discrete(self.cfg.action_grid.action_count)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.cfg.frame_stack, self.cfg.obs_height, self.cfg.obs_width),
            dtype=np.float32,
        )

        self._step_count = 0
        self._last_frame_bgr: np.ndarray | None = None
        self._health_missing_streak = 0
        self._last_click_time = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        _ = options

        self.backend.reset()
        raw = self.backend.capture_frame()
        hud = self.hud_reader.read(raw)
        processed = self.frame_processor.preprocess(raw)
        obs = self.frame_processor.reset(processed)
        self.reward_shaper.reset(processed, raw, observed_score=hud.score)

        self._step_count = 0
        self._last_frame_bgr = raw
        self._health_missing_streak = 0
        self._last_click_time = 0.0

        info = {"step": self._step_count, "hud": hud.as_dict()}
        return obs, info

    def step(self, action: int):
        cmd = decode_action(int(action), self.cfg.action_grid)
        repeats = max(1, self.cfg.episode.action_repeat)

        for _ in range(repeats):
            if cmd is not None:
                max_cps = self.cfg.episode.max_clicks_per_second
                if max_cps > 0:
                    min_interval = 1.0 / max_cps
                    now = time.monotonic()
                    wait_s = (self._last_click_time + min_interval) - now
                    if wait_s > 0:
                        time.sleep(wait_s)
                self.backend.click(
                    x=cmd.x,
                    y=cmd.y,
                    button=cmd.button,
                    hold_s=self.cfg.episode.click_hold_s,
                )
                self._last_click_time = time.monotonic()
            if self.cfg.episode.inter_step_sleep_s > 0:
                time.sleep(self.cfg.episode.inter_step_sleep_s)

        raw = self.backend.capture_frame()
        hud = self.hud_reader.read(raw)
        processed = self.frame_processor.preprocess(raw)
        obs = self.frame_processor.push(processed)

        reward, reward_terms = self.reward_shaper.step(processed, raw, observed_score=hud.score)
        self._step_count += 1

        backend_done = bool(getattr(self.backend, "game_over", False))
        template_done = self.termination_detector.is_game_over(raw)

        health_done = False
        if self.cfg.game_over_on_health_missing and self.cfg.health_bar.enabled:
            if hud.health_visible is False:
                self._health_missing_streak += 1
            else:
                self._health_missing_streak = 0
            health_done = self._health_missing_streak >= max(1, self.cfg.health_missing_patience)

        terminated = backend_done or template_done or health_done
        truncated = self._step_count >= self.cfg.episode.max_steps

        self._last_frame_bgr = raw
        info = {
            "step": self._step_count,
            "reward_terms": reward_terms,
            "hud": hud.as_dict(),
            "backend_done": backend_done,
            "template_done": template_done,
            "health_done": health_done,
            "health_missing_streak": self._health_missing_streak,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._last_frame_bgr is None:
            return None
        # Convert BGR -> RGB for viewer compatibility.
        return self._last_frame_bgr[:, :, ::-1]

    def close(self) -> None:
        self.backend.close()
