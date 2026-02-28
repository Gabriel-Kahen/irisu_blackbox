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


ObsType = np.ndarray | dict[str, np.ndarray]


class IrisuBlackBoxEnv(gym.Env[ObsType, int]):
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
        self.reset_ready_detector = TemplateTerminationDetector(
            cfg.reset_ready_template,
            threshold=cfg.reset_ready_threshold,
        )

        self.action_space = spaces.Discrete(self.cfg.action_grid.action_count)
        self._use_hud_features = bool(self.cfg.hud_features.enabled)
        frame_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.cfg.frame_stack * 3, self.cfg.obs_height, self.cfg.obs_width),
            dtype=np.uint8,
        )
        if self._use_hud_features:
            self.observation_space = spaces.Dict(
                {
                    "image": frame_space,
                    "hud": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = frame_space

        self._step_count = 0
        self._last_frame_bgr: np.ndarray | None = None
        self._health_missing_streak = 0
        self._last_click_time = 0.0
        self._pending_post_game_over_delay = False
        self._action_pause_until = 0.0

    def _normalize_score(self, score: int | None) -> float:
        if score is None or score <= 0:
            return 0.0
        max_score = max(1.0, float(self.cfg.hud_features.score_log_max))
        denom = float(np.log1p(max_score))
        if denom <= 0:
            return 0.0
        return float(min(1.0, np.log1p(float(score)) / denom))

    def _build_observation(self, image_obs: np.ndarray, hud) -> ObsType:
        if not self._use_hud_features:
            return image_obs
        hud_vector = np.array(
            [
                float(hud.health_percent) if hud.health_visible is True and hud.health_percent is not None else 0.0,
                self._normalize_score(hud.score),
                1.0 if hud.health_visible is True else 0.0,
                1.0 if hud.score is not None else 0.0,
            ],
            dtype=np.float32,
        )
        return {
            "image": image_obs,
            "hud": hud_vector,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        _ = options

        self._run_post_game_over_delay_if_needed()
        self._run_game_over_macro_if_needed()
        self._wait_for_reset_ready()
        self.backend.reset()
        self.hud_reader.reset()
        raw, hud = self._wait_for_round_start()
        reward_frame = self.frame_processor.preprocess(raw)
        image_frame = self.frame_processor.preprocess_observation(raw)
        image_obs = self.frame_processor.reset(image_frame)
        obs = self._build_observation(image_obs, hud)
        self.reward_shaper.reset(
            reward_frame,
            raw,
            observed_score=hud.score,
            observed_health_percent=hud.health_percent,
            health_visible=hud.health_visible,
        )

        self._step_count = 0
        self._last_frame_bgr = raw
        self._health_missing_streak = 0
        self._last_click_time = 0.0
        self._action_pause_until = 0.0

        info = {"step": self._step_count, "hud": hud.as_dict()}
        return obs, info

    def _arm_action_pause_on_health_missing(self) -> None:
        pause_s = max(0.0, float(self.cfg.action_pause_on_health_missing_s))
        if pause_s <= 0:
            return
        now = time.monotonic()
        self._action_pause_until = max(self._action_pause_until, now + pause_s)

    def _is_action_paused(self) -> bool:
        return time.monotonic() < self._action_pause_until

    def _run_post_game_over_delay_if_needed(self) -> None:
        if not self._pending_post_game_over_delay:
            return
        delay_s = max(0.0, float(self.cfg.post_game_over_delay_s))
        self._pending_post_game_over_delay = False
        if delay_s > 0:
            time.sleep(delay_s)

    def _wait_for_reset_ready(self) -> None:
        if self.cfg.reset_ready_template is None:
            return

        deadline = time.monotonic() + max(0.0, self.cfg.reset_ready_timeout_s)
        poll_s = max(0.0, self.cfg.reset_ready_poll_s)

        while True:
            frame = self.backend.capture_frame()
            if self.reset_ready_detector.matches(frame):
                return
            if time.monotonic() >= deadline:
                return
            if poll_s > 0:
                time.sleep(poll_s)

    def _run_game_over_macro_if_needed(self) -> None:
        if not self.cfg.game_over_macro:
            return

        frame = self.backend.capture_frame()
        if self.cfg.reset_ready_template and self.reset_ready_detector.matches(frame):
            return

        self.backend.run_macro(self.cfg.game_over_macro)

    def _wait_for_round_start(self) -> tuple[np.ndarray, Any]:
        raw = self.backend.capture_frame()
        hud = self.hud_reader.read(raw)

        if not self.cfg.health_bar.enabled:
            return raw, hud
        if hud.health_visible is True:
            return raw, hud

        deadline = time.monotonic() + max(0.0, self.cfg.round_start_timeout_s)
        poll_s = max(0.0, self.cfg.round_start_poll_s)

        while time.monotonic() < deadline:
            if poll_s > 0:
                time.sleep(poll_s)
            raw = self.backend.capture_frame()
            hud = self.hud_reader.read(raw)
            if hud.health_visible is True:
                return raw, hud

        return raw, hud

    def step(self, action: int):
        cmd = decode_action(int(action), self.cfg.action_grid)
        repeats = max(1, self.cfg.episode.action_repeat)
        action_suppressed = self._is_action_paused()

        for _ in range(repeats):
            if cmd is not None and not action_suppressed:
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
        reward_frame = self.frame_processor.preprocess(raw)
        image_frame = self.frame_processor.preprocess_observation(raw)
        image_obs = self.frame_processor.push(image_frame)
        obs = self._build_observation(image_obs, hud)

        reward, reward_terms = self.reward_shaper.step(
            reward_frame,
            raw,
            observed_score=hud.score,
            observed_health_percent=hud.health_percent,
            health_visible=hud.health_visible,
        )
        self._step_count += 1

        backend_done = bool(getattr(self.backend, "game_over", False))
        template_done = self.termination_detector.matches(raw)

        health_done = False
        if self.cfg.game_over_on_health_missing and self.cfg.health_bar.enabled:
            if hud.health_visible is False:
                self._arm_action_pause_on_health_missing()
                self._health_missing_streak += 1
            else:
                self._health_missing_streak = 0
                self._action_pause_until = 0.0
            health_done = self._health_missing_streak >= max(1, self.cfg.health_missing_patience)

        terminated = backend_done or template_done or health_done
        truncated = self._step_count >= self.cfg.episode.max_steps
        if terminated:
            self._pending_post_game_over_delay = True

        self._last_frame_bgr = raw
        info = {
            "step": self._step_count,
            "reward_terms": reward_terms,
            "hud": hud.as_dict(),
            "backend_done": backend_done,
            "template_done": template_done,
            "health_done": health_done,
            "health_missing_streak": self._health_missing_streak,
            "action_suppressed": action_suppressed,
            "action_pause_remaining_s": max(0.0, self._action_pause_until - time.monotonic()),
            "termination_reason": (
                "backend"
                if backend_done
                else "template"
                if template_done
                else "health_missing"
                if health_done
                else None
            ),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._last_frame_bgr is None:
            return None
        # Convert BGR -> RGB for viewer compatibility.
        return self._last_frame_bgr[:, :, ::-1]

    def close(self) -> None:
        self.backend.close()
