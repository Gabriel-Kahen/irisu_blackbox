from pathlib import Path

import cv2
import numpy as np

from irisu_blackbox.backends.base import GameBackend
from irisu_blackbox.backends.mock import MockBackendConfig, MockGameBackend
from irisu_blackbox.config import ActionGridConfig, EnvConfig, EpisodeConfig, HealthBarConfig, Rect
from irisu_blackbox.env import IrisuBlackBoxEnv


def test_env_terminates_when_health_bar_missing_for_patience():
    cfg = EnvConfig(
        backend="mock",
        obs_width=64,
        obs_height=64,
        frame_stack=2,
        action_grid=ActionGridConfig(rows=4, cols=4, left=0, top=0, right=64, bottom=64),
        episode=EpisodeConfig(max_steps=100, action_repeat=1),
        health_bar=HealthBarConfig(
            enabled=True,
            region=Rect(left=0, top=0, width=10, height=10),
            min_visible_pixels=1,
        ),
        game_over_on_health_missing=True,
        health_missing_patience=2,
    )

    backend = MockGameBackend(MockBackendConfig(width=64, height=64, seed=0))
    env = IrisuBlackBoxEnv(cfg=cfg, backend=backend)
    try:
        env.reset()
        _, _, terminated_1, _, info_1 = env.step(0)
        assert terminated_1 is False
        assert info_1["health_done"] is False

        _, _, terminated_2, _, info_2 = env.step(0)
        assert terminated_2 is True
        assert info_2["health_done"] is True
        assert info_2["termination_reason"] == "health_missing"
    finally:
        env.close()


class ScriptedResetBackend(GameBackend):
    def __init__(self, pre_reset_frames: list[np.ndarray], post_reset_frames: list[np.ndarray]) -> None:
        self.pre_reset_frames = pre_reset_frames
        self.post_reset_frames = post_reset_frames
        self.phase = "pre"
        self.pre_capture_count = 0
        self.post_capture_count = 0
        self.reset_calls = 0

    def capture_frame(self) -> np.ndarray:
        if self.phase == "pre":
            idx = min(self.pre_capture_count, len(self.pre_reset_frames) - 1)
            self.pre_capture_count += 1
            return self.pre_reset_frames[idx].copy()

        idx = min(self.post_capture_count, len(self.post_reset_frames) - 1)
        self.post_capture_count += 1
        return self.post_reset_frames[idx].copy()

    def click(self, x: int, y: int, button: str = "left", hold_s: float = 0.01) -> None:
        return

    def reset(self) -> None:
        self.reset_calls += 1
        self.phase = "post"

    def close(self) -> None:
        return


def _menu_frame(show_marker: bool) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    if show_marker:
        frame[8:20, 8:20] = (40, 40, 40)
        frame[10:18, 10:18] = (255, 255, 255)
        frame[12:16, 12:16] = (80, 80, 80)
    return frame


def _gameplay_frame() -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[0:10, 0:10] = (0, 0, 255)
    return frame


def test_env_reset_waits_for_menu_then_round_start(tmp_path: Path):
    template = np.full((12, 12), 40, dtype=np.uint8)
    template[2:10, 2:10] = 255
    template[4:8, 4:8] = 80
    template_path = tmp_path / "menu_template.png"
    ok = cv2.imwrite(str(template_path), template)
    assert ok

    cfg = EnvConfig(
        backend="windows",
        obs_width=64,
        obs_height=64,
        frame_stack=2,
        action_grid=ActionGridConfig(rows=4, cols=4, left=0, top=0, right=64, bottom=64),
        episode=EpisodeConfig(max_steps=100, action_repeat=1),
        health_bar=HealthBarConfig(
            enabled=True,
            region=Rect(left=0, top=0, width=10, height=10),
            min_visible_pixels=1,
            column_fill_threshold=0.05,
            smoothing_window=1,
        ),
        reset_ready_template=str(template_path),
        reset_ready_threshold=0.95,
        reset_ready_timeout_s=1.0,
        reset_ready_poll_s=0.0,
        round_start_timeout_s=1.0,
        round_start_poll_s=0.0,
    )

    backend = ScriptedResetBackend(
        pre_reset_frames=[_menu_frame(False), _menu_frame(False), _menu_frame(True)],
        post_reset_frames=[_menu_frame(False), _gameplay_frame()],
    )
    env = IrisuBlackBoxEnv(cfg=cfg, backend=backend)
    try:
        _, info = env.reset()
        assert backend.reset_calls == 1
        assert backend.pre_capture_count >= 3
        assert backend.post_capture_count >= 2
        assert info["hud"]["health_visible"] is True
    finally:
        env.close()


def test_env_reset_applies_post_game_over_delay(monkeypatch):
    cfg = EnvConfig(
        backend="mock",
        obs_width=64,
        obs_height=64,
        frame_stack=2,
        action_grid=ActionGridConfig(rows=4, cols=4, left=0, top=0, right=64, bottom=64),
        episode=EpisodeConfig(max_steps=10, action_repeat=1),
        post_game_over_delay_s=1.0,
    )

    backend = MockGameBackend(MockBackendConfig(width=64, height=64, seed=0))
    env = IrisuBlackBoxEnv(cfg=cfg, backend=backend)
    sleeps: list[float] = []

    def _fake_sleep(value: float) -> None:
        sleeps.append(value)

    monkeypatch.setattr("irisu_blackbox.env.time.sleep", _fake_sleep)

    try:
        env._pending_post_game_over_delay = True
        env.reset()
        assert 1.0 in sleeps
    finally:
        env.close()
