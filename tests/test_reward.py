import numpy as np

from irisu_blackbox.config import RewardConfig
from irisu_blackbox.reward import RewardShaper


def test_reward_survival_only_when_static():
    cfg = RewardConfig(
        survival_reward=0.1,
        activity_reward_scale=1.0,
        cascade_threshold=0.5,
        stale_threshold=0.01,
        stale_patience=3,
        stale_penalty=-1.0,
    )
    shaper = RewardShaper(cfg)

    frame = np.zeros((8, 8), dtype=np.float32)
    raw = np.zeros((8, 8, 3), dtype=np.uint8)

    shaper.reset(frame, raw)
    reward, terms = shaper.step(frame, raw)

    assert abs(reward - 0.1) < 1e-6
    assert terms["activity"] == 0.0


def test_reward_cascade_bonus_on_large_change():
    cfg = RewardConfig(
        survival_reward=0.0,
        activity_reward_scale=0.0,
        cascade_threshold=0.2,
        cascade_bonus=0.5,
    )
    shaper = RewardShaper(cfg)

    frame_a = np.zeros((8, 8), dtype=np.float32)
    frame_b = np.ones((8, 8), dtype=np.float32)
    raw = np.zeros((8, 8, 3), dtype=np.uint8)

    shaper.reset(frame_a, raw)
    reward, terms = shaper.step(frame_b, raw)

    assert abs(reward - 0.5) < 1e-6
    assert terms["cascade"] == 0.5


def test_reward_uses_score_and_health_signals():
    cfg = RewardConfig(
        survival_reward=0.0,
        activity_reward_scale=0.0,
        score_delta_scale=0.01,
        score_value_scale=0.2,
        score_value_log_max=1000.0,
        health_value_scale=0.3,
        health_delta_scale=2.0,
    )
    shaper = RewardShaper(cfg)

    frame = np.zeros((8, 8), dtype=np.float32)
    raw = np.zeros((8, 8, 3), dtype=np.uint8)

    shaper.reset(
        frame,
        raw,
        observed_score=100,
        observed_health_percent=0.8,
        health_visible=True,
    )
    reward, terms = shaper.step(
        frame,
        raw,
        observed_score=150,
        observed_health_percent=0.6,
        health_visible=True,
    )

    assert terms["score_delta"] == 0.5
    assert terms["score_value"] > 0.0
    assert np.isclose(terms["health_value"], 0.18)
    assert np.isclose(terms["health_delta"], -0.4)
    assert abs(reward - terms["total"]) < 1e-6
