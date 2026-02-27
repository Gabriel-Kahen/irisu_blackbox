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
