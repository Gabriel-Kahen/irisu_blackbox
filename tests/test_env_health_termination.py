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
    finally:
        env.close()
