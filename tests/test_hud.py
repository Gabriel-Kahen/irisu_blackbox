import numpy as np

from irisu_blackbox.config import HealthBarConfig, Rect, ScoreOCRConfig
from irisu_blackbox.hud import HUDReader


def test_health_percent_estimate_from_red_bar():
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    frame[70:90, 20:120] = (0, 0, 255)

    health_cfg = HealthBarConfig(
        enabled=True,
        region=Rect(left=20, top=70, width=180, height=20),
        min_visible_pixels=10,
        column_fill_threshold=0.05,
    )
    reader = HUDReader(ScoreOCRConfig(enabled=False), health_cfg)

    hud = reader.read(frame)
    assert hud.health_visible is True
    assert hud.health_percent is not None
    assert 0.5 <= hud.health_percent <= 0.6


def test_health_disappears_when_bar_missing():
    frame = np.zeros((120, 220, 3), dtype=np.uint8)

    health_cfg = HealthBarConfig(
        enabled=True,
        region=Rect(left=20, top=70, width=180, height=20),
        min_visible_pixels=10,
    )
    reader = HUDReader(ScoreOCRConfig(enabled=False), health_cfg)

    hud = reader.read(frame)
    assert hud.health_visible is False
    assert hud.health_percent == 0.0


def test_health_percent_can_be_inverted():
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    frame[70:90, 20:120] = (0, 0, 255)

    health_cfg = HealthBarConfig(
        enabled=True,
        region=Rect(left=20, top=70, width=180, height=20),
        min_visible_pixels=10,
        column_fill_threshold=0.05,
        invert_percent=True,
    )
    reader = HUDReader(ScoreOCRConfig(enabled=False), health_cfg)

    hud = reader.read(frame)
    assert hud.health_visible is True
    assert hud.health_percent is not None
    assert 0.4 <= hud.health_percent <= 0.5
