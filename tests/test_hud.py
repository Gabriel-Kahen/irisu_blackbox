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


def test_health_uses_bright_fill_not_dark_background():
    frame = np.zeros((80, 140, 3), dtype=np.uint8)
    # Dark red baseline across full bar.
    frame[20:40, 20:120] = (0, 0, 80)
    # Bright red fill on the right-most 20%.
    frame[20:40, 100:120] = (0, 0, 230)

    health_cfg = HealthBarConfig(
        enabled=True,
        region=Rect(left=20, top=20, width=100, height=20),
        min_visible_pixels=10,
        column_fill_threshold=0.08,
        adaptive_fill_peak_ratio=0.55,
    )
    reader = HUDReader(ScoreOCRConfig(enabled=False), health_cfg)

    hud = reader.read(frame)
    assert hud.health_visible is True
    assert hud.health_percent is not None
    assert 0.15 <= hud.health_percent <= 0.25
