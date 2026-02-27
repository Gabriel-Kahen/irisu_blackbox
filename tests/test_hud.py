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
    # Dark unfilled baseline across full bar.
    frame[20:40, 20:120] = (0, 0, 45)
    # Filled portion (left 80%) uses dim+bright red mix like in-game gradients.
    frame[20:40, 20:100] = (0, 0, 120)
    frame[20:40, 70:100] = (0, 0, 230)

    health_cfg = HealthBarConfig(
        enabled=True,
        region=Rect(left=20, top=20, width=100, height=20),
        min_visible_pixels=10,
        column_fill_threshold=0.08,
        adaptive_fill_peak_ratio=0.45,
    )
    reader = HUDReader(ScoreOCRConfig(enabled=False), health_cfg)

    hud = reader.read(frame)
    assert hud.health_visible is True
    assert hud.health_percent is not None
    assert 0.75 <= hud.health_percent <= 0.85


def test_scanline_boundary_estimates_right_to_left_fill():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    y = 50
    x0, x1 = 20, 179
    boundary = 60

    # Left side is unfilled (darker red), right side is filled (brighter red).
    frame[y - 1 : y + 2, x0 : boundary + 1] = (0, 0, 50)
    frame[y - 1 : y + 2, boundary + 1 : x1 + 1] = (0, 0, 210)

    health_cfg = HealthBarConfig(
        enabled=True,
        method="scanline",
        scanline_start_x=x0,
        scanline_end_x=x1,
        scanline_y=y,
        scanline_half_height=1,
        scanline_contrast_threshold=0.01,
        fill_direction="right_to_left",
        min_visible_pixels=10,
    )
    reader = HUDReader(ScoreOCRConfig(enabled=False), health_cfg)

    hud = reader.read(frame)
    assert hud.health_visible is True
    assert hud.health_percent is not None
    assert 0.72 <= hud.health_percent <= 0.78


def test_health_smoothing_window_reduces_single_frame_spike():
    y = 40
    x0 = 20
    x1 = 179

    def _frame(fill_end: int) -> np.ndarray:
        frame = np.zeros((100, 220, 3), dtype=np.uint8)
        frame[y - 1 : y + 2, x0 : fill_end + 1] = (0, 0, 210)
        frame[y - 1 : y + 2, fill_end + 1 : x1 + 1] = (0, 0, 50)
        return frame

    health_cfg = HealthBarConfig(
        enabled=True,
        method="scanline",
        scanline_start_x=x0,
        scanline_end_x=x1,
        scanline_y=y,
        scanline_half_height=1,
        scanline_contrast_threshold=0.01,
        fill_direction="left_to_right",
        min_visible_pixels=10,
        smoothing_window=3,
    )
    reader = HUDReader(ScoreOCRConfig(enabled=False), health_cfg)

    h1 = reader.read(_frame(145)).health_percent  # ~0.8
    h2 = reader.read(_frame(45)).health_percent   # ~0.2 spike
    h3 = reader.read(_frame(146)).health_percent  # ~0.8 again

    assert h1 is not None and h2 is not None and h3 is not None
    assert 0.75 <= h1 <= 0.9
    # Median of [~0.8, ~0.2, ~0.8] should settle near ~0.8.
    assert 0.7 <= h3 <= 0.9


def test_score_is_monotonic_and_spike_limited():
    reader = HUDReader(
        ScoreOCRConfig(
            enabled=False,
            monotonic_non_decreasing=True,
            hold_last_value_when_missing=True,
            max_step_increase=500,
        ),
        HealthBarConfig(enabled=False),
    )

    s1 = reader._stabilize_score(100)
    s2 = reader._stabilize_score(102)
    s3 = reader._stabilize_score(9999)
    s4 = reader._stabilize_score(103)

    assert s1 == 100
    assert s2 >= s1
    assert s3 == s2
    assert s4 >= s3

    # Lower read should not decrease the committed score.
    s5 = reader._stabilize_score(50)
    assert s5 == s4

    # Missing OCR keeps last value.
    s6 = reader._stabilize_score(None)
    assert s6 == s5


def test_score_reset_clears_monotonic_state():
    reader = HUDReader(
        ScoreOCRConfig(enabled=False, monotonic_non_decreasing=True),
        HealthBarConfig(enabled=False),
    )

    before = reader._stabilize_score(250)
    assert before == 250

    reader.reset()
    after = reader._stabilize_score(0)
    assert after == 0


def test_score_low_confidence_is_ignored():
    reader = HUDReader(
        ScoreOCRConfig(
            enabled=True,
            min_confidence=60.0,
            monotonic_non_decreasing=True,
            hold_last_value_when_missing=True,
        ),
        HealthBarConfig(enabled=False),
    )
    reader._committed_score = 123

    # Simulate candidate from OCR after confidence gate failed (i.e., read None).
    stable = reader._stabilize_score(None)
    assert stable == 123
