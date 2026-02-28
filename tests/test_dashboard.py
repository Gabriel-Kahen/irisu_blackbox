from pathlib import Path

from irisu_blackbox.dashboard import (
    _format_compact_int,
    _format_duration,
    _format_percent,
    _latest_run_dir,
    _obs_text,
)
from irisu_blackbox.config import RootConfig


def test_format_duration():
    assert _format_duration(0) == "00:00:00"
    assert _format_duration(3661) == "01:01:01"


def test_format_helpers():
    assert _format_compact_int(950) == "950"
    assert _format_compact_int(1_200) == "1.2K"
    assert _format_compact_int(2_500_000) == "2.50M"
    assert _format_percent(0.125, digits=1) == "12.5%"


def test_latest_run_dir_picks_newest(tmp_path: Path):
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()

    older_time = 1_700_000_000
    newer_time = older_time + 100
    older.touch()
    newer.touch()

    import os

    os.utime(older, (older_time, older_time))
    os.utime(newer, (newer_time, newer_time))

    assert _latest_run_dir(tmp_path) == newer


def test_obs_text_describes_rgb_stack():
    cfg = RootConfig()
    cfg.env.frame_stack = 4
    cfg.env.obs_height = 96
    cfg.env.obs_width = 96
    cfg.env.hud_features.enabled = True

    assert _obs_text(cfg) == "RGB 12x96x96 + HUD4"
