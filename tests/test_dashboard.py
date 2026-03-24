from pathlib import Path

from irisu_blackbox.dashboard import (
    _architecture_nodes,
    _action_output_info,
    _format_compact_int,
    _format_duration,
    _format_percent,
    _format_rate,
    _format_update_age,
    DashboardFileReader,
    _latest_run_dir,
    _obs_text,
)
from irisu_blackbox.config import RootConfig
from irisu_blackbox.live_metrics import dashboard_metrics_path


def test_format_duration():
    assert _format_duration(0) == "00:00:00"
    assert _format_duration(3661) == "01:01:01"
    assert _format_update_age(0.24) == "0.2s"
    assert _format_update_age(12.8) == "12s"


def test_format_helpers():
    assert _format_compact_int(950) == "950"
    assert _format_compact_int(1_200) == "1.2K"
    assert _format_compact_int(2_500_000) == "2.50M"
    assert _format_percent(0.125, digits=1) == "12.5%"
    assert _format_rate(0.1764) == "0.176"
    assert _format_rate(1.234) == "1.23"
    assert _format_rate(12.34) == "12.3"


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


def test_architecture_nodes_include_hud_and_heads():
    cfg = RootConfig()
    cfg.env.hud_features.enabled = True
    cfg.env.frame_stack = 4
    cfg.env.obs_width = 96
    cfg.env.obs_height = 96

    nodes = _architecture_nodes(cfg)
    titles = [node["title"] for node in nodes]

    assert titles == [
        "SCREEN STACK",
        "HUD FEATURES",
        "CNN ENCODER",
        "FUSION",
        "LSTM MEMORY",
        "POLICY HEAD",
        "VALUE HEAD",
    ]


def test_dashboard_file_reader_loads_hud_payload(tmp_path: Path):
    path = dashboard_metrics_path(tmp_path)
    path.write_text(
        """
{
  "updated_at": 1700000000.0,
  "status": "live",
  "detail": "ok",
  "elapsed_time_s": 15.0,
  "metrics": {"timesteps": 10.0},
  "hud": {"score": 999, "health_percent": 0.5, "health_visible": true},
  "control": {"last_action": 17, "last_action_at": 1700000000.0}
}
""".strip(),
        encoding="utf-8",
    )

    snapshot = DashboardFileReader(tmp_path).read()
    assert snapshot is not None
    assert snapshot.metrics["timesteps"] == 10.0
    assert snapshot.hud["score"] == 999
    assert snapshot.hud["health_percent"] == 0.5
    assert snapshot.hud["health_visible"] is True
    assert snapshot.control["last_action"] == 17
    assert snapshot.control["last_action_at"] == 1700000000.0
    assert snapshot.updated_at_s == 1700000000.0
    assert snapshot.elapsed_time_s == 15.0


def test_action_output_info_decodes_left_and_right_actions():
    cfg = RootConfig().env.action_grid
    kind, row, col, label = _action_output_info(1, cfg)
    assert kind == "left"
    assert row == 0
    assert col == 0
    assert "LEFT" in label

    kind, row, col, label = _action_output_info(1 + cfg.grid_size, cfg)
    assert kind == "right"
    assert row == 0
    assert col == 0
    assert "RIGHT" in label
