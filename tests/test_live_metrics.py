from pathlib import Path

import pytest

import irisu_blackbox.live_metrics as live_metrics
from irisu_blackbox.live_metrics import (
    DashboardMetricsRecorder,
    dashboard_metrics_path,
    load_dashboard_metrics,
)


def test_dashboard_metrics_recorder_writes_file(tmp_path: Path):
    recorder = DashboardMetricsRecorder(tmp_path, flush_interval_s=0.0, episode_window=5)
    recorder.on_training_start(total_timesteps=1000, n_envs=1)
    recorder.on_rollout_end()
    recorder.on_step(
        num_timesteps=42,
        infos=[{"episode": {"r": 1.25, "l": 42}}],
        logger_values={
            "time/fps": 3.0,
            "train/approx_kl": 0.01,
        },
        force=True,
    )

    path = dashboard_metrics_path(tmp_path)
    assert path.exists()

    payload = load_dashboard_metrics(tmp_path)
    assert payload is not None
    assert payload["metrics"]["timesteps"] == 42.0
    assert payload["metrics"]["iterations"] == 1.0
    assert payload["metrics"]["fps"] == 3.0
    assert payload["metrics"]["ep_rew_mean"] == 1.25
    assert payload["metrics"]["ep_len_mean"] == 42.0
    assert payload["metrics"]["approx_kl"] == 0.01


def test_dashboard_metrics_recorder_ignores_write_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    recorder = DashboardMetricsRecorder(tmp_path, flush_interval_s=0.0, episode_window=5)

    def _boom(path, payload):
        raise PermissionError("locked")

    monkeypatch.setattr(live_metrics, "_atomic_write_json", _boom)

    recorder.on_training_start(total_timesteps=1000, n_envs=1)
    recorder.on_step(
        num_timesteps=10,
        infos=[],
        logger_values={},
        force=True,
    )


def test_load_dashboard_metrics_returns_none_for_partial_json(tmp_path: Path):
    path = dashboard_metrics_path(tmp_path)
    path.write_text("{", encoding="utf-8")

    assert load_dashboard_metrics(tmp_path) is None
