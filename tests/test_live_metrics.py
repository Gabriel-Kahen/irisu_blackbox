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
        infos=[{"episode": {"r": 1.25, "l": 42}, "hud": {"score": 321, "health_percent": 0.75}}],
        logger_values={
            "time/fps": 3.0,
            "train/approx_kl": 0.01,
        },
        actions=[17],
        force=True,
    )

    path = dashboard_metrics_path(tmp_path)
    assert path.exists()

    payload = load_dashboard_metrics(tmp_path)
    assert payload is not None
    assert payload["metrics"]["timesteps"] == 42.0
    assert payload["metrics"]["iterations"] == 1.0
    assert payload["metrics"]["games_played"] == 1.0
    assert payload["metrics"]["high_score"] == 321.0
    assert payload["metrics"]["fps"] == 3.0
    assert payload["metrics"]["ep_rew_mean"] == 1.25
    assert payload["metrics"]["ep_len_mean"] == 42.0
    assert payload["metrics"]["approx_kl"] == 0.01
    assert payload["hud"]["score"] == 321
    assert payload["hud"]["health_percent"] == 0.75
    assert payload["control"]["last_action"] == 17
    assert payload["control"]["last_action_at"] >= 0.0
    assert payload["elapsed_time_s"] >= 0.0


def test_dashboard_metrics_recorder_resumes_games_runtime_and_best_score(tmp_path: Path):
    path = dashboard_metrics_path(tmp_path)
    path.write_text(
        """
{
  "updated_at": 1700000000.0,
  "status": "stopped",
  "detail": "done",
  "elapsed_time_s": 120.0,
  "metrics": {"games_played": 3.0, "high_score": 900.0},
  "hud": {"score": 900}
}
""".strip(),
        encoding="utf-8",
    )

    recorder = DashboardMetricsRecorder(tmp_path, flush_interval_s=0.0, episode_window=5)
    recorder.on_training_start(total_timesteps=1000, n_envs=1)
    recorder.on_step(
        num_timesteps=10,
        infos=[{"episode": {"r": 1.0, "l": 5}, "hud": {"score": 1200, "health_percent": 0.8}}],
        logger_values={},
        force=True,
    )

    payload = load_dashboard_metrics(tmp_path)
    assert payload is not None
    assert payload["metrics"]["games_played"] == 4.0
    assert payload["metrics"]["high_score"] == 1200.0
    assert payload["elapsed_time_s"] >= 120.0


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


def test_dashboard_metrics_recorder_flushes_early_for_live_hud_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    now = 100.0

    def _time() -> float:
        return now

    monkeypatch.setattr(live_metrics.time, "time", _time)
    recorder = DashboardMetricsRecorder(tmp_path, flush_interval_s=0.2, episode_window=5)
    recorder.on_training_start(total_timesteps=1000, n_envs=1)

    now = 100.06
    recorder.on_step(
        num_timesteps=1,
        infos=[{"hud": {"score": 7, "health_percent": 0.8}}],
        logger_values={},
        actions=[3],
        force=False,
    )

    payload = load_dashboard_metrics(tmp_path)
    assert payload is not None
    assert payload["metrics"]["timesteps"] == 1.0
    assert payload["hud"]["score"] == 7
    assert payload["control"]["last_action"] == 3
