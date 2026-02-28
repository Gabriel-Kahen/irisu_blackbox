from pathlib import Path

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
