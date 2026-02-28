from __future__ import annotations

import json
import time
from collections import deque
from json import JSONDecodeError
from pathlib import Path
from typing import Any


DASHBOARD_METRICS_FILENAME = "dashboard_metrics.json"

METRIC_TAGS: dict[str, str] = {
    "timesteps": "time/total_timesteps",
    "fps": "time/fps",
    "iterations": "time/iterations",
    "ep_rew_mean": "rollout/ep_rew_mean",
    "ep_len_mean": "rollout/ep_len_mean",
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
    "entropy_loss": "train/entropy_loss",
    "explained_variance": "train/explained_variance",
    "learning_rate": "train/learning_rate",
    "loss": "train/loss",
    "policy_gradient_loss": "train/policy_gradient_loss",
    "value_loss": "train/value_loss",
    "n_updates": "train/n_updates",
}


def dashboard_metrics_path(run_dir: Path) -> Path:
    return run_dir / DASHBOARD_METRICS_FILENAME


def load_dashboard_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = dashboard_metrics_path(run_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError):
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload_text = json.dumps(payload, indent=2, sort_keys=True)
    last_error: OSError | None = None

    for _ in range(10):
        try:
            tmp_path.write_text(payload_text, encoding="utf-8")
            try:
                tmp_path.replace(path)
            except PermissionError:
                # Windows can reject os.replace when another process is briefly reading
                # the destination file. Fall back to direct overwrite and retry on failure.
                path.write_text(payload_text, encoding="utf-8")
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)

    if last_error is not None:
        raise last_error


class DashboardMetricsRecorder:
    def __init__(
        self,
        run_dir: Path,
        *,
        flush_interval_s: float = 1.0,
        episode_window: int = 50,
    ) -> None:
        self.run_dir = run_dir
        self.path = dashboard_metrics_path(run_dir)
        self.flush_interval_s = max(0.2, float(flush_interval_s))
        self.episode_rewards: deque[float] = deque(maxlen=max(1, int(episode_window)))
        self.episode_lengths: deque[float] = deque(maxlen=max(1, int(episode_window)))
        self.start_time = time.time()
        self.last_write_time = 0.0
        self.iteration = 0
        self.total_timesteps_target: int | None = None
        self.n_envs: int | None = None

    def on_training_start(self, *, total_timesteps: int | None, n_envs: int | None) -> None:
        self.total_timesteps_target = total_timesteps
        self.n_envs = n_envs
        self._write_payload(
            num_timesteps=0,
            logger_values={},
            status="starting",
            detail="Training callback attached",
            force=True,
        )

    def on_rollout_end(self) -> None:
        self.iteration += 1

    def on_step(
        self,
        *,
        num_timesteps: int,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        logger_values: dict[str, Any] | None,
        status: str = "live",
        detail: str = "",
        force: bool = False,
    ) -> None:
        if infos:
            for info in infos:
                episode = info.get("episode")
                if not episode:
                    continue
                reward = episode.get("r")
                length = episode.get("l")
                if reward is not None:
                    self.episode_rewards.append(float(reward))
                if length is not None:
                    self.episode_lengths.append(float(length))

        self._write_payload(
            num_timesteps=num_timesteps,
            logger_values=logger_values or {},
            status=status,
            detail=detail,
            force=force,
        )

    def on_training_end(self, *, num_timesteps: int, logger_values: dict[str, Any] | None) -> None:
        self._write_payload(
            num_timesteps=num_timesteps,
            logger_values=logger_values or {},
            status="stopped",
            detail="Training finished",
            force=True,
        )

    def _write_payload(
        self,
        *,
        num_timesteps: int,
        logger_values: dict[str, Any],
        status: str,
        detail: str,
        force: bool,
    ) -> None:
        now = time.time()
        if not force and (now - self.last_write_time) < self.flush_interval_s:
            return

        elapsed = max(0.0, now - self.start_time)
        metrics: dict[str, float] = {
            "timesteps": float(num_timesteps),
            "fps": float(num_timesteps / elapsed) if elapsed > 0 else 0.0,
            "iterations": float(self.iteration),
        }

        for metric_name, tag in METRIC_TAGS.items():
            value = logger_values.get(tag)
            if value is None:
                continue
            try:
                metrics[metric_name] = float(value)
            except (TypeError, ValueError):
                continue

        if self.episode_rewards:
            metrics["ep_rew_mean"] = sum(self.episode_rewards) / len(self.episode_rewards)
        if self.episode_lengths:
            metrics["ep_len_mean"] = sum(self.episode_lengths) / len(self.episode_lengths)

        payload = {
            "updated_at": now,
            "status": status,
            "detail": detail,
            "elapsed_time_s": elapsed,
            "total_timesteps_target": self.total_timesteps_target,
            "n_envs": self.n_envs,
            "metrics": metrics,
        }
        try:
            _atomic_write_json(self.path, payload)
        except OSError:
            # Dashboard sidecar writes are best-effort and must never kill training.
            return
        self.last_write_time = now
