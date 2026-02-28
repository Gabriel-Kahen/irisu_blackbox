from __future__ import annotations

import argparse
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from irisu_blackbox.config import ActionGridConfig, RootConfig, load_config

try:  # pragma: no cover - depends on installed runtime packages
    from tensorboard.backend.event_processing import event_accumulator as tb_event_accumulator
except Exception as exc:  # pragma: no cover
    tb_event_accumulator = None
    _TENSORBOARD_IMPORT_ERROR = exc
else:  # pragma: no cover
    _TENSORBOARD_IMPORT_ERROR = None


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


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_compact_int(value: float | int | None) -> str:
    if value is None:
        return "--"
    n = float(value)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(round(n))}"


def _format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _format_percent(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{float(value) * 100.0:.{digits}f}%"


def _latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.expanduser().resolve()
    latest = _latest_run_dir(Path("runs").resolve())
    if latest is None:
        raise FileNotFoundError("No run directory found under runs/")
    return latest


def _policy_name(cfg: RootConfig) -> str:
    return "MultiInputLstmPolicy" if cfg.env.hud_features.enabled else "CnnLstmPolicy"


def _obs_text(cfg: RootConfig) -> str:
    channels = cfg.env.frame_stack * 3
    hud = " + HUD4" if cfg.env.hud_features.enabled else ""
    return f"RGB {channels}x{cfg.env.obs_height}x{cfg.env.obs_width}{hud}"


def _action_text(grid: ActionGridConfig) -> str:
    return f"{grid.rows}x{grid.cols} ({grid.action_count} actions)"


@dataclass(slots=True)
class TrainingSnapshot:
    metrics: dict[str, float]
    event_path: Path | None
    event_age_s: float | None
    status: str
    detail: str


class TensorboardMetricsReader:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.tensorboard_dir = run_dir / "tensorboard"
        self._event_path: Path | None = None
        self._accumulator = None

    def _latest_event_file(self) -> Path | None:
        if not self.tensorboard_dir.exists():
            return None
        event_files = sorted(
            self.tensorboard_dir.rglob("events.out.tfevents.*"),
            key=lambda path: path.stat().st_mtime,
        )
        return event_files[-1] if event_files else None

    def _ensure_accumulator(self, event_path: Path):
        if tb_event_accumulator is None:
            return None
        if self._event_path == event_path and self._accumulator is not None:
            return self._accumulator
        self._event_path = event_path
        self._accumulator = tb_event_accumulator.EventAccumulator(
            str(event_path),
            size_guidance={"scalars": 0},
        )
        return self._accumulator

    def read(self) -> TrainingSnapshot:
        if tb_event_accumulator is None:
            return TrainingSnapshot(
                metrics={},
                event_path=None,
                event_age_s=None,
                status="NO TENSORBOARD",
                detail=str(_TENSORBOARD_IMPORT_ERROR),
            )

        event_path = self._latest_event_file()
        if event_path is None:
            detail = (
                f"Waiting for logs in {self.tensorboard_dir}"
                if self.tensorboard_dir.exists()
                else f"Missing tensorboard dir: {self.tensorboard_dir}"
            )
            return TrainingSnapshot(
                metrics={},
                event_path=None,
                event_age_s=None,
                status="WAITING",
                detail=detail,
            )

        accumulator = self._ensure_accumulator(event_path)
        try:
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags().get("scalars", []))
        except Exception as exc:
            return TrainingSnapshot(
                metrics={},
                event_path=event_path,
                event_age_s=max(0.0, time.time() - event_path.stat().st_mtime),
                status="ERROR",
                detail=f"Failed to read event file: {exc}",
            )

        metrics: dict[str, float] = {}
        latest_wall_time: float | None = None
        for metric_name, tag in METRIC_TAGS.items():
            if tag not in scalar_tags:
                continue
            try:
                events = accumulator.Scalars(tag)
            except Exception:
                continue
            if not events:
                continue
            latest = events[-1]
            metrics[metric_name] = float(latest.value)
            latest_wall_time = max(latest_wall_time or latest.wall_time, latest.wall_time)

        if latest_wall_time is None:
            return TrainingSnapshot(
                metrics={},
                event_path=event_path,
                event_age_s=max(0.0, time.time() - event_path.stat().st_mtime),
                status="WAITING",
                detail="Event file exists but no scalar metrics are available yet",
            )

        age_s = max(0.0, time.time() - latest_wall_time)
        status = "LIVE" if age_s <= 20.0 else "STALE"
        detail = event_path.parent.name
        return TrainingSnapshot(
            metrics=metrics,
            event_path=event_path,
            event_age_s=age_s,
            status=status,
            detail=detail,
        )


class DashboardWindow:
    def __init__(
        self,
        *,
        cfg: RootConfig,
        run_dir: Path,
        reader: TensorboardMetricsReader,
        interval_s: float,
        geometry: str,
        always_on_top: bool,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.reader = reader
        self.interval_ms = max(200, int(interval_s * 1000.0))

        self.root = tk.Tk()
        self.root.title("Irisu RL Dashboard")
        self.root.geometry(geometry)
        self.root.configure(bg="#0a0f1a")
        self.root.attributes("-topmost", bool(always_on_top))
        self.root.resizable(False, False)
        self.root.bind("<Escape>", lambda _event: self.close())

        self._build_ui()

    def _build_ui(self) -> None:
        title_font = ("Consolas", 24, "bold")
        hero_font = ("Consolas", 34, "bold")
        metric_title_font = ("Consolas", 12, "bold")
        metric_value_font = ("Consolas", 18, "bold")
        small_font = ("Consolas", 12)

        shell = tk.Frame(self.root, bg="#0a0f1a")
        shell.pack(fill="both", expand=True, padx=20, pady=18)

        tk.Label(
            shell,
            text="IRISU RL DASHBOARD",
            bg="#0a0f1a",
            fg="#f8fafc",
            font=title_font,
            anchor="w",
        ).pack(fill="x")

        self.status_label = tk.Label(
            shell,
            text="WAITING",
            bg="#0a0f1a",
            fg="#7dd3fc",
            font=("Consolas", 18, "bold"),
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(6, 2))

        self.detail_label = tk.Label(
            shell,
            text=str(self.run_dir),
            bg="#0a0f1a",
            fg="#94a3b8",
            font=small_font,
            anchor="w",
            justify="left",
            wraplength=430,
        )
        self.detail_label.pack(fill="x", pady=(0, 18))

        self.hero_timestep = self._hero_block(shell, "TIMESTEPS", hero_font)
        self.hero_reward = self._hero_block(shell, "EP REWARD", hero_font)

        grid = tk.Frame(shell, bg="#0a0f1a")
        grid.pack(fill="x", pady=(8, 16))

        self.metric_boxes: dict[str, tk.Label] = {}
        metric_layout = [
            ("fps", "FPS"),
            ("iterations", "ITERATIONS"),
            ("n_updates", "UPDATES"),
            ("ep_len_mean", "EP LENGTH"),
            ("approx_kl", "APPROX KL"),
            ("clip_fraction", "CLIP FRAC"),
            ("entropy_loss", "ENTROPY"),
            ("explained_variance", "EXPLAINED VAR"),
            ("policy_gradient_loss", "POLICY LOSS"),
            ("value_loss", "VALUE LOSS"),
            ("loss", "TOTAL LOSS"),
            ("learning_rate", "LR"),
        ]
        for idx, (key, title) in enumerate(metric_layout):
            box = tk.Frame(
                grid,
                bg="#101827",
                highlightthickness=1,
                highlightbackground="#23304d",
                bd=0,
            )
            box.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=(0, 12), pady=(0, 12))
            grid.grid_columnconfigure(idx % 2, weight=1)

            tk.Label(
                box,
                text=title,
                bg="#101827",
                fg="#94a3b8",
                font=metric_title_font,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 2))
            value_label = tk.Label(
                box,
                text="--",
                bg="#101827",
                fg="#f8fafc",
                font=metric_value_font,
                anchor="w",
            )
            value_label.pack(fill="x", padx=12, pady=(0, 10))
            self.metric_boxes[key] = value_label

        config_box = tk.Frame(
            shell,
            bg="#101827",
            highlightthickness=1,
            highlightbackground="#23304d",
            bd=0,
        )
        config_box.pack(fill="x", pady=(4, 16))

        tk.Label(
            config_box,
            text="MODEL CONFIG",
            bg="#101827",
            fg="#94a3b8",
            font=metric_title_font,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        static_lines = [
            f"Policy: {_policy_name(self.cfg)}",
            f"Obs: {_obs_text(self.cfg)}",
            f"Grid: {_action_text(self.cfg.env.action_grid)}",
            f"LR: {_format_float(self.cfg.train.learning_rate, 5)}   Gamma: {_format_float(self.cfg.train.gamma, 3)}",
            f"Rollout: {self.cfg.train.n_steps}   Batch: {self.cfg.train.batch_size}   Envs: {self.cfg.train.n_envs}",
        ]
        for line in static_lines:
            tk.Label(
                config_box,
                text=line,
                bg="#101827",
                fg="#e2e8f0",
                font=small_font,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=(0, 6))

        self.footer_label = tk.Label(
            shell,
            text="ESC to close",
            bg="#0a0f1a",
            fg="#94a3b8",
            font=small_font,
            anchor="w",
        )
        self.footer_label.pack(fill="x")

    def _hero_block(self, parent: tk.Widget, title: str, value_font) -> tk.Label:
        box = tk.Frame(
            parent,
            bg="#101827",
            highlightthickness=1,
            highlightbackground="#23304d",
            bd=0,
        )
        box.pack(fill="x", pady=(0, 14))
        tk.Label(
            box,
            text=title,
            bg="#101827",
            fg="#94a3b8",
            font=("Consolas", 12, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        value = tk.Label(
            box,
            text="--",
            bg="#101827",
            fg="#f8fafc",
            font=value_font,
            anchor="w",
        )
        value.pack(fill="x", padx=12, pady=(0, 10))
        return value

    def _metric_text(self, key: str, metrics: dict[str, float]) -> str:
        value = metrics.get(key)
        if key in {"fps", "iterations", "n_updates", "ep_len_mean"}:
            return _format_compact_int(value)
        if key == "clip_fraction":
            return _format_percent(value, digits=1)
        if key == "learning_rate":
            return _format_float(value, digits=6)
        if key == "explained_variance":
            return _format_float(value, digits=3)
        return _format_float(value, digits=4)

    def refresh(self) -> None:
        snapshot = self.reader.read()

        status_color = {
            "LIVE": "#4ade80",
            "STALE": "#facc15",
            "WAITING": "#7dd3fc",
            "ERROR": "#f97316",
            "NO TENSORBOARD": "#ef4444",
        }.get(snapshot.status, "#e2e8f0")

        self.status_label.configure(text=snapshot.status, fg=status_color)
        self.detail_label.configure(text=snapshot.detail)
        self.hero_timestep.configure(text=_format_compact_int(snapshot.metrics.get("timesteps")))
        self.hero_reward.configure(text=_format_float(snapshot.metrics.get("ep_rew_mean"), digits=3))

        for key, label in self.metric_boxes.items():
            label.configure(text=self._metric_text(key, snapshot.metrics))

        age_text = "--"
        if snapshot.event_age_s is not None:
            age_text = _format_duration(snapshot.event_age_s)
        event_name = snapshot.event_path.name if snapshot.event_path is not None else "none"
        self.footer_label.configure(text=f"Event age: {age_text}   Source: {event_name}   ESC to close")

        self.root.after(self.interval_ms, self.refresh)

    def run(self) -> None:
        self.refresh()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    def close(self) -> None:
        self.root.destroy()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a stream-friendly Irisu RL metrics dashboard")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory. Defaults to the newest subdirectory under runs/",
    )
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--geometry", type=str, default="480x1080+0+0")
    parser.add_argument("--topmost", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = load_config(args.config)
    run_dir = _resolve_run_dir(args.run_dir)
    reader = TensorboardMetricsReader(run_dir)
    dashboard = DashboardWindow(
        cfg=cfg,
        run_dir=run_dir,
        reader=reader,
        interval_s=args.interval_s,
        geometry=args.geometry,
        always_on_top=args.topmost,
    )
    dashboard.run()


if __name__ == "__main__":
    main()
