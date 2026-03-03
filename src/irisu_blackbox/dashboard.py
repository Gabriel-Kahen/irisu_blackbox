from __future__ import annotations

import argparse
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from irisu_blackbox.config import ActionGridConfig, RootConfig, load_config
from irisu_blackbox.live_metrics import (
    METRIC_TAGS,
    dashboard_metrics_path,
    load_dashboard_metrics,
)

try:  # pragma: no cover - depends on installed runtime packages
    from tensorboard.backend.event_processing import event_accumulator as tb_event_accumulator
    from tensorboard.util import tensor_util as tb_tensor_util
except Exception as exc:  # pragma: no cover
    tb_event_accumulator = None
    tb_tensor_util = None
    _TENSORBOARD_IMPORT_ERROR = exc
else:  # pragma: no cover
    _TENSORBOARD_IMPORT_ERROR = None


BG = "#0a0f1a"
PANEL_BG = "#101827"
PANEL_BORDER = "#23304d"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
ACCENT = "#7dd3fc"
GOOD = "#4ade80"
WARN = "#facc15"
BAD = "#ef4444"
INPUT_A = "#38bdf8"
INPUT_B = "#f59e0b"
ENCODER = "#818cf8"
FUSION = "#22c55e"
MEMORY = "#f472b6"
HEAD = "#fb7185"
VALUE = "#c084fc"


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


def _architecture_nodes(cfg: RootConfig) -> list[dict[str, str]]:
    channels = cfg.env.frame_stack * 3
    nodes = [
        {
            "title": "SCREEN STACK",
            "subtitle": f"{channels} channels",
            "detail": f"{cfg.env.obs_width}x{cfg.env.obs_height} RGB replay stack",
            "color": INPUT_A,
        },
    ]
    if cfg.env.hud_features.enabled:
        nodes.append(
            {
                "title": "HUD FEATURES",
                "subtitle": "4 scalars",
                "detail": "health, score, visible flags",
                "color": INPUT_B,
            }
        )
    nodes.extend(
        [
            {
                "title": "CNN ENCODER",
                "subtitle": "Nature-style vision trunk",
                "detail": "conv -> conv -> conv -> flatten",
                "color": ENCODER,
            },
            {
                "title": "FUSION",
                "subtitle": "joint latent",
                "detail": "concat image embedding with HUD stream"
                if cfg.env.hud_features.enabled
                else "image embedding only",
                "color": FUSION,
            },
            {
                "title": "LSTM MEMORY",
                "subtitle": "recurrent state",
                "detail": "temporal credit assignment across frames",
                "color": MEMORY,
            },
            {
                "title": "POLICY HEAD",
                "subtitle": f"{cfg.env.action_grid.action_count} logits",
                "detail": f"discrete click grid {_action_text(cfg.env.action_grid)}",
                "color": HEAD,
            },
            {
                "title": "VALUE HEAD",
                "subtitle": "1 scalar",
                "detail": "state-value baseline for PPO",
                "color": VALUE,
            },
        ]
    )
    return nodes


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
        search_roots = []
        if self.tensorboard_dir.exists():
            search_roots.append(self.tensorboard_dir)
        if self.run_dir.exists():
            search_roots.append(self.run_dir)

        event_files: list[Path] = []
        for root in search_roots:
            event_files.extend(root.rglob("events.out.tfevents.*"))

        event_files = sorted(
            {path.resolve() for path in event_files},
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
            size_guidance={"scalars": 0, "tensors": 0},
        )
        return self._accumulator

    @staticmethod
    def _latest_scalar_from_tensors(accumulator, tag: str) -> tuple[float, float] | None:
        if tb_tensor_util is None:
            return None
        try:
            tensor_events = accumulator.Tensors(tag)
        except Exception:
            return None
        if not tensor_events:
            return None
        latest = tensor_events[-1]
        try:
            value = tb_tensor_util.make_ndarray(latest.tensor_proto).item()
        except Exception:
            return None
        return float(value), float(latest.wall_time)

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
            tags = accumulator.Tags()
            scalar_tags = set(tags.get("scalars", []))
            tensor_tags = set(tags.get("tensors", []))
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
            if tag in scalar_tags:
                try:
                    events = accumulator.Scalars(tag)
                except Exception:
                    events = []
                if events:
                    latest = events[-1]
                    metrics[metric_name] = float(latest.value)
                    latest_wall_time = max(latest_wall_time or latest.wall_time, latest.wall_time)
                    continue

            if tag in tensor_tags:
                tensor_value = self._latest_scalar_from_tensors(accumulator, tag)
                if tensor_value is None:
                    continue
                value, wall_time = tensor_value
                metrics[metric_name] = value
                latest_wall_time = max(latest_wall_time or wall_time, wall_time)

        if latest_wall_time is None:
            return TrainingSnapshot(
                metrics={},
                event_path=event_path,
                event_age_s=max(0.0, time.time() - event_path.stat().st_mtime),
                status="WAITING",
                detail="Event file exists but no readable scalar/tensor metrics are available yet",
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


class DashboardFileReader:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.metrics_path = dashboard_metrics_path(run_dir)

    def read(self) -> TrainingSnapshot | None:
        payload = load_dashboard_metrics(self.run_dir)
        if payload is None:
            return None

        updated_at = payload.get("updated_at")
        try:
            updated_at_f = float(updated_at)
        except (TypeError, ValueError):
            updated_at_f = None

        age_s = max(0.0, time.time() - updated_at_f) if updated_at_f is not None else None
        file_status = str(payload.get("status", "live")).upper()
        if file_status == "LIVE" and age_s is not None and age_s > 10.0:
            status = "STALE"
        elif file_status == "STARTING":
            status = "WAITING"
        elif file_status == "STOPPED":
            status = "STOPPED"
        else:
            status = file_status

        detail = str(payload.get("detail", "dashboard_metrics.json"))
        return TrainingSnapshot(
            metrics={
                str(key): float(value)
                for key, value in dict(payload.get("metrics", {})).items()
                if value is not None
            },
            event_path=self.metrics_path,
            event_age_s=age_s,
            status=status,
            detail=detail,
        )


class CompositeMetricsReader:
    def __init__(self, run_dir: Path) -> None:
        self.dashboard_reader = DashboardFileReader(run_dir)
        self.tensorboard_reader = TensorboardMetricsReader(run_dir)

    def read(self) -> TrainingSnapshot:
        dashboard_snapshot = self.dashboard_reader.read()
        if dashboard_snapshot is not None:
            return dashboard_snapshot

        tensorboard_snapshot = self.tensorboard_reader.read()
        if tensorboard_snapshot.status == "WAITING":
            metrics_path = self.dashboard_reader.metrics_path
            tensorboard_snapshot.detail = (
                f"No {metrics_path.name} yet. "
                "Restart training on the latest code for true live dashboard metrics."
            )
        return tensorboard_snapshot


class BaseDashboardWindow:
    def __init__(
        self,
        *,
        cfg: RootConfig,
        run_dir: Path,
        reader: CompositeMetricsReader,
        interval_s: float,
        geometry: str,
        always_on_top: bool,
        title: str,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.reader = reader
        self.interval_ms = max(200, int(interval_s * 1000.0))

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(geometry)
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", bool(always_on_top))
        self.root.resizable(True, True)
        self.root.bind("<Escape>", lambda _event: self.close())

    def run(self) -> None:
        self.refresh()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    def refresh(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.root.destroy()


class MetricsDashboardWindow(BaseDashboardWindow):
    def __init__(
        self,
        *,
        cfg: RootConfig,
        run_dir: Path,
        reader: CompositeMetricsReader,
        interval_s: float,
        geometry: str,
        always_on_top: bool,
    ) -> None:
        super().__init__(
            cfg=cfg,
            run_dir=run_dir,
            reader=reader,
            interval_s=interval_s,
            geometry=geometry,
            always_on_top=always_on_top,
            title="Irisu RL Dashboard",
        )
        self._build_ui()

    def _build_ui(self) -> None:
        title_font = ("Consolas", 24, "bold")
        hero_font = ("Consolas", 34, "bold")
        metric_title_font = ("Consolas", 12, "bold")
        metric_value_font = ("Consolas", 18, "bold")
        small_font = ("Consolas", 12)

        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True, padx=20, pady=18)

        tk.Label(
            shell,
            text="IRISU RL DASHBOARD",
            bg=BG,
            fg=TEXT,
            font=title_font,
            anchor="w",
        ).pack(fill="x")

        self.status_label = tk.Label(
            shell,
            text="WAITING",
            bg=BG,
            fg=ACCENT,
            font=("Consolas", 18, "bold"),
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(6, 2))

        self.detail_label = tk.Label(
            shell,
            text=str(self.run_dir),
            bg=BG,
            fg=MUTED,
            font=small_font,
            anchor="w",
            justify="left",
            wraplength=430,
        )
        self.detail_label.pack(fill="x", pady=(0, 18))

        self.hero_timestep = self._hero_block(shell, "TIMESTEPS", hero_font)
        self.hero_reward = self._hero_block(shell, "EP REWARD", hero_font)

        grid = tk.Frame(shell, bg=BG)
        grid.pack(fill="x", pady=(8, 16), expand=True)

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
                bg=PANEL_BG,
                highlightthickness=1,
                highlightbackground=PANEL_BORDER,
                bd=0,
            )
            box.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=(0, 12), pady=(0, 12))
            grid.grid_columnconfigure(idx % 2, weight=1)

            tk.Label(
                box,
                text=title,
                bg=PANEL_BG,
                fg=MUTED,
                font=metric_title_font,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 2))
            value_label = tk.Label(
                box,
                text="--",
                bg=PANEL_BG,
                fg=TEXT,
                font=metric_value_font,
                anchor="w",
            )
            value_label.pack(fill="x", padx=12, pady=(0, 10))
            self.metric_boxes[key] = value_label

        config_box = tk.Frame(
            shell,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            bd=0,
        )
        config_box.pack(fill="x", pady=(4, 16))

        tk.Label(
            config_box,
            text="MODEL CONFIG",
            bg=PANEL_BG,
            fg=MUTED,
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
                bg=PANEL_BG,
                fg="#e2e8f0",
                font=small_font,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=(0, 6))

        self.footer_label = tk.Label(
            shell,
            text="ESC to close",
            bg=BG,
            fg=MUTED,
            font=small_font,
            anchor="w",
        )
        self.footer_label.pack(fill="x")

    def _hero_block(self, parent: tk.Widget, title: str, value_font) -> tk.Label:
        box = tk.Frame(
            parent,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            bd=0,
        )
        box.pack(fill="x", pady=(0, 14))
        tk.Label(
            box,
            text=title,
            bg=PANEL_BG,
            fg=MUTED,
            font=("Consolas", 12, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        value = tk.Label(
            box,
            text="--",
            bg=PANEL_BG,
            fg=TEXT,
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
            "LIVE": GOOD,
            "STALE": WARN,
            "WAITING": ACCENT,
            "STOPPED": MUTED,
            "ERROR": "#f97316",
            "NO TENSORBOARD": BAD,
        }.get(snapshot.status, "#e2e8f0")

        self.status_label.configure(text=snapshot.status, fg=status_color)
        detail = snapshot.detail
        if snapshot.event_path is not None:
            detail = f"{detail}\n{snapshot.event_path}"
        self.detail_label.configure(text=detail)
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


class NetworkDashboardWindow(BaseDashboardWindow):
    def __init__(
        self,
        *,
        cfg: RootConfig,
        run_dir: Path,
        reader: CompositeMetricsReader,
        interval_s: float,
        geometry: str,
        always_on_top: bool,
    ) -> None:
        super().__init__(
            cfg=cfg,
            run_dir=run_dir,
            reader=reader,
            interval_s=interval_s,
            geometry=geometry,
            always_on_top=always_on_top,
            title="Irisu Network Dashboard",
        )
        self.nodes = _architecture_nodes(cfg)
        self._build_ui()

    def _build_ui(self) -> None:
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg=BG)
        header.pack(fill="x", padx=18, pady=(16, 12))

        tk.Label(
            header,
            text="IRISU POLICY MAP",
            bg=BG,
            fg=TEXT,
            font=("Consolas", 24, "bold"),
            anchor="w",
        ).pack(fill="x")

        self.status_label = tk.Label(
            header,
            text="WAITING",
            bg=BG,
            fg=ACCENT,
            font=("Consolas", 18, "bold"),
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(6, 2))

        self.subtitle_label = tk.Label(
            header,
            text=f"{_policy_name(self.cfg)}   |   {_obs_text(self.cfg)}",
            bg=BG,
            fg=MUTED,
            font=("Consolas", 12),
            anchor="w",
        )
        self.subtitle_label.pack(fill="x")

        self.badge_row = tk.Frame(shell, bg=BG)
        self.badge_row.pack(fill="x", padx=18, pady=(0, 10))

        self.badges: dict[str, tk.Label] = {}
        for key, title in [
            ("timesteps", "TIMESTEPS"),
            ("fps", "FPS"),
            ("iterations", "ROLLOUTS"),
            ("ep_rew_mean", "MEAN REWARD"),
        ]:
            badge = tk.Frame(
                self.badge_row,
                bg=PANEL_BG,
                highlightthickness=1,
                highlightbackground=PANEL_BORDER,
                bd=0,
            )
            badge.pack(side="left", fill="x", expand=True, padx=(0, 10))
            tk.Label(
                badge,
                text=title,
                bg=PANEL_BG,
                fg=MUTED,
                font=("Consolas", 10, "bold"),
                anchor="w",
            ).pack(fill="x", padx=10, pady=(8, 0))
            value = tk.Label(
                badge,
                text="--",
                bg=PANEL_BG,
                fg=TEXT,
                font=("Consolas", 18, "bold"),
                anchor="w",
            )
            value.pack(fill="x", padx=10, pady=(0, 8))
            self.badges[key] = value

        self.canvas = tk.Canvas(
            shell,
            bg=BG,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.canvas.pack(fill="both", expand=True, padx=14, pady=(2, 8))
        self.canvas.bind("<Configure>", lambda _event: self._draw_architecture(None))

        footer = tk.Frame(shell, bg=BG)
        footer.pack(fill="x", padx=18, pady=(0, 16))

        self.detail_label = tk.Label(
            footer,
            text=str(self.run_dir),
            bg=BG,
            fg=MUTED,
            font=("Consolas", 11),
            anchor="w",
            justify="left",
            wraplength=680,
        )
        self.detail_label.pack(fill="x")

        self.footer_label = tk.Label(
            footer,
            text="ESC to close",
            bg=BG,
            fg=MUTED,
            font=("Consolas", 11),
            anchor="w",
        )
        self.footer_label.pack(fill="x", pady=(6, 0))

    def _badge_text(self, key: str, snapshot: TrainingSnapshot) -> str:
        metrics = snapshot.metrics
        if key in {"timesteps", "iterations", "fps"}:
            return _format_compact_int(metrics.get(key))
        if key == "ep_rew_mean":
            return _format_float(metrics.get(key), digits=3)
        return "--"

    def _draw_architecture(self, snapshot: TrainingSnapshot | None) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(640, canvas.winfo_width())
        height = max(720, canvas.winfo_height())

        for x in range(0, width, 28):
            line_color = "#0f1728" if (x // 28) % 2 == 0 else "#0c1321"
            canvas.create_line(x, 0, x, height, fill=line_color)
        for y in range(0, height, 28):
            canvas.create_line(0, y, width, y, fill="#0f1728")

        status_color = {
            "LIVE": GOOD,
            "STALE": WARN,
            "WAITING": ACCENT,
            "STOPPED": MUTED,
            "ERROR": "#f97316",
            "NO TENSORBOARD": BAD,
        }.get(snapshot.status if snapshot else "WAITING", ACCENT)

        top = 36
        left_margin = 40
        box_w = min(230, max(170, (width - 110) // 3))
        box_h = 90
        v_gap = 56

        x_left = left_margin
        x_mid = left_margin + box_w + 90
        x_right = width - box_w - 40

        positions = {
            "screen": (x_left, top),
            "hud": (x_left, top + box_h + v_gap) if self.cfg.env.hud_features.enabled else None,
            "cnn": (x_mid, top + 8),
            "fusion": (x_mid, top + box_h + v_gap + 8),
            "lstm": (x_mid, top + (box_h + v_gap) * 2),
            "policy": (x_right, top + 56),
            "value": (x_right, top + 56 + box_h + v_gap),
        }

        screen_node = self.nodes[0]
        self._draw_node(canvas, *positions["screen"], box_w, box_h, screen_node)

        node_idx = 1
        if self.cfg.env.hud_features.enabled:
            self._draw_node(canvas, *positions["hud"], box_w, box_h, self.nodes[node_idx])
            node_idx += 1
        self._draw_node(canvas, *positions["cnn"], box_w, box_h, self.nodes[node_idx])
        self._draw_node(canvas, *positions["fusion"], box_w, box_h, self.nodes[node_idx + 1])
        self._draw_node(canvas, *positions["lstm"], box_w, box_h, self.nodes[node_idx + 2])
        self._draw_node(canvas, *positions["policy"], box_w, box_h, self.nodes[node_idx + 3])
        self._draw_node(canvas, *positions["value"], box_w, box_h, self.nodes[node_idx + 4])

        self._draw_arrow(
            canvas,
            x_left + box_w,
            top + box_h / 2,
            x_mid,
            top + box_h / 2 + 8,
            INPUT_A,
            "vision features",
        )
        if self.cfg.env.hud_features.enabled and positions["hud"] is not None:
            hud_x, hud_y = positions["hud"]
            fusion_x, fusion_y = positions["fusion"]
            self._draw_arrow(
                canvas,
                x_left + box_w,
                hud_y + box_h / 2,
                x_mid,
                fusion_y + box_h / 2,
                INPUT_B,
                "scalar stream",
            )
            self._draw_arrow(
                canvas,
                x_mid + box_w / 2,
                top + box_h + 8,
                x_mid + box_w / 2,
                fusion_y,
                ENCODER,
                "embedding",
            )
        else:
            fusion_x, fusion_y = positions["fusion"]
            self._draw_arrow(
                canvas,
                x_mid + box_w / 2,
                top + box_h + 8,
                x_mid + box_w / 2,
                fusion_y,
                ENCODER,
                "image latent",
            )

        fusion_x, fusion_y = positions["fusion"]
        lstm_x, lstm_y = positions["lstm"]
        self._draw_arrow(
            canvas,
            fusion_x + box_w / 2,
            fusion_y + box_h,
            lstm_x + box_w / 2,
            lstm_y,
            FUSION,
            "temporal sequence",
        )
        policy_x, policy_y = positions["policy"]
        value_x, value_y = positions["value"]
        self._draw_arrow(
            canvas,
            x_mid + box_w,
            lstm_y + box_h / 2,
            policy_x,
            policy_y + box_h / 2,
            MEMORY,
            "actor branch",
        )
        self._draw_arrow(
            canvas,
            x_mid + box_w,
            lstm_y + box_h / 2,
            value_x,
            value_y + box_h / 2,
            MEMORY,
            "critic branch",
        )

        loop_x = x_mid + box_w / 2
        canvas.create_arc(
            loop_x - 70,
            lstm_y - 56,
            loop_x + 70,
            lstm_y + 28,
            start=20,
            extent=300,
            style="arc",
            width=3,
            outline=status_color,
        )
        canvas.create_text(
            loop_x,
            lstm_y - 66,
            text="memory carries across frames",
            fill=MUTED,
            font=("Consolas", 11, "bold"),
        )

        if snapshot is not None:
            policy_temp = snapshot.metrics.get("entropy_loss")
            confidence = snapshot.metrics.get("clip_fraction")
            ev = snapshot.metrics.get("explained_variance")
            canvas.create_text(
                x_right + box_w / 2,
                height - 82,
                text=(
                    f"policy entropy: {_format_float(policy_temp, 3)}    "
                    f"clip frac: {_format_percent(confidence, 1)}    "
                    f"explained var: {_format_float(ev, 3)}"
                ),
                fill=MUTED,
                font=("Consolas", 12),
            )

    def _draw_node(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        w: float,
        h: float,
        node: dict[str, str],
    ) -> None:
        canvas.create_rectangle(
            x,
            y,
            x + w,
            y + h,
            fill=PANEL_BG,
            outline=node["color"],
            width=2,
        )
        canvas.create_rectangle(
            x + 10,
            y + 12,
            x + 22,
            y + h - 12,
            fill=node["color"],
            outline="",
        )
        canvas.create_text(
            x + 34,
            y + 22,
            text=node["title"],
            fill=TEXT,
            font=("Consolas", 14, "bold"),
            anchor="w",
        )
        canvas.create_text(
            x + 34,
            y + 45,
            text=node["subtitle"],
            fill=node["color"],
            font=("Consolas", 11, "bold"),
            anchor="w",
        )
        canvas.create_text(
            x + 34,
            y + 68,
            text=node["detail"],
            fill=MUTED,
            font=("Consolas", 10),
            anchor="w",
        )

    def _draw_arrow(
        self,
        canvas: tk.Canvas,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        label: str,
    ) -> None:
        canvas.create_line(x1, y1, x2, y2, fill=color, width=3, arrow="last", smooth=True)
        canvas.create_text(
            (x1 + x2) / 2,
            ((y1 + y2) / 2) - 12,
            text=label,
            fill=color,
            font=("Consolas", 10, "bold"),
        )

    def refresh(self) -> None:
        snapshot = self.reader.read()

        status_color = {
            "LIVE": GOOD,
            "STALE": WARN,
            "WAITING": ACCENT,
            "STOPPED": MUTED,
            "ERROR": "#f97316",
            "NO TENSORBOARD": BAD,
        }.get(snapshot.status, TEXT)
        self.status_label.configure(text=snapshot.status, fg=status_color)

        for key, label in self.badges.items():
            label.configure(text=self._badge_text(key, snapshot))

        detail = snapshot.detail
        if snapshot.event_path is not None:
            detail = f"{detail}\n{snapshot.event_path}"
        self.detail_label.configure(text=detail)

        age_text = "--"
        if snapshot.event_age_s is not None:
            age_text = _format_duration(snapshot.event_age_s)
        self.footer_label.configure(
            text=(
                f"Policy: {_policy_name(self.cfg)}   "
                f"Actions: {_action_text(self.cfg.env.action_grid)}   "
                f"Age: {age_text}   ESC to close"
            )
        )
        self._draw_architecture(snapshot)
        self.root.after(self.interval_ms, self.refresh)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a stream-friendly Irisu RL dashboard")
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
    parser.add_argument(
        "--view",
        choices=("metrics", "network"),
        default="metrics",
        help="Choose the original metrics HUD or the neural-network architecture view",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = load_config(args.config)
    run_dir = _resolve_run_dir(args.run_dir)
    reader = CompositeMetricsReader(run_dir)
    dashboard_cls = MetricsDashboardWindow if args.view == "metrics" else NetworkDashboardWindow
    dashboard = dashboard_cls(
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
