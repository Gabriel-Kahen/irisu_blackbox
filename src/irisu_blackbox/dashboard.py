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
PANEL_SOFT = "#162238"
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


def _format_update_age(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    if seconds < 10.0:
        return f"{seconds:.1f}s"
    if seconds < 60.0:
        return f"{int(seconds)}s"
    return _format_duration(seconds)


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


def _format_rate(value: float | int | None) -> str:
    if value is None:
        return "--"
    rate = float(value)
    if abs(rate) >= 100:
        return _format_compact_int(rate)
    if abs(rate) >= 10:
        return f"{rate:.1f}"
    if abs(rate) >= 1:
        return f"{rate:.2f}"
    return f"{rate:.3f}"


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


def _action_output_info(
    action: int | None,
    grid: ActionGridConfig,
) -> tuple[str, int | None, int | None, str]:
    if action is None:
        return "waiting", None, None, "NONE"
    if action == 0:
        return "noop", None, None, "NO-OP"

    grid_size = grid.grid_size
    if action < 0 or action >= grid.action_count:
        return "unknown", None, None, "UNKNOWN"

    button = "LEFT" if action <= grid_size else "RIGHT"
    offset = action - 1
    if action > grid_size:
        offset -= grid_size
    row = offset // grid.cols
    col = offset % grid.cols
    return button.lower(), row, col, f"{button} R{row + 1} C{col + 1}"


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
    hud: dict[str, float | int | bool | None]
    control: dict[str, float | int | bool | None]
    updated_at_s: float | None
    elapsed_time_s: float | None
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
                hud={},
                control={},
                updated_at_s=None,
                elapsed_time_s=None,
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
                hud={},
                control={},
                updated_at_s=None,
                elapsed_time_s=None,
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
                hud={},
                control={},
                updated_at_s=None,
                elapsed_time_s=None,
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
                hud={},
                control={},
                updated_at_s=None,
                elapsed_time_s=None,
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
            hud={},
            control={},
            updated_at_s=latest_wall_time,
            elapsed_time_s=None,
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
        elapsed_raw = payload.get("elapsed_time_s")
        try:
            elapsed_time_s = float(elapsed_raw) if elapsed_raw is not None else None
        except (TypeError, ValueError):
            elapsed_time_s = None
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
        hud_raw = payload.get("hud", {})
        hud = dict(hud_raw) if isinstance(hud_raw, dict) else {}
        control_raw = payload.get("control", {})
        control = dict(control_raw) if isinstance(control_raw, dict) else {}
        return TrainingSnapshot(
            metrics={
                str(key): float(value)
                for key, value in dict(payload.get("metrics", {})).items()
                if value is not None
            },
            hud=hud,
            control=control,
            updated_at_s=updated_at_f,
            elapsed_time_s=elapsed_time_s,
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
        self._last_speed_sample: tuple[float, float] | None = None
        self._smoothed_train_speed: float | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        title_font = ("Bahnschrift SemiBold", 25, "bold")
        hero_font = ("Bahnschrift SemiBold", 30, "bold")
        metric_title_font = ("Consolas", 11, "bold")
        metric_value_font = ("Bahnschrift SemiBold", 18, "bold")
        small_font = ("Consolas", 11)

        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True, padx=22, pady=20)

        header_row = tk.Frame(shell, bg=BG)
        header_row.pack(fill="x", pady=(0, 14))

        title_block = tk.Frame(header_row, bg=BG)
        title_block.pack(side="left", fill="x", expand=True)

        tk.Label(
            title_block,
            text="IRISU RL DASHBOARD",
            bg=BG,
            fg=TEXT,
            font=title_font,
            anchor="w",
        ).pack(fill="x")

        self.status_label = tk.Label(
            header_row,
            text="WAITING",
            bg=PANEL_BG,
            fg=ACCENT,
            font=("Consolas", 12, "bold"),
            anchor="center",
            padx=14,
            pady=8,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            bd=0,
        )
        self.status_label.pack(side="right")

        hero_grid = tk.Frame(shell, bg=BG)
        hero_grid.pack(fill="x", pady=(0, 12))
        self.hero_boxes: dict[str, tk.Label] = {}
        self.hero_cards: dict[str, tk.Frame] = {}
        hero_spec = {
            "score": {"title": "SCORE", "accent": PANEL_BORDER, "bg": PANEL_BG},
            "health": {"title": "HEALTH", "accent": PANEL_BORDER, "bg": PANEL_BG},
            "high_score": {"title": "BEST SCORE", "accent": PANEL_BORDER, "bg": PANEL_BG},
            "games_played": {"title": "GAMES PLAYED", "accent": PANEL_BORDER, "bg": PANEL_BG},
        }
        self.health_meter_fill: tk.Frame | None = None
        for idx, (key, title) in enumerate(
            [
                ("score", "SCORE"),
                ("health", "HEALTH"),
                ("high_score", "BEST SCORE"),
                ("games_played", "GAMES PLAYED"),
            ]
        ):
            block = tk.Frame(
                hero_grid,
                bg=hero_spec[key]["bg"],
                highlightthickness=1,
                highlightbackground=hero_spec[key]["accent"],
                bd=0,
            )
            row = idx // 2
            col = idx % 2
            block.grid(row=row, column=col, sticky="nsew", padx=(0, 12 if col == 0 else 0), pady=(0, 12))
            hero_grid.grid_columnconfigure(col, weight=1)

            accent = hero_spec[key]["accent"]
            tk.Frame(block, bg=accent, height=4).pack(fill="x")
            tk.Label(
                block,
                text=title,
                bg=hero_spec[key]["bg"],
                fg=MUTED,
                font=metric_title_font,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 2))
            value = tk.Label(
                block,
                text="--",
                bg=hero_spec[key]["bg"],
                fg=TEXT,
                font=hero_font,
                anchor="w",
            )
            value.pack(fill="x", padx=12, pady=(0, 10))
            if key == "health":
                meter_track = tk.Frame(block, bg="#203043", height=8)
                meter_track.pack(fill="x", padx=12, pady=(0, 12))
                meter_fill = tk.Frame(meter_track, bg="#7c8aa0", height=8)
                meter_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)
                self.health_meter_fill = meter_fill
            self.hero_boxes[key] = value
            self.hero_cards[key] = block

        grid = tk.Frame(shell, bg=BG)
        grid.pack(fill="x", pady=(0, 12))

        self.metric_boxes: dict[str, tk.Label] = {}
        metric_layout = [
            ("timesteps", "TIMESTEPS"),
            ("fps", "TRAIN SPEED"),
            ("ep_rew_mean", "AVG REWARD"),
            ("time_running", "TIME RUNNING"),
        ]
        for idx, (key, title) in enumerate(metric_layout):
            box = tk.Frame(
                grid,
                bg=PANEL_SOFT,
                highlightthickness=1,
                highlightbackground=PANEL_BORDER,
                bd=0,
            )
            box.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=(0, 12 if (idx % 2) == 0 else 0), pady=(0, 12))
            grid.grid_columnconfigure(idx % 2, weight=1)

            tk.Label(
                box,
                text=title,
                bg=PANEL_SOFT,
                fg=MUTED,
                font=metric_title_font,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 2))
            value_label = tk.Label(
                box,
                text="--",
                bg=PANEL_SOFT,
                fg=TEXT,
                font=metric_value_font,
                anchor="w",
            )
            value_label.pack(fill="x", padx=12, pady=(0, 10))
            self.metric_boxes[key] = value_label

        network_box = tk.Frame(
            shell,
            bg="#0f172a",
            highlightthickness=1,
            highlightbackground="#2b4263",
            bd=0,
        )
        network_box.pack(fill="both", expand=True, pady=(0, 14))
        tk.Label(
            network_box,
            text="LIVE OUTPUT",
            bg="#0f172a",
            fg=MUTED,
            font=metric_title_font,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        self.network_canvas = tk.Canvas(
            network_box,
            bg="#0f172a",
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.network_canvas.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.network_canvas.bind("<Configure>", lambda _event: self._draw_stream_policy_map(None))

    def _metric_text(self, key: str, metrics: dict[str, float]) -> str:
        value = metrics.get(key)
        if key == "timesteps":
            return _format_compact_int(value)
        if key == "fps":
            return _format_rate(value)
        if key == "ep_rew_mean":
            return _format_float(value, digits=3)
        return "--"

    def _live_train_speed(self, snapshot: TrainingSnapshot) -> float | None:
        updated_at_s = snapshot.updated_at_s
        timesteps = snapshot.metrics.get("timesteps")
        if updated_at_s is None or timesteps is None:
            return self._smoothed_train_speed

        sample = (float(updated_at_s), float(timesteps))
        previous = self._last_speed_sample
        self._last_speed_sample = sample
        if previous is None:
            return snapshot.metrics.get("fps")

        prev_time, prev_steps = previous
        delta_t = sample[0] - prev_time
        delta_steps = sample[1] - prev_steps
        if delta_t <= 0:
            return self._smoothed_train_speed if self._smoothed_train_speed is not None else snapshot.metrics.get("fps")
        if delta_steps < 0:
            self._smoothed_train_speed = None
            return snapshot.metrics.get("fps")

        inst_speed = delta_steps / delta_t
        if self._smoothed_train_speed is None:
            self._smoothed_train_speed = inst_speed
        else:
            self._smoothed_train_speed = (0.65 * self._smoothed_train_speed) + (0.35 * inst_speed)
        return self._smoothed_train_speed

    def _draw_stream_policy_map(self, snapshot: TrainingSnapshot | None) -> None:
        canvas = self.network_canvas
        canvas.delete("all")
        width = max(360, canvas.winfo_width())
        height = max(340, canvas.winfo_height())

        canvas.create_rectangle(0, 0, width, height, fill="#0d1424", outline="")

        last_action = None
        if snapshot and snapshot.control.get("last_action") is not None:
            last_action_at_raw = snapshot.control.get("last_action_at")
            try:
                last_action_at = float(last_action_at_raw) if last_action_at_raw is not None else None
            except (TypeError, ValueError):
                last_action_at = None
            if last_action_at is not None and (time.time() - last_action_at) <= 0.5:
                last_action = int(snapshot.control["last_action"])
        action_kind, action_row, action_col, action_label = _action_output_info(last_action, self.cfg.env.action_grid)
        output_size = min(width - 36, height - 76)
        output_x = (width - output_size) / 2
        output_y = 8

        self._draw_stream_output_grid(
            canvas,
            x=output_x,
            y=output_y,
            size=output_size,
            rows=self.cfg.env.action_grid.rows,
            cols=self.cfg.env.action_grid.cols,
            action_kind=action_kind,
            action_row=action_row,
            action_col=action_col,
            action_label=action_label,
        )

    def _draw_stream_output_grid(
        self,
        canvas: tk.Canvas,
        *,
        x: float,
        y: float,
        size: float,
        rows: int,
        cols: int,
        action_kind: str,
        action_row: int | None,
        action_col: int | None,
        action_label: str,
    ) -> None:
        panel_h = size + 58
        board_pad = 10
        board_size = size - (board_pad * 2)
        gap = max(2, int(board_size * 0.012))
        total_gap_x = gap * (cols - 1)
        total_gap_y = gap * (rows - 1)
        cell_w = (board_size - total_gap_x) / cols
        cell_h = (board_size - total_gap_y) / rows
        grid_top = y + 10
        grid_left = x + board_pad

        canvas.create_rectangle(x + 6, y + 8, x + size + 6, y + panel_h + 8, fill="#08111f", outline="")
        canvas.create_rectangle(x, y, x + size, y + panel_h, fill="#101a2e", outline="#35507a", width=2)
        for row in range(rows):
            for col in range(cols):
                x0 = grid_left + (col * (cell_w + gap))
                y0 = grid_top + (row * (cell_h + gap))
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                fill = "#20283a"
                outline = "#32435f"
                width = 1
                if action_row == row and action_col == col:
                    is_left = action_kind == "left"
                    fill = "#2d7ef7" if is_left else "#e38b14"
                    outline = "#dff7ff" if is_left else "#fff1bf"
                    width = 3
                canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=width)
                if action_row == row and action_col == col:
                    letter = "L" if action_kind == "left" else "R" if action_kind == "right" else ""
                    if letter:
                        canvas.create_text(
                            (x0 + x1) / 2,
                            (y0 + y1) / 2,
                            text=letter,
                            fill=TEXT,
                            font=("Bahnschrift SemiBold", max(11, int(cell_h * 0.52)), "bold"),
                        )

        if action_kind == "noop":
            canvas.create_text(
                x + (size / 2),
                grid_top + (board_size / 2),
                text="NO-OP",
                fill=TEXT,
                font=("Consolas", 18, "bold"),
            )

        accent = "#7dd3fc" if action_kind == "left" else "#fbbf24" if action_kind == "right" else MUTED
        pill_w = min(220, size - 18)
        pill_x0 = x + ((size - pill_w) / 2)
        pill_y0 = y + size + 10
        canvas.create_rectangle(pill_x0 + 4, pill_y0 + 4, pill_x0 + pill_w + 4, pill_y0 + 36, fill="#09111f", outline="")
        canvas.create_rectangle(pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + 32, fill="#121f35", outline=accent, width=2)
        canvas.create_text(
            x + (size / 2),
            pill_y0 + 16,
            text=action_label,
            fill=accent,
            font=("Consolas", 14, "bold"),
        )

    def refresh(self) -> None:
        snapshot = self.reader.read()
        display_metrics = dict(snapshot.metrics)
        live_speed = self._live_train_speed(snapshot)
        if live_speed is not None:
            display_metrics["fps"] = live_speed

        status_color = {
            "LIVE": GOOD,
            "STALE": WARN,
            "WAITING": ACCENT,
            "STOPPED": MUTED,
            "ERROR": "#f97316",
            "NO TENSORBOARD": BAD,
        }.get(snapshot.status, "#e2e8f0")

        self.status_label.configure(text=snapshot.status, fg=status_color)
        self.hero_boxes["score"].configure(text=_format_compact_int(snapshot.hud.get("score")))
        self.hero_boxes["health"].configure(
            text=_format_percent(snapshot.hud.get("health_percent"), digits=1)
            if snapshot.hud.get("health_percent") is not None
            else "--"
        )
        self.hero_boxes["high_score"].configure(
            text=_format_compact_int(snapshot.metrics.get("high_score"))
        )
        self.hero_boxes["games_played"].configure(
            text=_format_compact_int(snapshot.metrics.get("games_played"))
        )

        health_value = snapshot.hud.get("health_percent")
        if self.health_meter_fill is not None:
            health_ratio = 0.0 if health_value is None else max(0.0, min(1.0, float(health_value)))
            self.health_meter_fill.place(x=0, y=0, relheight=1.0, relwidth=health_ratio)

        for key, label in self.metric_boxes.items():
            if key == "time_running":
                label.configure(
                    text=_format_duration(snapshot.elapsed_time_s)
                    if snapshot.elapsed_time_s is not None
                    else "--"
                )
            else:
                label.configure(text=self._metric_text(key, display_metrics))

        self._draw_stream_policy_map(snapshot)

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
            ("score", "SCORE"),
            ("health_percent", "HEALTH"),
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
        if key in {"timesteps", "iterations"}:
            return _format_compact_int(metrics.get(key))
        if key == "fps":
            return _format_rate(metrics.get(key))
        if key == "ep_rew_mean":
            return _format_float(metrics.get(key), digits=3)
        if key == "score":
            return _format_compact_int(snapshot.hud.get("score"))
        if key == "health_percent":
            return _format_percent(snapshot.hud.get("health_percent"), digits=1)
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
    parser.add_argument("--interval-s", type=float, default=0.2)
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
