from __future__ import annotations

import argparse
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from irisu_blackbox.config import load_config
from irisu_blackbox.factory import make_env_factory
from irisu_blackbox.hud import HUDState


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _score_text(score: int | None) -> str:
    if score is None:
        return "--"
    return f"{score:,}"


def _resize_for_preview(frame_bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Frame has invalid shape")
    scale = min(max_width / width, max_height / height)
    scale = max(scale, 0.01)
    out_w = max(1, int(width * scale))
    out_h = max(1, int(height * scale))
    return cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _ppm_photo(frame_bgr: np.ndarray) -> tk.PhotoImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    header = f"P6 {width} {height} 255 ".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


@dataclass(slots=True)
class DashboardStats:
    session_runtime_s: float
    round_runtime_s: float
    episode_count: int
    reset_count: int
    session_best_score: int
    round_best_score: int
    missing_streak: int
    state: str


class SessionTracker:
    def __init__(self, patience: int) -> None:
        self.patience = max(1, int(patience))
        self.session_start = time.monotonic()
        self.round_start = self.session_start
        self.episode_count = 0
        self.reset_count = 0
        self.session_best_score = 0
        self.round_best_score = 0
        self.missing_streak = 0
        self._round_live = False
        self._round_end_recorded = False

    def update(self, hud: HUDState, *, now: float | None = None) -> DashboardStats:
        now = time.monotonic() if now is None else now

        if hud.score is not None:
            self.session_best_score = max(self.session_best_score, int(hud.score))
            self.round_best_score = max(self.round_best_score, int(hud.score))

        if hud.health_visible is True:
            self.missing_streak = 0
            self._round_end_recorded = False
            if not self._round_live:
                self._round_live = True
                self.episode_count += 1
                self.round_start = now
                self.round_best_score = max(0, int(hud.score or 0))
            state = "LIVE"
            round_runtime_s = now - self.round_start
        else:
            self.missing_streak += 1
            round_runtime_s = now - self.round_start if self._round_live else 0.0
            if self._round_live and self.missing_streak >= self.patience:
                if not self._round_end_recorded:
                    self.reset_count += 1
                    self._round_end_recorded = True
                self._round_live = False
                self.round_best_score = 0
                state = "RESETTING"
            else:
                state = "WAITING" if not self._round_live else "TRANSITION"

        return DashboardStats(
            session_runtime_s=now - self.session_start,
            round_runtime_s=max(0.0, round_runtime_s),
            episode_count=self.episode_count,
            reset_count=self.reset_count,
            session_best_score=self.session_best_score,
            round_best_score=self.round_best_score,
            missing_streak=self.missing_streak,
            state=state,
        )


class DashboardWindow:
    def __init__(
        self,
        *,
        env,
        interval_s: float,
        patience: int,
        geometry: str,
        always_on_top: bool,
    ) -> None:
        self.env = env
        self.interval_ms = max(50, int(interval_s * 1000.0))
        self.tracker = SessionTracker(patience=patience)
        self.root = tk.Tk()
        self.root.title("Irisu Training HUD")
        self.root.geometry(geometry)
        self.root.configure(bg="#0b1020")
        self.root.attributes("-topmost", bool(always_on_top))
        self.root.resizable(False, False)
        self.root.bind("<Escape>", lambda _event: self.close())

        self.preview_photo: tk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        title_font = ("Consolas", 24, "bold")
        big_font = ("Consolas", 36, "bold")
        medium_font = ("Consolas", 18, "bold")
        small_font = ("Consolas", 14)

        header = tk.Frame(self.root, bg="#0b1020")
        header.pack(fill="x", padx=20, pady=(18, 8))

        tk.Label(
            header,
            text="IRISU TRAINING HUD",
            bg="#0b1020",
            fg="#f3f6ff",
            font=title_font,
            anchor="w",
        ).pack(fill="x")

        self.state_label = tk.Label(
            header,
            text="WAITING",
            bg="#0b1020",
            fg="#7dd3fc",
            font=medium_font,
            anchor="w",
        )
        self.state_label.pack(fill="x", pady=(6, 0))

        self.preview_label = tk.Label(
            self.root,
            bg="#111827",
            bd=0,
            highlightthickness=1,
            highlightbackground="#23304d",
        )
        self.preview_label.pack(fill="x", padx=20, pady=(8, 18))

        stats = tk.Frame(self.root, bg="#0b1020")
        stats.pack(fill="both", expand=True, padx=20)

        self.score_value = tk.Label(
            stats,
            text="--",
            bg="#0b1020",
            fg="#f8fafc",
            font=big_font,
            anchor="w",
        )
        self._stat_row(stats, "SCORE", self.score_value, value_pady=(0, 18))

        self.health_value = tk.Label(
            stats,
            text="--",
            bg="#0b1020",
            fg="#f8fafc",
            font=big_font,
            anchor="w",
        )
        self._stat_row(stats, "HEALTH", self.health_value, value_pady=(0, 10))

        self.health_canvas = tk.Canvas(
            stats,
            width=420,
            height=28,
            bg="#101827",
            highlightthickness=1,
            highlightbackground="#23304d",
            bd=0,
        )
        self.health_canvas.pack(anchor="w", pady=(0, 22))

        grid = tk.Frame(stats, bg="#0b1020")
        grid.pack(fill="x", pady=(0, 18))

        self.runtime_value = self._kv_label(grid, "SESSION")
        self.round_value = self._kv_label(grid, "ROUND")
        self.episodes_value = self._kv_label(grid, "EPISODES")
        self.resets_value = self._kv_label(grid, "RESETS")
        self.session_best_value = self._kv_label(grid, "BEST SCORE")
        self.round_best_value = self._kv_label(grid, "ROUND BEST")
        self.missing_value = self._kv_label(grid, "MISS STREAK")

        for idx, child in enumerate(grid.winfo_children()):
            child.grid(row=idx // 2, column=idx % 2, sticky="w", padx=(0, 28), pady=(0, 14))

        footer = tk.Label(
            self.root,
            text="ESC to close",
            bg="#0b1020",
            fg="#94a3b8",
            font=small_font,
            anchor="w",
        )
        footer.pack(fill="x", padx=20, pady=(0, 18))

    def _stat_row(
        self,
        parent: tk.Widget,
        title: str,
        value_widget: tk.Label,
        *,
        value_pady: tuple[int, int] = (0, 12),
    ) -> None:
        tk.Label(
            parent,
            text=title,
            bg="#0b1020",
            fg="#94a3b8",
            font=("Consolas", 14, "bold"),
            anchor="w",
        ).pack(fill="x")
        value_widget.pack(fill="x", pady=value_pady)

    def _kv_label(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg="#0b1020")
        tk.Label(
            frame,
            text=title,
            bg="#0b1020",
            fg="#94a3b8",
            font=("Consolas", 12, "bold"),
            anchor="w",
        ).pack(fill="x")
        value = tk.Label(
            frame,
            text="--",
            bg="#0b1020",
            fg="#f8fafc",
            font=("Consolas", 16, "bold"),
            anchor="w",
        )
        value.pack(fill="x", pady=(2, 0))
        frame.value_label = value  # type: ignore[attr-defined]
        return frame

    def _set_kv(self, holder: tk.Frame, text: str) -> None:
        holder.value_label.configure(text=text)  # type: ignore[attr-defined]

    def _draw_health_bar(self, percent: float | None, visible: bool | None) -> None:
        self.health_canvas.delete("all")
        width = int(self.health_canvas["width"])
        height = int(self.health_canvas["height"])
        self.health_canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")

        if visible is not True or percent is None:
            self.health_canvas.create_text(
                width // 2,
                height // 2,
                text="NO HEALTH BAR",
                fill="#fca5a5",
                font=("Consolas", 12, "bold"),
            )
            return

        pct = max(0.0, min(1.0, float(percent)))
        fill_w = int(width * pct)
        color = "#22c55e" if pct >= 0.6 else "#eab308" if pct >= 0.3 else "#ef4444"
        self.health_canvas.create_rectangle(0, 0, fill_w, height, fill=color, outline="")
        self.health_canvas.create_text(
            width // 2,
            height // 2,
            text=f"{pct * 100.0:5.1f}%",
            fill="#f8fafc",
            font=("Consolas", 12, "bold"),
        )

    def refresh(self) -> None:
        frame = self.env.backend.capture_frame()
        hud = self.env.hud_reader.read(frame)
        stats = self.tracker.update(hud)

        preview = _resize_for_preview(frame, max_width=440, max_height=340)
        self.preview_photo = _ppm_photo(preview)
        self.preview_label.configure(image=self.preview_photo)

        state_color = {
            "LIVE": "#4ade80",
            "RESETTING": "#f97316",
            "TRANSITION": "#facc15",
            "WAITING": "#7dd3fc",
        }.get(stats.state, "#e2e8f0")
        self.state_label.configure(text=stats.state, fg=state_color)
        self.score_value.configure(text=_score_text(hud.score))
        if hud.health_percent is None:
            self.health_value.configure(text="--")
        else:
            self.health_value.configure(text=f"{hud.health_percent * 100.0:5.1f}%")
        self._draw_health_bar(hud.health_percent, hud.health_visible)

        self._set_kv(self.runtime_value, _format_duration(stats.session_runtime_s))
        self._set_kv(self.round_value, _format_duration(stats.round_runtime_s))
        self._set_kv(self.episodes_value, str(stats.episode_count))
        self._set_kv(self.resets_value, str(stats.reset_count))
        self._set_kv(self.session_best_value, _score_text(stats.session_best_score))
        self._set_kv(self.round_best_value, _score_text(stats.round_best_score))
        self._set_kv(self.missing_value, str(stats.missing_streak))

        self.root.after(self.interval_ms, self.refresh)

    def run(self) -> None:
        self.refresh()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.root.destroy()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a stream-friendly Irisu training HUD window")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("--window-title", type=str, default=None)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--geometry", type=str, default="480x1080+0+0")
    parser.add_argument("--topmost", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = load_config(args.config)
    cfg.env.window.focus_before_step = False

    window_titles = [args.window_title] if args.window_title else None
    env = make_env_factory(cfg, rank=0, seed=cfg.train.seed, window_titles=window_titles)()

    patience = args.patience if args.patience is not None else cfg.env.health_missing_patience
    dashboard = DashboardWindow(
        env=env,
        interval_s=args.interval_s,
        patience=patience,
        geometry=args.geometry,
        always_on_top=args.topmost,
    )
    dashboard.run()


if __name__ == "__main__":
    main()
