from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(slots=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_mss_monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rect":
        return cls(
            left=int(data["left"]),
            top=int(data["top"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass(slots=True)
class ActionGridConfig:
    rows: int = 8
    cols: int = 8
    left: int = 0
    top: int = 0
    right: int = 640
    bottom: int = 480

    @property
    def grid_size(self) -> int:
        return self.rows * self.cols

    @property
    def action_count(self) -> int:
        # 0 = no-op, 1..N = left-click cells, N+1..2N = right-click cells
        return 1 + (2 * self.grid_size)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionGridConfig":
        return cls(
            rows=int(data.get("rows", cls.rows)),
            cols=int(data.get("cols", cls.cols)),
            left=int(data.get("left", cls.left)),
            top=int(data.get("top", cls.top)),
            right=int(data.get("right", cls.right)),
            bottom=int(data.get("bottom", cls.bottom)),
        )


@dataclass(slots=True)
class ScoreOCRConfig:
    enabled: bool = False
    region: Rect | None = None
    tesseract_cmd: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScoreOCRConfig":
        if not data:
            return cls()
        region_raw = data.get("region")
        region = Rect.from_dict(region_raw) if region_raw else None
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            region=region,
            tesseract_cmd=data.get("tesseract_cmd"),
        )


def _parse_hsv_triplet(
    value: list[int] | tuple[int, int, int] | None,
    default: tuple[int, int, int],
) -> tuple[int, int, int]:
    if value is None:
        return default
    if len(value) != 3:
        raise ValueError(f"Expected HSV triplet with length=3, got: {value!r}")
    return int(value[0]), int(value[1]), int(value[2])


@dataclass(slots=True)
class HealthBarConfig:
    enabled: bool = False
    region: Rect | None = None
    method: str = "profile"
    scanline_start_x: int | None = None
    scanline_end_x: int | None = None
    scanline_y: int | None = None
    scanline_half_height: int = 1
    scanline_contrast_threshold: float = 0.02
    hsv_lower_1: tuple[int, int, int] = (0, 70, 40)
    hsv_upper_1: tuple[int, int, int] = (12, 255, 255)
    hsv_lower_2: tuple[int, int, int] = (170, 70, 40)
    hsv_upper_2: tuple[int, int, int] = (179, 255, 255)
    column_fill_threshold: float = 0.08
    adaptive_fill_peak_ratio: float = 0.55
    min_visible_pixels: int = 200
    fill_direction: str = "left_to_right"
    invert_percent: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HealthBarConfig":
        if not data:
            return cls()
        region_raw = data.get("region")
        region = Rect.from_dict(region_raw) if region_raw else None
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            region=region,
            method=str(data.get("method", cls.method)),
            scanline_start_x=(
                int(data["scanline_start_x"]) if data.get("scanline_start_x") is not None else None
            ),
            scanline_end_x=(
                int(data["scanline_end_x"]) if data.get("scanline_end_x") is not None else None
            ),
            scanline_y=int(data["scanline_y"]) if data.get("scanline_y") is not None else None,
            scanline_half_height=int(
                data.get("scanline_half_height", cls.scanline_half_height)
            ),
            scanline_contrast_threshold=float(
                data.get("scanline_contrast_threshold", cls.scanline_contrast_threshold)
            ),
            hsv_lower_1=_parse_hsv_triplet(data.get("hsv_lower_1"), cls.hsv_lower_1),
            hsv_upper_1=_parse_hsv_triplet(data.get("hsv_upper_1"), cls.hsv_upper_1),
            hsv_lower_2=_parse_hsv_triplet(data.get("hsv_lower_2"), cls.hsv_lower_2),
            hsv_upper_2=_parse_hsv_triplet(data.get("hsv_upper_2"), cls.hsv_upper_2),
            column_fill_threshold=float(
                data.get("column_fill_threshold", cls.column_fill_threshold)
            ),
            adaptive_fill_peak_ratio=float(
                data.get("adaptive_fill_peak_ratio", cls.adaptive_fill_peak_ratio)
            ),
            min_visible_pixels=int(data.get("min_visible_pixels", cls.min_visible_pixels)),
            fill_direction=str(data.get("fill_direction", cls.fill_direction)),
            invert_percent=bool(data.get("invert_percent", cls.invert_percent)),
        )


@dataclass(slots=True)
class RewardConfig:
    survival_reward: float = 0.01
    activity_reward_scale: float = 0.15
    cascade_threshold: float = 0.18
    cascade_bonus: float = 0.25
    stale_threshold: float = 0.01
    stale_patience: int = 90
    stale_penalty: float = -0.25
    score_delta_scale: float = 0.001

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewardConfig":
        if not data:
            return cls()
        return cls(
            survival_reward=float(data.get("survival_reward", cls.survival_reward)),
            activity_reward_scale=float(data.get("activity_reward_scale", cls.activity_reward_scale)),
            cascade_threshold=float(data.get("cascade_threshold", cls.cascade_threshold)),
            cascade_bonus=float(data.get("cascade_bonus", cls.cascade_bonus)),
            stale_threshold=float(data.get("stale_threshold", cls.stale_threshold)),
            stale_patience=int(data.get("stale_patience", cls.stale_patience)),
            stale_penalty=float(data.get("stale_penalty", cls.stale_penalty)),
            score_delta_scale=float(data.get("score_delta_scale", cls.score_delta_scale)),
        )


@dataclass(slots=True)
class EpisodeConfig:
    max_steps: int = 5000
    action_repeat: int = 1
    click_hold_s: float = 0.01
    max_clicks_per_second: float = 3.0
    inter_step_sleep_s: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EpisodeConfig":
        if not data:
            return cls()
        return cls(
            max_steps=int(data.get("max_steps", cls.max_steps)),
            action_repeat=int(data.get("action_repeat", cls.action_repeat)),
            click_hold_s=float(data.get("click_hold_s", cls.click_hold_s)),
            max_clicks_per_second=float(
                data.get("max_clicks_per_second", cls.max_clicks_per_second)
            ),
            inter_step_sleep_s=float(data.get("inter_step_sleep_s", cls.inter_step_sleep_s)),
        )


@dataclass(slots=True)
class ResetMacroStep:
    kind: str
    duration_s: float = 0.0
    x: int | None = None
    y: int | None = None
    button: str = "left"
    key: str | None = None
    relative_to_capture: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResetMacroStep":
        return cls(
            kind=str(data["kind"]),
            duration_s=float(data.get("duration_s", 0.0)),
            x=int(data["x"]) if data.get("x") is not None else None,
            y=int(data["y"]) if data.get("y") is not None else None,
            button=str(data.get("button", "left")),
            key=data.get("key"),
            relative_to_capture=bool(data.get("relative_to_capture", cls.relative_to_capture)),
        )


@dataclass(slots=True)
class WindowConfig:
    title_regex: str = "Irisu"
    window_index: int = 0
    capture_region: Rect | None = None
    focus_before_step: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WindowConfig":
        if not data:
            return cls()
        capture_raw = data.get("capture_region")
        return cls(
            title_regex=str(data.get("title_regex", cls.title_regex)),
            window_index=int(data.get("window_index", cls.window_index)),
            capture_region=Rect.from_dict(capture_raw) if capture_raw else None,
            focus_before_step=bool(data.get("focus_before_step", cls.focus_before_step)),
        )


@dataclass(slots=True)
class EnvConfig:
    backend: str = "mock"
    obs_width: int = 96
    obs_height: int = 96
    frame_stack: int = 4
    window: WindowConfig = field(default_factory=WindowConfig)
    action_grid: ActionGridConfig = field(default_factory=ActionGridConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    reset_macro: list[ResetMacroStep] = field(default_factory=list)
    score_ocr: ScoreOCRConfig = field(default_factory=ScoreOCRConfig)
    health_bar: HealthBarConfig = field(default_factory=HealthBarConfig)
    game_over_on_health_missing: bool = False
    health_missing_patience: int = 3
    game_over_template: str | None = None
    game_over_threshold: float = 0.9

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EnvConfig":
        if not data:
            return cls()
        reset_macro = [ResetMacroStep.from_dict(step) for step in data.get("reset_macro", [])]
        return cls(
            backend=str(data.get("backend", cls.backend)),
            obs_width=int(data.get("obs_width", cls.obs_width)),
            obs_height=int(data.get("obs_height", cls.obs_height)),
            frame_stack=int(data.get("frame_stack", cls.frame_stack)),
            window=WindowConfig.from_dict(data.get("window")),
            action_grid=ActionGridConfig.from_dict(data.get("action_grid", {})),
            reward=RewardConfig.from_dict(data.get("reward")),
            episode=EpisodeConfig.from_dict(data.get("episode")),
            reset_macro=reset_macro,
            score_ocr=ScoreOCRConfig.from_dict(data.get("score_ocr")),
            health_bar=HealthBarConfig.from_dict(data.get("health_bar")),
            game_over_on_health_missing=bool(
                data.get("game_over_on_health_missing", cls.game_over_on_health_missing)
            ),
            health_missing_patience=int(
                data.get("health_missing_patience", cls.health_missing_patience)
            ),
            game_over_template=data.get("game_over_template"),
            game_over_threshold=float(data.get("game_over_threshold", cls.game_over_threshold)),
        )


@dataclass(slots=True)
class TrainConfig:
    n_envs: int = 1
    total_timesteps: int = 1_000_000
    n_steps: int = 256
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    checkpoint_every: int = 100_000
    seed: int = 1
    device: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainConfig":
        if not data:
            return cls()
        return cls(
            n_envs=int(data.get("n_envs", cls.n_envs)),
            total_timesteps=int(data.get("total_timesteps", cls.total_timesteps)),
            n_steps=int(data.get("n_steps", cls.n_steps)),
            batch_size=int(data.get("batch_size", cls.batch_size)),
            learning_rate=float(data.get("learning_rate", cls.learning_rate)),
            gamma=float(data.get("gamma", cls.gamma)),
            gae_lambda=float(data.get("gae_lambda", cls.gae_lambda)),
            clip_range=float(data.get("clip_range", cls.clip_range)),
            ent_coef=float(data.get("ent_coef", cls.ent_coef)),
            vf_coef=float(data.get("vf_coef", cls.vf_coef)),
            max_grad_norm=float(data.get("max_grad_norm", cls.max_grad_norm)),
            checkpoint_every=int(data.get("checkpoint_every", cls.checkpoint_every)),
            seed=int(data.get("seed", cls.seed)),
            device=str(data.get("device", cls.device)),
        )


@dataclass(slots=True)
class RootConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    window_titles: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RootConfig":
        return cls(
            env=EnvConfig.from_dict(data.get("env")),
            train=TrainConfig.from_dict(data.get("train")),
            window_titles=[str(item) for item in data.get("window_titles", [])],
        )


def load_config(path: str | Path) -> RootConfig:
    path_obj = Path(path).expanduser().resolve()
    raw = tomllib.loads(path_obj.read_text(encoding="utf-8"))
    cfg = RootConfig.from_dict(raw)

    if cfg.env.game_over_template:
        cfg.env.game_over_template = str(Path(cfg.env.game_over_template).expanduser().resolve())

    return cfg
