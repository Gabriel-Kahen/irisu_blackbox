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
        defaults = cls()
        return cls(
            rows=int(data.get("rows", defaults.rows)),
            cols=int(data.get("cols", defaults.cols)),
            left=int(data.get("left", defaults.left)),
            top=int(data.get("top", defaults.top)),
            right=int(data.get("right", defaults.right)),
            bottom=int(data.get("bottom", defaults.bottom)),
        )


@dataclass(slots=True)
class ScoreOCRConfig:
    enabled: bool = False
    region: Rect | None = None
    method: str = "tesseract"
    tesseract_cmd: str | None = None
    template_dir: str | None = None
    template_min_similarity: float = 0.32
    template_fallback_to_tesseract: bool = True
    template_expected_digits: int = 8
    template_inner_left: int = 0
    template_inner_right: int = 0
    monotonic_non_decreasing: bool = False
    score_smoothing_window: int = 5
    max_step_decrease: int = 1000
    outlier_confirm_frames: int = 2
    hold_last_value_when_missing: bool = True
    min_confidence: float = 40.0
    max_step_increase: int = 1000

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScoreOCRConfig":
        if not data:
            return cls()
        defaults = cls()
        region_raw = data.get("region")
        region = Rect.from_dict(region_raw) if region_raw else None
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            region=region,
            method=str(data.get("method", defaults.method)),
            tesseract_cmd=data.get("tesseract_cmd"),
            template_dir=data.get("template_dir"),
            template_min_similarity=float(
                data.get("template_min_similarity", defaults.template_min_similarity)
            ),
            template_fallback_to_tesseract=bool(
                data.get(
                    "template_fallback_to_tesseract",
                    defaults.template_fallback_to_tesseract,
                )
            ),
            template_expected_digits=int(
                data.get("template_expected_digits", defaults.template_expected_digits)
            ),
            template_inner_left=int(data.get("template_inner_left", defaults.template_inner_left)),
            template_inner_right=int(data.get("template_inner_right", defaults.template_inner_right)),
            monotonic_non_decreasing=bool(
                data.get("monotonic_non_decreasing", defaults.monotonic_non_decreasing)
            ),
            score_smoothing_window=int(
                data.get("score_smoothing_window", defaults.score_smoothing_window)
            ),
            max_step_decrease=int(data.get("max_step_decrease", defaults.max_step_decrease)),
            outlier_confirm_frames=int(
                data.get("outlier_confirm_frames", defaults.outlier_confirm_frames)
            ),
            hold_last_value_when_missing=bool(
                data.get("hold_last_value_when_missing", defaults.hold_last_value_when_missing)
            ),
            min_confidence=float(data.get("min_confidence", defaults.min_confidence)),
            max_step_increase=int(data.get("max_step_increase", defaults.max_step_increase)),
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
    smoothing_window: int = 5
    max_delta_per_step: float = 0.14
    outlier_confirm_frames: int = 2
    fill_direction: str = "left_to_right"
    invert_percent: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HealthBarConfig":
        if not data:
            return cls()
        defaults = cls()
        region_raw = data.get("region")
        region = Rect.from_dict(region_raw) if region_raw else None
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            region=region,
            method=str(data.get("method", defaults.method)),
            scanline_start_x=(
                int(data["scanline_start_x"]) if data.get("scanline_start_x") is not None else None
            ),
            scanline_end_x=(
                int(data["scanline_end_x"]) if data.get("scanline_end_x") is not None else None
            ),
            scanline_y=int(data["scanline_y"]) if data.get("scanline_y") is not None else None,
            scanline_half_height=int(
                data.get("scanline_half_height", defaults.scanline_half_height)
            ),
            scanline_contrast_threshold=float(
                data.get("scanline_contrast_threshold", defaults.scanline_contrast_threshold)
            ),
            hsv_lower_1=_parse_hsv_triplet(data.get("hsv_lower_1"), defaults.hsv_lower_1),
            hsv_upper_1=_parse_hsv_triplet(data.get("hsv_upper_1"), defaults.hsv_upper_1),
            hsv_lower_2=_parse_hsv_triplet(data.get("hsv_lower_2"), defaults.hsv_lower_2),
            hsv_upper_2=_parse_hsv_triplet(data.get("hsv_upper_2"), defaults.hsv_upper_2),
            column_fill_threshold=float(
                data.get("column_fill_threshold", defaults.column_fill_threshold)
            ),
            adaptive_fill_peak_ratio=float(
                data.get("adaptive_fill_peak_ratio", defaults.adaptive_fill_peak_ratio)
            ),
            min_visible_pixels=int(data.get("min_visible_pixels", defaults.min_visible_pixels)),
            smoothing_window=int(data.get("smoothing_window", defaults.smoothing_window)),
            max_delta_per_step=float(
                data.get("max_delta_per_step", defaults.max_delta_per_step)
            ),
            outlier_confirm_frames=int(
                data.get("outlier_confirm_frames", defaults.outlier_confirm_frames)
            ),
            fill_direction=str(data.get("fill_direction", defaults.fill_direction)),
            invert_percent=bool(data.get("invert_percent", defaults.invert_percent)),
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
    score_value_scale: float = 0.02
    score_value_log_max: float = 50000.0
    health_value_scale: float = 0.03
    health_delta_scale: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewardConfig":
        if not data:
            return cls()
        defaults = cls()
        return cls(
            survival_reward=float(data.get("survival_reward", defaults.survival_reward)),
            activity_reward_scale=float(data.get("activity_reward_scale", defaults.activity_reward_scale)),
            cascade_threshold=float(data.get("cascade_threshold", defaults.cascade_threshold)),
            cascade_bonus=float(data.get("cascade_bonus", defaults.cascade_bonus)),
            stale_threshold=float(data.get("stale_threshold", defaults.stale_threshold)),
            stale_patience=int(data.get("stale_patience", defaults.stale_patience)),
            stale_penalty=float(data.get("stale_penalty", defaults.stale_penalty)),
            score_delta_scale=float(data.get("score_delta_scale", defaults.score_delta_scale)),
            score_value_scale=float(data.get("score_value_scale", defaults.score_value_scale)),
            score_value_log_max=float(
                data.get("score_value_log_max", defaults.score_value_log_max)
            ),
            health_value_scale=float(data.get("health_value_scale", defaults.health_value_scale)),
            health_delta_scale=float(data.get("health_delta_scale", defaults.health_delta_scale)),
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
        defaults = cls()
        return cls(
            max_steps=int(data.get("max_steps", defaults.max_steps)),
            action_repeat=int(data.get("action_repeat", defaults.action_repeat)),
            click_hold_s=float(data.get("click_hold_s", defaults.click_hold_s)),
            max_clicks_per_second=float(
                data.get("max_clicks_per_second", defaults.max_clicks_per_second)
            ),
            inter_step_sleep_s=float(data.get("inter_step_sleep_s", defaults.inter_step_sleep_s)),
        )


@dataclass(slots=True)
class HUDFeatureConfig:
    enabled: bool = True
    score_log_max: float = 50000.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HUDFeatureConfig":
        if not data:
            return cls()
        defaults = cls()
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            score_log_max=float(data.get("score_log_max", defaults.score_log_max)),
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
        defaults = cls(kind=str(data["kind"]))
        return cls(
            kind=str(data["kind"]),
            duration_s=float(data.get("duration_s", 0.0)),
            x=int(data["x"]) if data.get("x") is not None else None,
            y=int(data["y"]) if data.get("y") is not None else None,
            button=str(data.get("button", "left")),
            key=data.get("key"),
            relative_to_capture=bool(data.get("relative_to_capture", defaults.relative_to_capture)),
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
        defaults = cls()
        capture_raw = data.get("capture_region")
        return cls(
            title_regex=str(data.get("title_regex", defaults.title_regex)),
            window_index=int(data.get("window_index", defaults.window_index)),
            capture_region=Rect.from_dict(capture_raw) if capture_raw else None,
            focus_before_step=bool(data.get("focus_before_step", defaults.focus_before_step)),
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
    hud_features: HUDFeatureConfig = field(default_factory=HUDFeatureConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    game_over_macro: list[ResetMacroStep] = field(default_factory=list)
    reset_macro: list[ResetMacroStep] = field(default_factory=list)
    score_ocr: ScoreOCRConfig = field(default_factory=ScoreOCRConfig)
    health_bar: HealthBarConfig = field(default_factory=HealthBarConfig)
    game_over_on_health_missing: bool = False
    health_missing_patience: int = 3
    action_pause_on_health_missing_s: float = 1.0
    game_over_template: str | None = None
    game_over_threshold: float = 0.9
    post_game_over_delay_s: float = 1.0
    reset_ready_template: str | None = None
    reset_ready_threshold: float = 0.92
    reset_ready_timeout_s: float = 6.0
    reset_ready_poll_s: float = 0.05
    round_start_timeout_s: float = 6.0
    round_start_poll_s: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EnvConfig":
        if not data:
            return cls()
        defaults = cls()
        game_over_macro = [
            ResetMacroStep.from_dict(step) for step in data.get("game_over_macro", [])
        ]
        reset_macro = [ResetMacroStep.from_dict(step) for step in data.get("reset_macro", [])]
        return cls(
            backend=str(data.get("backend", defaults.backend)),
            obs_width=int(data.get("obs_width", defaults.obs_width)),
            obs_height=int(data.get("obs_height", defaults.obs_height)),
            frame_stack=int(data.get("frame_stack", defaults.frame_stack)),
            window=WindowConfig.from_dict(data.get("window")),
            action_grid=ActionGridConfig.from_dict(data.get("action_grid", {})),
            reward=RewardConfig.from_dict(data.get("reward")),
            hud_features=HUDFeatureConfig.from_dict(data.get("hud_features")),
            episode=EpisodeConfig.from_dict(data.get("episode")),
            game_over_macro=game_over_macro,
            reset_macro=reset_macro,
            score_ocr=ScoreOCRConfig.from_dict(data.get("score_ocr")),
            health_bar=HealthBarConfig.from_dict(data.get("health_bar")),
            game_over_on_health_missing=bool(
                data.get("game_over_on_health_missing", defaults.game_over_on_health_missing)
            ),
            health_missing_patience=int(
                data.get("health_missing_patience", defaults.health_missing_patience)
            ),
            action_pause_on_health_missing_s=float(
                data.get(
                    "action_pause_on_health_missing_s",
                    defaults.action_pause_on_health_missing_s,
                )
            ),
            game_over_template=data.get("game_over_template"),
            game_over_threshold=float(data.get("game_over_threshold", defaults.game_over_threshold)),
            post_game_over_delay_s=float(
                data.get("post_game_over_delay_s", defaults.post_game_over_delay_s)
            ),
            reset_ready_template=data.get("reset_ready_template"),
            reset_ready_threshold=float(
                data.get("reset_ready_threshold", defaults.reset_ready_threshold)
            ),
            reset_ready_timeout_s=float(
                data.get("reset_ready_timeout_s", defaults.reset_ready_timeout_s)
            ),
            reset_ready_poll_s=float(data.get("reset_ready_poll_s", defaults.reset_ready_poll_s)),
            round_start_timeout_s=float(
                data.get("round_start_timeout_s", defaults.round_start_timeout_s)
            ),
            round_start_poll_s=float(
                data.get("round_start_poll_s", defaults.round_start_poll_s)
            ),
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
        defaults = cls()
        return cls(
            n_envs=int(data.get("n_envs", defaults.n_envs)),
            total_timesteps=int(data.get("total_timesteps", defaults.total_timesteps)),
            n_steps=int(data.get("n_steps", defaults.n_steps)),
            batch_size=int(data.get("batch_size", defaults.batch_size)),
            learning_rate=float(data.get("learning_rate", defaults.learning_rate)),
            gamma=float(data.get("gamma", defaults.gamma)),
            gae_lambda=float(data.get("gae_lambda", defaults.gae_lambda)),
            clip_range=float(data.get("clip_range", defaults.clip_range)),
            ent_coef=float(data.get("ent_coef", defaults.ent_coef)),
            vf_coef=float(data.get("vf_coef", defaults.vf_coef)),
            max_grad_norm=float(data.get("max_grad_norm", defaults.max_grad_norm)),
            checkpoint_every=int(data.get("checkpoint_every", defaults.checkpoint_every)),
            seed=int(data.get("seed", defaults.seed)),
            device=str(data.get("device", defaults.device)),
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


def _resolve_optional_path(
    value: str | None,
    *,
    config_path: Path,
    expect_dir: bool,
) -> str | None:
    if not value:
        return value

    raw = Path(value).expanduser()
    if raw.is_absolute():
        return str(raw.resolve())

    candidates = [
        (Path.cwd() / raw).resolve(),
        (config_path.parent / raw).resolve(),
    ]
    if config_path.parent.name.lower() in {"config", "configs"}:
        candidates.append((config_path.parent.parent / raw).resolve())

    for candidate in candidates:
        if expect_dir and candidate.is_dir():
            return str(candidate)
        if not expect_dir and candidate.exists():
            return str(candidate)

    # Fall back to config-relative path when nothing exists yet.
    return str((config_path.parent / raw).resolve())


def load_config(path: str | Path) -> RootConfig:
    path_obj = Path(path).expanduser().resolve()
    raw = tomllib.loads(path_obj.read_text(encoding="utf-8"))
    cfg = RootConfig.from_dict(raw)

    cfg.env.game_over_template = _resolve_optional_path(
        cfg.env.game_over_template,
        config_path=path_obj,
        expect_dir=False,
    )
    cfg.env.reset_ready_template = _resolve_optional_path(
        cfg.env.reset_ready_template,
        config_path=path_obj,
        expect_dir=False,
    )
    cfg.env.score_ocr.template_dir = _resolve_optional_path(
        cfg.env.score_ocr.template_dir,
        config_path=path_obj,
        expect_dir=True,
    )

    return cfg
