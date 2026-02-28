from __future__ import annotations

import copy
from typing import Callable

from irisu_blackbox.backends import MockBackendConfig, MockGameBackend, WindowBinding, WindowsGameBackend
from irisu_blackbox.config import EnvConfig, RootConfig
from irisu_blackbox.env import IrisuBlackBoxEnv


def _cfg_for_rank(
    root_cfg: RootConfig,
    rank: int,
    window_titles: list[str] | None = None,
) -> EnvConfig:
    cfg = copy.deepcopy(root_cfg.env)
    titles = window_titles if window_titles is not None else root_cfg.window_titles
    if titles and rank < len(titles):
        cfg.window.title_regex = titles[rank]
        cfg.window.window_index = 0
    else:
        cfg.window.window_index = rank
    return cfg


def _make_backend(cfg: EnvConfig, rank: int, seed: int):
    backend_name = cfg.backend.lower()
    if backend_name == "mock":
        if cfg.window.capture_region is not None:
            width = cfg.window.capture_region.width
            height = cfg.window.capture_region.height
        else:
            width = max(64, cfg.action_grid.right - cfg.action_grid.left)
            height = max(64, cfg.action_grid.bottom - cfg.action_grid.top)

        return MockGameBackend(
            MockBackendConfig(
                width=width,
                height=height,
                seed=seed + rank,
            )
        )

    if backend_name == "windows":
        binding = WindowBinding(
            title_regex=cfg.window.title_regex,
            window_index=cfg.window.window_index,
            capture_region=cfg.window.capture_region,
            focus_before_step=cfg.window.focus_before_step,
            relaunch_on_missing_window=cfg.window.relaunch_on_missing_window,
            relaunch_timeout_s=cfg.window.relaunch_timeout_s,
            relaunch_poll_s=cfg.window.relaunch_poll_s,
            launch_executable=cfg.window.launch_executable,
            launch_args=tuple(cfg.window.launch_args),
            launch_workdir=cfg.window.launch_workdir,
        )
        return WindowsGameBackend(binding=binding, reset_macro=cfg.reset_macro)

    raise ValueError(f"Unsupported backend: {cfg.backend!r}")


def make_env_factory(
    root_cfg: RootConfig,
    rank: int,
    seed: int,
    window_titles: list[str] | None = None,
) -> Callable[[], IrisuBlackBoxEnv]:
    def _make() -> IrisuBlackBoxEnv:
        cfg = _cfg_for_rank(root_cfg, rank=rank, window_titles=window_titles)
        backend = _make_backend(cfg, rank=rank, seed=seed)
        return IrisuBlackBoxEnv(cfg=cfg, backend=backend)

    return _make
