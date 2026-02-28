from __future__ import annotations

import argparse
import math
from pathlib import Path

from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from irisu_blackbox.checkpoints import find_latest_resume_path
from irisu_blackbox.config import RootConfig, load_config
from irisu_blackbox.factory import make_env_factory
from irisu_blackbox.live_metrics import DashboardMetricsRecorder


def _parse_window_titles(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",")]
    return [item for item in items if item]


def _resolve_batch_size(requested: int, rollout_size: int) -> int:
    if requested <= rollout_size and (rollout_size % requested == 0):
        return requested

    if requested > rollout_size:
        return rollout_size

    candidates = [d for d in range(requested, 0, -1) if rollout_size % d == 0]
    return candidates[0] if candidates else math.gcd(rollout_size, requested)


def _make_vec_env(cfg: RootConfig, window_titles: list[str] | None):
    n_envs = cfg.train.n_envs
    env_fns = [
        make_env_factory(
            root_cfg=cfg,
            rank=idx,
            seed=cfg.train.seed,
            window_titles=window_titles,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    return VecMonitor(vec_env)


def _make_model(cfg: RootConfig, vec_env: VecMonitor, run_dir: Path) -> RecurrentPPO:
    rollout_size = cfg.train.n_steps * cfg.train.n_envs
    batch_size = _resolve_batch_size(cfg.train.batch_size, rollout_size)
    policy = (
        "MultiInputLstmPolicy"
        if isinstance(vec_env.observation_space, spaces.Dict)
        else "CnnLstmPolicy"
    )

    return RecurrentPPO(
        policy=policy,
        env=vec_env,
        learning_rate=cfg.train.learning_rate,
        n_steps=cfg.train.n_steps,
        batch_size=batch_size,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_range=cfg.train.clip_range,
        ent_coef=cfg.train.ent_coef,
        vf_coef=cfg.train.vf_coef,
        max_grad_norm=cfg.train.max_grad_norm,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
        seed=cfg.train.seed,
        device=cfg.train.device,
    )

def _load_or_make_model(
    cfg: RootConfig,
    vec_env: VecMonitor,
    run_dir: Path,
    resume_from: Path | None = None,
) -> tuple[RecurrentPPO, bool]:
    if resume_from is None:
        return _make_model(cfg, vec_env, run_dir=run_dir), False

    model = RecurrentPPO.load(str(resume_from), env=vec_env, device=cfg.train.device)
    model.tensorboard_log = str(run_dir / "tensorboard")
    model.verbose = 1
    return model, True


class DashboardMetricsCallback(BaseCallback):
    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.recorder = DashboardMetricsRecorder(run_dir)

    def _on_training_start(self) -> None:
        total_timesteps = getattr(self.model, "_total_timesteps", None)
        self.recorder.on_training_start(
            total_timesteps=int(total_timesteps) if total_timesteps is not None else None,
            n_envs=int(getattr(self.training_env, "num_envs", 1)),
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        logger_values = getattr(self.logger, "name_to_value", {})
        self.recorder.on_step(
            num_timesteps=int(self.num_timesteps),
            infos=infos,
            logger_values=logger_values,
        )
        return True

    def _on_rollout_end(self) -> None:
        self.recorder.on_rollout_end()
        logger_values = getattr(self.logger, "name_to_value", {})
        self.recorder.on_step(
            num_timesteps=int(self.num_timesteps),
            infos=self.locals.get("infos"),
            logger_values=logger_values,
            force=True,
        )

    def _on_training_end(self) -> None:
        logger_values = getattr(self.logger, "name_to_value", {})
        self.recorder.on_training_end(
            num_timesteps=int(self.num_timesteps),
            logger_values=logger_values,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RecurrentPPO on Irisu black-box env")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"))
    parser.add_argument(
        "--window-titles",
        type=str,
        default=None,
        help="Comma-separated window-title regex list to pin envs to specific windows",
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from a specific .zip checkpoint/model path",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume training from the latest checkpoint in --run-dir/checkpoints",
    )
    return parser


def _apply_overrides(cfg: RootConfig, args: argparse.Namespace) -> RootConfig:
    if args.total_timesteps is not None:
        cfg.train.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.train.n_envs = args.n_envs
    if args.device is not None:
        cfg.train.device = args.device
    return cfg


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.resume_from is not None and args.resume_latest:
        raise SystemExit("Use only one of --resume-from or --resume-latest")

    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    window_titles = _parse_window_titles(args.window_titles)

    run_dir = args.run_dir.expanduser().resolve()
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(cfg.train.seed)

    vec_env = _make_vec_env(cfg, window_titles=window_titles)
    resume_path: Path | None = None
    if args.resume_from is not None:
        resume_path = args.resume_from.expanduser().resolve()
        if not resume_path.exists():
            vec_env.close()
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
    elif args.resume_latest:
        resume_path = find_latest_resume_path(run_dir)
        if resume_path is None:
            vec_env.close()
            raise SystemExit(
                f"No checkpoint or final model found under: {run_dir / 'checkpoints'}"
            )

    if resume_path is not None:
        print(f"Resuming from {resume_path}")
    model, resumed = _load_or_make_model(cfg, vec_env, run_dir=run_dir, resume_from=resume_path)

    save_freq = max(1, cfg.train.checkpoint_every // cfg.train.n_envs)
    callback = CallbackList(
        [
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(checkpoints_dir),
                name_prefix="irisu_ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            DashboardMetricsCallback(run_dir=run_dir),
        ]
    )

    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=not resumed,
    )
    model.save(str(run_dir / "final_model"))
    vec_env.close()


if __name__ == "__main__":
    main()
