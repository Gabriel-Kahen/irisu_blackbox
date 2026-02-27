from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from irisu_blackbox.config import load_config
from irisu_blackbox.factory import make_env_factory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trained policy in Irisu black-box env")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--window-title", type=str, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = load_config(args.config)
    if args.window_title:
        window_titles = [args.window_title]
    else:
        window_titles = None

    env_fn = make_env_factory(root_cfg=cfg, rank=0, seed=cfg.train.seed, window_titles=window_titles)
    vec_env = DummyVecEnv([env_fn])
    model = RecurrentPPO.load(str(args.model), env=vec_env, device=cfg.train.device)

    for episode in range(args.episodes):
        obs = vec_env.reset()
        done = np.array([False], dtype=bool)
        episode_starts = np.ones((1,), dtype=bool)
        lstm_states = None
        episode_reward = 0.0

        while not done[0]:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            obs, rewards, dones, infos = vec_env.step(action)
            done = dones
            episode_starts = dones
            episode_reward += float(rewards[0])

        print(f"episode={episode + 1} reward={episode_reward:.3f}")

    vec_env.close()


if __name__ == "__main__":
    main()
