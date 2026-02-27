from __future__ import annotations

import argparse
import time
from pathlib import Path

from irisu_blackbox.config import load_config
from irisu_blackbox.factory import make_env_factory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously read score/health HUD values from the configured game window"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("--window-title", type=str, default=None)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    return parser


def _print_score_template_status(cfg_path: Path, method: str, template_dir: str | None) -> None:
    method_key = method.strip().lower()
    if method_key not in {"template", "auto"}:
        return

    if not template_dir:
        print(f"[score_ocr] method={method_key} but template_dir is not set (config: {cfg_path})")
        return

    template_path = Path(template_dir)
    if not template_path.exists():
        print(f"[score_ocr] template_dir missing: {template_path}")
        return
    if not template_path.is_dir():
        print(f"[score_ocr] template_dir is not a directory: {template_path}")
        return

    found_digits: set[int] = set()
    for child in template_path.iterdir():
        if not child.is_file():
            continue
        stem = child.stem.strip()
        if len(stem) == 1 and stem.isdigit():
            found_digits.add(int(stem))

    missing = [str(i) for i in range(10) if i not in found_digits]
    if missing:
        print(
            "[score_ocr] missing digit templates: "
            + ", ".join(missing)
            + f" (dir: {template_path})"
        )
    else:
        print(f"[score_ocr] template_dir ok: {template_path}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = load_config(args.config)
    _print_score_template_status(
        cfg_path=args.config,
        method=cfg.env.score_ocr.method,
        template_dir=cfg.env.score_ocr.template_dir,
    )
    window_titles = [args.window_title] if args.window_title else None
    env = make_env_factory(cfg, rank=0, seed=cfg.train.seed, window_titles=window_titles)()

    patience = args.patience if args.patience is not None else cfg.env.health_missing_patience
    patience = max(1, int(patience))

    missing_streak = 0

    try:
        while True:
            frame = env.backend.capture_frame()
            hud = env.hud_reader.read(frame)

            if hud.health_visible is False:
                missing_streak += 1
            else:
                missing_streak = 0

            health_pct = "n/a"
            if hud.health_percent is not None:
                health_pct = f"{hud.health_percent * 100.0:6.2f}%"

            print(
                f"\rscore={hud.score!s:>8}  health={health_pct:>8}  "
                f"visible={str(hud.health_visible):>5}  missing_streak={missing_streak:>3}",
                end="",
                flush=True,
            )

            if args.auto_reset and cfg.env.health_bar.enabled and missing_streak >= patience:
                env.backend.reset()
                env.hud_reader.reset()
                missing_streak = 0
                print("\n[reset] health bar missing -> ran reset macro")
                time.sleep(0.6)

            time.sleep(max(0.01, args.interval_s))
    except KeyboardInterrupt:
        print("\nStopped monitor")
    finally:
        env.close()


if __name__ == "__main__":
    main()
