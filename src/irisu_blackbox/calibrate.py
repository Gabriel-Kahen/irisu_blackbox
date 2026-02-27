from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from irisu_blackbox.config import load_config
from irisu_blackbox.factory import make_env_factory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a frame and overlay action grid for calibration")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/grid_overlay.png"))
    parser.add_argument("--window-title", type=str, default=None)
    return parser


def _draw_grid(frame_bgr, left: int, top: int, right: int, bottom: int, rows: int, cols: int):
    out = frame_bgr.copy()

    for c in range(cols + 1):
        x = int(left + (right - left) * (c / cols))
        cv2.line(out, (x, top), (x, bottom), (0, 255, 255), 1)
    for r in range(rows + 1):
        y = int(top + (bottom - top) * (r / rows))
        cv2.line(out, (left, y), (right, y), (0, 255, 255), 1)

    cv2.rectangle(out, (left, top), (right, bottom), (0, 200, 255), 2)
    return out


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = load_config(args.config)

    window_titles = [args.window_title] if args.window_title else None
    env_fn = make_env_factory(cfg, rank=0, seed=cfg.train.seed, window_titles=window_titles)
    env = env_fn()

    try:
        env.reset()
        frame_rgb = env.render()
        if frame_rgb is None:
            raise RuntimeError("Could not capture frame for calibration")

        frame_bgr = frame_rgb[:, :, ::-1]
        g = cfg.env.action_grid
        preview = _draw_grid(
            frame_bgr,
            left=g.left,
            top=g.top,
            right=g.right,
            bottom=g.bottom,
            rows=g.rows,
            cols=g.cols,
        )

        out_path = args.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), preview)
        if not ok:
            raise RuntimeError(f"Failed to write calibration image: {out_path}")
        print(f"Saved: {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
