from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from irisu_blackbox.config import Rect, load_config
from irisu_blackbox.factory import make_env_factory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a frame and overlay action grid for calibration")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("--out", type=Path, default=Path("grid_overlay.png"))
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


def _draw_region(frame_bgr, region: Rect, label: str, color: tuple[int, int, int]):
    out = frame_bgr.copy()
    cv2.rectangle(
        out,
        (region.left, region.top),
        (region.right, region.bottom),
        color,
        2,
    )
    cv2.putText(
        out,
        label,
        (region.left, max(12, region.top - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def _draw_scanline(
    frame_bgr,
    x_start: int,
    x_end: int,
    y: int,
    half_height: int,
    label: str,
    color: tuple[int, int, int],
):
    out = frame_bgr.copy()
    left = min(x_start, x_end)
    right = max(x_start, x_end)
    top = max(0, y - half_height)
    bottom = y + half_height
    cv2.rectangle(out, (left, top), (right, bottom), color, 2)
    cv2.line(out, (left, y), (right, y), color, 1)
    cv2.putText(
        out,
        label,
        (left, max(12, top - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def _to_frame_coords(x: int, y: int, capture_region: Rect | None) -> tuple[int, int]:
    if capture_region is None:
        return x, y
    return x - capture_region.left, y - capture_region.top


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
        grid_left, grid_top = _to_frame_coords(g.left, g.top, cfg.env.window.capture_region)
        grid_right, grid_bottom = _to_frame_coords(g.right, g.bottom, cfg.env.window.capture_region)
        preview = _draw_grid(
            frame_bgr,
            left=grid_left,
            top=grid_top,
            right=grid_right,
            bottom=grid_bottom,
            rows=g.rows,
            cols=g.cols,
        )
        if cfg.env.score_ocr.enabled and cfg.env.score_ocr.region is not None:
            preview = _draw_region(preview, cfg.env.score_ocr.region, "score_ocr", (0, 255, 80))
        if cfg.env.health_bar.enabled:
            if cfg.env.health_bar.method.lower() == "scanline":
                if (
                    cfg.env.health_bar.scanline_start_x is not None
                    and cfg.env.health_bar.scanline_end_x is not None
                    and cfg.env.health_bar.scanline_y is not None
                ):
                    preview = _draw_scanline(
                        preview,
                        x_start=cfg.env.health_bar.scanline_start_x,
                        x_end=cfg.env.health_bar.scanline_end_x,
                        y=cfg.env.health_bar.scanline_y,
                        half_height=cfg.env.health_bar.scanline_half_height,
                        label="health_scanline",
                        color=(0, 80, 255),
                    )
            elif cfg.env.health_bar.region is not None:
                preview = _draw_region(preview, cfg.env.health_bar.region, "health_bar", (0, 80, 255))

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
