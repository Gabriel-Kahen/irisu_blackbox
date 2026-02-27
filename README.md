# Irisu Black-Box RL Baseline

This repository is a practical baseline for training an RL agent against **unmodified Irisu Syndrome** via:

- live screen capture
- simulated mouse clicks
- no internal game-state access

It is built around a Gymnasium environment plus `RecurrentPPO` (`CnnLstmPolicy`) so you can start with a mock backend, then switch to live Windows control.

## What Is Included

- `IrisuBlackBoxEnv`: Gymnasium env with discrete click-grid actions
- `WindowsGameBackend`: real-time capture + mouse control (screen-level black-box)
- `MockGameBackend`: deterministic toy backend for local pipeline validation
- reward shaping components:
  - survival reward
  - frame-activity reward
  - cascade bonus proxy (large frame deltas)
  - stale-state penalty
  - optional OCR score-delta reward
- HUD extraction hooks:
  - score OCR reader
  - health bar percentage reader
  - game-over trigger from health bar disappearance
- `irisu-train` / `irisu-play` / `irisu-calibrate` CLIs
- `irisu-monitor`: live HUD readout (`score`, `health %`, `health visible`)

## Project Layout

- `configs/base.toml`: environment and PPO defaults
- `src/irisu_blackbox/env.py`: Gym env wrapper
- `src/irisu_blackbox/backends/windows.py`: live game backend
- `src/irisu_blackbox/train.py`: training entrypoint
- `src/irisu_blackbox/calibrate.py`: captures frame with action-grid overlay

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional OCR score reward:

```bash
pip install -e '.[ocr]'
```

## Quick Start (Mock Backend)

Default config uses `env.backend = "mock"` so you can validate the full PPO pipeline immediately:

```bash
irisu-train --config configs/base.toml --run-dir runs/mock_smoke --total-timesteps 200000
```

Run an inference rollout:

```bash
irisu-play --config configs/base.toml --model runs/mock_smoke/final_model.zip --episodes 3 --deterministic
```

## Live Windows Backend

1. Set backend and window matching in `configs/base.toml`:

- `env.backend = "windows"`
- `env.window.title_regex = "Irisu"`
- set optional `env.window.capture_region` if full-window capture is not ideal

2. Set click-grid bounds (`env.action_grid.left/top/right/bottom`) in screen coordinates.

3. Define `env.reset_macro` so `env.reset()` reliably starts a new run.
4. Configure HUD regions to track score and health:

- `env.score_ocr`: score region + optional tesseract path
- `env.health_bar`: health bar detection settings (`method = "profile"` or `"scanline"`)
- `env.game_over_on_health_missing = true` to terminate when bar disappears
- set `env.episode.max_clicks_per_second = 3.0` to cap click rate
- set `env.health_bar.invert_percent = true` if monitor output is directionally inverted
- tune `env.health_bar.adaptive_fill_peak_ratio` if dark baseline is being counted as fill
- set `env.health_bar.fill_direction = "left_to_right"` (or `"right_to_left"`) for edge-based fill %
- for constant UI, use `method = "scanline"` with fixed `scanline_start_x/end_x/y`
- increase `env.health_bar.smoothing_window` (e.g. 5-9) to reduce one-frame health spikes

5. Use calibration preview to verify grid alignment:

```bash
irisu-calibrate --config configs/base.toml --out artifacts/grid_overlay.png
```

6. Train:

```bash
irisu-train --config configs/base.toml --run-dir runs/live_01
```

### Live HUD Monitor

Use this while tuning score/health regions:

```bash
irisu-monitor --config configs/base.toml
```

Auto-run reset macro when health disappears for configured patience:

```bash
irisu-monitor --config configs/base.toml --auto-reset
```

### Irisu Menu Restart Macro (Start Click)

Use a click step in `env.reset_macro` that targets the `Start` button.
If `relative_to_capture = true`, `x`/`y` are interpreted relative to the capture frame's top-left corner.

```toml
[[env.reset_macro]]
kind = "click"
x = 420
y = 265
button = "left"
relative_to_capture = true
duration_s = 0.04

[[env.reset_macro]]
kind = "sleep"
duration_s = 0.8
```

## Multi-Instance Training

Set parallel env count:

```toml
[train]
n_envs = 4
```

You can pin each env to a different window title with CLI override:

```bash
irisu-train \
  --config configs/base.toml \
  --window-titles "Irisu #1,Irisu #2,Irisu #3,Irisu #4" \
  --run-dir runs/live_4x
```

If no explicit title list is provided, envs use the same regex and select windows by `window_index`.

## Reward Shaping Notes

Reward is composed as:

`survival + activity + cascade_bonus + stale_penalty + score_delta`

For strict black-box setups with no OCR, leave score reward disabled and rely on time/activity proxies until your capture/reset loop is stable.

When score OCR is enabled, `score_delta` uses the per-step score increase from HUD readings.

## Safety Notes

- Keep `pyautogui` fail-safe enabled (top-left corner abort).
- Run this on a dedicated desktop/session to avoid unwanted clicks.
- Verify reset macro behavior before long training runs.

## Test

```bash
pytest
```
