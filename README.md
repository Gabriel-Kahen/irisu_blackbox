# Irisu Black-Box RL Baseline

This repository is a practical baseline for training an RL agent against **unmodified Irisu Syndrome** via:

- live screen capture
- simulated mouse clicks
- no internal game-state access

It is built around a Gymnasium environment plus `RecurrentPPO`. By default the policy is now recurrent multi-input: a CNN/LSTM over stacked RGB game frames plus explicit HUD scalars for score and health.

## What Is Included

- `IrisuBlackBoxEnv`: Gymnasium env with discrete click-grid actions
- `WindowsGameBackend`: real-time capture + mouse control (screen-level black-box)
- `MockGameBackend`: deterministic toy backend for local pipeline validation
- reward shaping components:
  - survival reward
  - frame-activity reward
  - cascade bonus proxy (large frame deltas)
  - stale-state penalty
  - score-delta and score-value reward
  - health-value and health-delta reward
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

Template score reading (no OCR dependency) can be used instead by placing
digit templates in `assets/score_templates/` as `0.png` ... `9.png`.

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

3. Define `env.game_over_macro` to dismiss the death screen, then `env.reset_macro` to start a new run.
4. Configure HUD regions to track score and health:

- `env.score_ocr`: score region + optional tesseract path
- `env.health_bar`: health bar detection settings (`method = "profile"` or `"scanline"`)
- `env.game_over_on_health_missing = true` to terminate when bar disappears
- `env.game_over_template` can match the death-overlay screen directly
- `env.post_game_over_delay_s` can pause briefly after death is detected before reset logic starts
- `env.reset_ready_template` can match the menu screen before pressing `Start`
- `env.round_start_timeout_s` lets `reset()` wait until the HUD comes back after clicking `Start`
- set `env.episode.max_clicks_per_second = 3.0` to cap click rate
- set `env.health_bar.invert_percent = true` if monitor output is directionally inverted
- tune `env.health_bar.adaptive_fill_peak_ratio` if dark baseline is being counted as fill
- set `env.health_bar.fill_direction = "left_to_right"` (or `"right_to_left"`) for edge-based fill %
- for constant UI, use `method = "scanline"` with fixed `scanline_start_x/end_x/y`
- increase `env.health_bar.smoothing_window` (e.g. 5-9) to reduce one-frame health spikes
- set `env.health_bar.max_delta_per_step` to reject implausible one-frame health jumps
- set `env.score_ocr.score_smoothing_window = 5` for light anti-jitter smoothing
- raise `env.score_ocr.min_confidence` to ignore weak OCR reads
- set `env.score_ocr.max_step_increase` / `max_step_decrease` to reject implausible score jumps
  around `1000+` points per frame
- set `env.score_ocr.outlier_confirm_frames` for how many consecutive suspicious reads are needed
- set `env.score_ocr.method = "template"` to use digit-template matching
- set `env.score_ocr.template_dir = "assets/score_templates"` and add `0..9` image files
- set `env.score_ocr.template_expected_digits = 8` for the fixed 8-slot Irisu score
- tune `env.score_ocr.template_inner_left/right` if the score region has extra side padding
- set `env.score_ocr.template_fallback_to_tesseract = false` for deterministic template-only reads
- leave `env.hud_features.enabled = true` so the policy gets explicit `health` and `score` scalars

5. Use calibration preview to verify grid alignment:

```bash
irisu-calibrate --config configs/base.toml --out grid_overlay.png
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

### Stream Dashboard

For a stream-friendly side dashboard that shows RL training metrics, run:

```bash
irisu-dashboard --config configs/base.toml --run-dir runs/live_rgb_v1 --geometry 480x1080+0+0 --topmost
```

This opens a separate dashboard window showing:

- total timesteps
- fps
- episode reward mean
- episode length mean
- approx KL
- clip fraction
- entropy / value / policy losses
- explained variance
- static model config summary

The dashboard prefers a live `dashboard_metrics.json` file written by the training process.
If you point it at an older run that started before this feature existed, it falls back to
TensorBoard logs, which may update sparsely on slow real-time runs.

### Irisu Menu Restart Macro (Start Click)

Use a click step in `env.reset_macro` that targets the `Start` button.
If `relative_to_capture = true`, `x`/`y` are interpreted relative to the capture frame's top-left corner.
For a robust death -> menu -> new run loop:

- terminate on health-bar disappearance and/or a death-overlay template
- run `env.game_over_macro` once if reset begins off-menu (for example, on the death screen)
- optionally wait for `env.reset_ready_template` (menu screen) before running the reset macro
- let `env.reset()` wait for the HUD to reappear before returning the first observation

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

`survival + activity + cascade_bonus + stale_penalty + score_delta + score_value + health_value + health_delta`

The policy observation is also richer than before:

- `image`: stacked RGB frames from the captured game window
- `hud`: `[health_percent, normalized_score, health_visible, score_visible]`

For strict black-box setups with no OCR, leave score reward disabled and rely on time/activity proxies until your capture/reset loop is stable.

When score OCR is enabled, `score_delta` uses the per-step score increase from HUD readings, while `score_value` gives a smaller dense reward for staying in higher-scoring states. `health_value` rewards staying healthy; `health_delta` penalizes drops immediately.

## Safety Notes

- Keep `pyautogui` fail-safe enabled (top-left corner abort).
- Run this on a dedicated desktop/session to avoid unwanted clicks.
- Verify reset macro behavior before long training runs.

## Test

```bash
pytest
```
