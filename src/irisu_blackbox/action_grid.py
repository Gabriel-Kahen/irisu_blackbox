from __future__ import annotations

from dataclasses import dataclass

from irisu_blackbox.config import ActionGridConfig


@dataclass(slots=True)
class ActionCommand:
    x: int
    y: int
    button: str


NOOP_ACTION = 0


def _cell_center(index: int, cfg: ActionGridConfig) -> tuple[int, int]:
    row = index // cfg.cols
    col = index % cfg.cols

    cell_w = (cfg.right - cfg.left) / cfg.cols
    cell_h = (cfg.bottom - cfg.top) / cfg.rows

    x = int(cfg.left + ((col + 0.5) * cell_w))
    y = int(cfg.top + ((row + 0.5) * cell_h))
    return x, y


def decode_action(action: int, cfg: ActionGridConfig) -> ActionCommand | None:
    if action == NOOP_ACTION:
        return None

    grid_size = cfg.grid_size
    if action < 1 or action >= (1 + 2 * grid_size):
        raise ValueError(f"Action {action} out of range for action_count={cfg.action_count}")

    button = "left" if action <= grid_size else "right"
    offset = action - 1
    if button == "right":
        offset -= grid_size

    x, y = _cell_center(offset, cfg)
    return ActionCommand(x=x, y=y, button=button)
