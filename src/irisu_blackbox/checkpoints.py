from __future__ import annotations

import re
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_resume_path(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints: list[Path] = []
    if checkpoints_dir.is_dir():
        checkpoints.extend(checkpoints_dir.glob("*.zip"))
    final_model = run_dir / "final_model.zip"
    if not checkpoints and not final_model.exists():
        return None

    latest_checkpoint = (
        max(checkpoints, key=lambda path: (checkpoint_step(path), path.stat().st_mtime))
        if checkpoints
        else None
    )
    if not final_model.exists():
        return latest_checkpoint
    if latest_checkpoint is None:
        return final_model

    final_mtime = final_model.stat().st_mtime
    checkpoint_mtime = latest_checkpoint.stat().st_mtime
    if final_mtime >= checkpoint_mtime:
        return final_model
    return latest_checkpoint
