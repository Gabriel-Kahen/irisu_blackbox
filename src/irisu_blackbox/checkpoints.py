from __future__ import annotations

import re
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_resume_path(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    candidates: list[Path] = []
    if checkpoints_dir.is_dir():
        candidates.extend(checkpoints_dir.glob("*.zip"))
    final_model = run_dir / "final_model.zip"
    if final_model.exists():
        candidates.append(final_model)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (checkpoint_step(path), path.stat().st_mtime))
