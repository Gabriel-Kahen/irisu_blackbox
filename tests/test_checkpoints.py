from pathlib import Path

from irisu_blackbox.checkpoints import find_latest_resume_path


def test_find_latest_resume_path_prefers_highest_checkpoint_step(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "irisu_ppo_100_steps.zip").write_text("", encoding="utf-8")
    (checkpoints / "irisu_ppo_900_steps.zip").write_text("", encoding="utf-8")
    (run_dir / "final_model.zip").write_text("", encoding="utf-8")

    latest = find_latest_resume_path(run_dir)
    assert latest == checkpoints / "irisu_ppo_900_steps.zip"


def test_find_latest_resume_path_falls_back_to_final_model(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    final_model = run_dir / "final_model.zip"
    final_model.write_text("", encoding="utf-8")

    latest = find_latest_resume_path(run_dir)
    assert latest == final_model
