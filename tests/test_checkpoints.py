from pathlib import Path

from irisu_blackbox.checkpoints import find_latest_resume_path


def test_find_latest_resume_path_prefers_newest_checkpoint_over_older_final_model(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    older_checkpoint = checkpoints / "irisu_ppo_100_steps.zip"
    newer_checkpoint = checkpoints / "irisu_ppo_900_steps.zip"
    final_model = run_dir / "final_model.zip"
    older_checkpoint.write_text("", encoding="utf-8")
    newer_checkpoint.write_text("", encoding="utf-8")
    final_model.write_text("", encoding="utf-8")

    base_time = 1_700_000_000

    import os

    os.utime(final_model, (base_time, base_time))
    os.utime(older_checkpoint, (base_time + 10, base_time + 10))
    os.utime(newer_checkpoint, (base_time + 20, base_time + 20))

    latest = find_latest_resume_path(run_dir)
    assert latest == newer_checkpoint


def test_find_latest_resume_path_prefers_newer_final_model(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "irisu_ppo_900_steps.zip"
    final_model = run_dir / "final_model.zip"
    checkpoint.write_text("", encoding="utf-8")
    final_model.write_text("", encoding="utf-8")

    older_time = 1_700_000_000
    newer_time = older_time + 100

    checkpoint.touch()
    final_model.touch()

    import os

    os.utime(checkpoint, (older_time, older_time))
    os.utime(final_model, (newer_time, newer_time))

    latest = find_latest_resume_path(run_dir)
    assert latest == final_model


def test_find_latest_resume_path_falls_back_to_final_model(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    final_model = run_dir / "final_model.zip"
    final_model.write_text("", encoding="utf-8")

    latest = find_latest_resume_path(run_dir)
    assert latest == final_model
