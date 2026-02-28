from pathlib import Path

from irisu_blackbox.config import load_config


def test_template_dir_resolves_when_running_outside_repo(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    config_dir = repo / "configs"
    template_dir = repo / "assets" / "score_templates"

    config_dir.mkdir(parents=True)
    template_dir.mkdir(parents=True)

    cfg_path = config_dir / "base.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[env]",
                'backend = "windows"',
                "obs_width = 96",
                "obs_height = 96",
                "frame_stack = 4",
                "[env.action_grid]",
                "rows = 8",
                "cols = 8",
                "left = 0",
                "top = 0",
                "right = 640",
                "bottom = 480",
                "[env.score_ocr]",
                'enabled = true',
                'method = "template"',
                'template_dir = "assets/score_templates"',
                "template_min_similarity = 0.32",
                "template_fallback_to_tesseract = false",
                "template_expected_digits = 8",
                "template_inner_left = 0",
                "template_inner_right = 0",
                "monotonic_non_decreasing = true",
                "hold_last_value_when_missing = true",
                "min_confidence = 40.0",
                "max_step_increase = 2500",
            ]
        ),
        encoding="utf-8",
    )

    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    monkeypatch.chdir(outside)

    cfg = load_config(cfg_path)
    assert cfg.env.score_ocr.template_dir == str(template_dir.resolve())


def test_window_launch_paths_resolve_when_running_outside_repo(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    config_dir = repo / "configs"
    game_dir = repo / "game"
    executable = game_dir / "Irisu.exe"

    config_dir.mkdir(parents=True)
    game_dir.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")

    cfg_path = config_dir / "base.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[env]",
                'backend = "windows"',
                "[env.window]",
                'launch_executable = "game/Irisu.exe"',
                'launch_workdir = "game"',
            ]
        ),
        encoding="utf-8",
    )

    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    monkeypatch.chdir(outside)

    cfg = load_config(cfg_path)
    assert cfg.env.window.launch_executable == str(executable.resolve())
    assert cfg.env.window.launch_workdir == str(game_dir.resolve())
