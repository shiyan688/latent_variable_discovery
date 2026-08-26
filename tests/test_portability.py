from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_executable_sources_do_not_bind_a_local_home_or_virtualenv() -> None:
    if PROJECT_ROOT.joinpath(".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "lvs/*.py", "scripts/*.py", "scripts/*.sh"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        paths = [PROJECT_ROOT / name for name in tracked]
    else:
        paths = [
            *PROJECT_ROOT.joinpath("lvs").rglob("*.py"),
            *PROJECT_ROOT.joinpath("scripts").rglob("*.py"),
            *PROJECT_ROOT.joinpath("scripts").rglob("*.sh"),
        ]
    for path in paths:
        source = path.read_text()
        assert "/public/home/" not in source, path
        assert ".venv-lvs-gpu" not in source, path


def test_stage_c_manifest_uses_repository_relative_paths() -> None:
    manifest_path = (
        PROJECT_ROOT
        / "runs"
        / "nasa_battery_reviewer_clean_inner_symbolic_20260825"
        / "manifest.json"
    )
    config = json.loads(manifest_path.read_text())["config"]
    assert config["path_base"] == "repository_root"
    assert not Path(config["q_root"]).is_absolute()
    assert not Path(config["output_root"]).is_absolute()
