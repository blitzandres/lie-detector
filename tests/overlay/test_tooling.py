"""Tooling guards: deps declared, tools importable, env handling present."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_declares_overlay_packages_and_deps():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packages = data["tool"]["setuptools"]["packages"]
    assert "blitz_overlay" in packages
    assert "blitz_overlay.cues" in packages
    deps = data["project"]["dependencies"]
    joined = " ".join(deps)
    assert "fastapi" in joined
    assert "uvicorn" in joined
    assert "numpy" in joined


def test_ruff_and_pytest_importable():
    import pytest  # noqa: F401
    import subprocess, sys
    assert subprocess.run([sys.executable, "-m", "ruff", "--version"]).returncode == 0


def test_env_example_and_gitignore():
    assert (ROOT / ".env.example").exists()
    gitignore = (ROOT / ".gitignore").read_text()
    assert ".env" in gitignore
    assert "logs/" in gitignore


def test_ci_workflow_runs_ruff_and_pytest():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "ruff" in ci
    assert "pytest" in ci
