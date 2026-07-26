"""Integration tests for the Windows batch test runner."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch runner")
def test_batch_runner_propagates_pytest_failure():
    """A pytest argument error must produce a non-zero batch exit code."""
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(sys.executable).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )

    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            str(project_root / "run_tests.bat"),
            "--definitely-invalid-pytest-option",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Tests failed with exit code" in result.stdout
