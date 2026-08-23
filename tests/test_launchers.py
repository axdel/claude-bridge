"""Launcher isolation regression tests (CWE-427 search-path injection).

The ``claude-grok`` / ``claude-codex`` launchers must start the bridge in Python
*isolated* mode (``-I``), not merely safe-path mode (``-P``). ``-P`` drops only the
current-directory prepend; it still honors an inherited ``PYTHONPATH``, so a hostile
``PYTHONPATH=.`` (or a shadowing ``httpx.py`` / ``sitecustomize.py`` in the project
dir) can execute ahead of the real bridge with access to the provider ``auth.json``.
``-I`` additionally ignores ``PYTHONPATH`` and the per-user site dir, closing the vector.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Prints the resolved httpx module and whether the hostile sitecustomize marker is set.
_PROBE = (
    "import sys, httpx\n"
    "print('HTTPX', httpx.__file__)\n"
    "print('SITEC', hasattr(sys, '_HOSTILE_SITECUSTOMIZE_RAN'))\n"
)


def _write_hostile(directory: Path) -> None:
    """Plant a hostile httpx shadow and sitecustomize in *directory*.

    ``httpx.py`` aborts on import (a real dependency shadow); ``sitecustomize.py`` is
    auto-imported by CPython from any ``sys.path`` entry at startup and sets a marker.
    A launcher that puts *directory* on the path would run one or both.
    """
    (directory / "httpx.py").write_text(
        'raise SystemExit("HOSTILE httpx.py executed — search-path injection not defeated")\n'
    )
    (directory / "sitecustomize.py").write_text(
        "import sys\nsys._HOSTILE_SITECUSTOMIZE_RAN = True\n"
    )


def test_isolated_python_ignores_hostile_pythonpath_and_cwd(tmp_path):
    """``python -I`` loads the real httpx and never runs the hostile sitecustomize,
    even with the hostile dir on both ``PYTHONPATH`` and the working directory."""
    _write_hostile(tmp_path)
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # The real httpx resolves from the venv, never the shadow planted in tmp_path.
    assert str(tmp_path) not in result.stdout
    assert "SITEC False" in result.stdout


def test_unsafe_path_python_loads_hostile_module(tmp_path):
    """F2P canary: the fixture is genuinely injectable. Under ``-P`` (the mode the
    launchers previously used) the hostile httpx shadow DOES execute — exactly the
    CWE-427 vector that ``-I`` closes above. If this ever stops failing, the isolation
    test above proves nothing."""
    _write_hostile(tmp_path)
    result = subprocess.run(
        [sys.executable, "-P", "-c", "import httpx"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "HOSTILE" in result.stderr


def test_launchers_use_isolated_python():
    """Both launchers invoke ``python -I`` and never ``-P`` or an inherited
    ``PYTHONPATH`` export — the durable guard against regressing the CWE-427 fix."""
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("claude-grok", "claude-codex"):
        text = (repo_root / name).read_text()
        assert '"$BRIDGE_PY" -I ' in text, f"{name} must invoke python in isolated mode (-I)"
        assert '"$BRIDGE_PY" -P' not in text, f"{name} must not use -P (leaves PYTHONPATH open)"
        assert "export PYTHONPATH" not in text, f"{name} must not re-export inherited PYTHONPATH"
