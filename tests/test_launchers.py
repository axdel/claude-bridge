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
import shutil
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
    """All three launchers invoke ``python -I`` and never ``-P`` or an inherited
    ``PYTHONPATH`` export — the durable guard against regressing the CWE-427 fix.
    ``start.sh`` runs the bridge through the same project-venv interpreter (D-RUNTIME-003),
    so it takes full isolated mode too rather than a system-python + PYTHONPATH=src hack —
    the venv install makes ``claude_bridge`` and ``httpx`` importable without any path
    entry, so no untrusted path is ever needed on ``sys.path``."""
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("claude-grok", "claude-codex", "start.sh"):
        text = (repo_root / name).read_text()
        assert '"$BRIDGE_PY" -I ' in text, f"{name} must invoke python in isolated mode (-I)"
        assert '"$BRIDGE_PY" -P' not in text, f"{name} must not use -P (leaves PYTHONPATH open)"
        assert "export PYTHONPATH" not in text, f"{name} must not re-export inherited PYTHONPATH"


def test_launcher_banners_derive_model_from_config_owner():
    """Neither banner may name a model literal — each resolves it from the module that
    owns the id, so the printed model cannot drift from the one actually sent.

    D-XAI-008 established this for ``claude-grok``. ``claude-codex`` kept a hardcoded
    ``gpt-5.6-sol`` beside the authoritative ``DEFAULT_MODEL``: a second writer that
    stayed correct only by luck, and would have printed the old id while sending the
    new one the moment either changed. The banner is the operator's only in-terminal
    statement of what is on the wire, so a stale one is a lie, not a cosmetic bug.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("claude-codex", "claude-grok"):
        text = (repo_root / name).read_text()
        banner = next(line for line in text.splitlines() if " model:" in line)
        _, _, after_model = banner.partition("model:")
        assert after_model.startswith("$"), (
            f"{name} banner hardcodes a model literal — derive it from the owning "
            f"module instead (D-XAI-008): {banner.strip()!r}"
        )
        assert "BRIDGE_MODEL=$(" in text, f"{name} must resolve its banner model at runtime"
        assert '"$BRIDGE_PY" -I -c' in text, (
            f"{name} must resolve the banner model through the isolated venv python"
        )


# Skill-recipe argv that /plan and /review actually dispatch (flags, then wrapper --, then prompt).
_SKILL_RECIPE = (
    "-p",
    "--effort",
    "max",
    "--output-format",
    "json",
    "--permission-mode",
    "plan",
    "--",
    "You are QualityReviewer.",
)
# Wrapper consumes the first `--`. Skill recipe therefore forwards flags + prompt
# without that `--`. Frontmatter prompts need a *second* `--` (protocol recipe:
# `<cli> -- -p … -- PROMPT`).
_SKILL_RECIPE_FORWARDED = (
    "-p",
    "--effort",
    "max",
    "--output-format",
    "json",
    "--permission-mode",
    "plan",
    "You are QualityReviewer.",
)


def _extract_claude_args_parse(text: str) -> str:
    """Slice the launcher's CLAUDE_ARGS while-loop, nothing else (no bridge spawn)."""
    start = text.index("CLAUDE_ARGS=()")
    end = text.index("\ndone\n", start) + len("\ndone\n")
    return text[start:end]


def _run_claude_args_parse(loop: str, argv: tuple[str, ...]) -> list[str]:
    """Evaluate the extracted loop under bash; return the resulting CLAUDE_ARGS."""
    script = loop + 'printf "%s\\0" "${CLAUDE_ARGS[@]}"\n'
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, "-c", script, "_", *argv],
        capture_output=True,
        timeout=5,
        check=True,
    )
    if not result.stdout:
        return []
    return [part.decode() for part in result.stdout.split(b"\0") if part]


def test_launcher_parse_skill_recipe_keeps_print_and_plan_flags():
    """Skill recipe ``<cli> -p --flags -- PROMPT`` must forward -p/json/plan.

    Live A-vs-B (2026-08-28): replace-on-``--`` dropped every flag accumulated by
    ``*)``, so inner argv was ``claude <prompt>`` (plain text, no json, not plan
    mode). Flags after wrapper ``--`` kept ``-p`` and returned JSON
    ``{"result":"OK"}``. Oracle is that forwarded argv, not the running wrapper.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("claude-codex", "claude-grok"):
        loop = _extract_claude_args_parse((repo_root / name).read_text())
        forwarded = _run_claude_args_parse(loop, _SKILL_RECIPE)
        assert forwarded == list(_SKILL_RECIPE_FORWARDED), (
            f"{name} parse dropped print/plan flags: {forwarded!r}"
        )


def test_launcher_parse_flags_after_double_dash_still_forwards():
    """``<cli> -- -p --flags -- PROMPT`` (wrapper-doc form) keeps the same argv."""
    repo_root = Path(__file__).resolve().parents[1]
    argv = ("--", *_SKILL_RECIPE)
    for name in ("claude-codex", "claude-grok"):
        loop = _extract_claude_args_parse((repo_root / name).read_text())
        forwarded = _run_claude_args_parse(loop, argv)
        assert forwarded == list(_SKILL_RECIPE), (
            f"{name} flags-after-`--` parse drifted: {forwarded!r}"
        )


def test_launcher_parse_preflight_without_double_dash_keeps_print():
    """Pre-flight has no wrapper ``--``; ``*)`` accumulation must still keep ``-p``."""
    repo_root = Path(__file__).resolve().parents[1]
    argv = (
        "-p",
        "Status check — respond with OK and your model name.",
        "--effort",
        "max",
        "--output-format",
        "text",
        "--permission-mode",
        "plan",
    )
    for name in ("claude-codex", "claude-grok"):
        loop = _extract_claude_args_parse((repo_root / name).read_text())
        forwarded = _run_claude_args_parse(loop, argv)
        assert forwarded == list(argv), f"{name} pre-flight parse drifted: {forwarded!r}"


def test_launcher_parse_debug_is_consumed_not_forwarded():
    """``--debug`` is a wrapper flag; it must not appear in CLAUDE_ARGS."""
    repo_root = Path(__file__).resolve().parents[1]
    argv = ("--debug", "-p", "--", "hello")
    for name in ("claude-codex", "claude-grok"):
        loop = _extract_claude_args_parse((repo_root / name).read_text())
        forwarded = _run_claude_args_parse(loop, argv)
        assert "--debug" not in forwarded
        assert forwarded == ["-p", "hello"], f"{name}: {forwarded!r}"
