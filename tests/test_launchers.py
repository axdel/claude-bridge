"""Launcher isolation regression tests (CWE-427 search-path injection).

The ``claude-grok`` / ``claude-codex`` launchers must start the bridge in Python
*isolated* mode (``-I``), not merely safe-path mode (``-P``). ``-P`` drops only the
current-directory prepend; it still honors an inherited ``PYTHONPATH``, so a hostile
``PYTHONPATH=.`` (or a shadowing ``httpx.py`` / ``sitecustomize.py`` in the project
dir) can execute ahead of the real bridge with access to the provider ``auth.json``.
``-I`` additionally ignores ``PYTHONPATH`` and the per-user site dir, closing the vector.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

import claude_bridge

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


def test_package_version_derives_from_the_distribution_metadata():
    """``__version__`` must equal the installed distribution version, not restate it.

    The banner prints this string as the operator's only in-terminal statement of which
    bridge is running, so a stale one is a lie in exactly the way a stale model id is
    (the sibling assertion above, D-XAI-008) -- and this one drifted for real. Releases
    bump ``pyproject.toml`` and nothing bumps ``__init__.py``, so every release needed a
    manual follow-up commit to re-sync it: ``chore: sync launcher __version__ to 0.10.0``,
    ``chore: sync runtime version with 0.7.0 release``, and -- after the sync was
    forgotten -- ``fix: sync __init__.py version with pyproject.toml (0.6.3)``. Cutting
    v0.11.0 drifted it again, to 0.10.0.

    The expected value comes from the packaging metadata, which the build derives from
    ``pyproject.toml``; nothing here consults the attribute under test to decide what it
    should be.
    """
    assert claude_bridge.__version__ == importlib.metadata.version("claude-bridge"), (
        "__version__ is a second writer for a fact pyproject.toml owns -- derive it from "
        "importlib.metadata instead of restating the literal"
    )


def test_package_version_falls_back_to_a_sentinel_when_not_installed(monkeypatch):
    """Importing from a source tree with no install must yield a sentinel, not a crash.

    ``importlib.metadata.version`` raises ``PackageNotFoundError`` when no distribution is
    registered -- which would otherwise propagate out of ``import claude_bridge`` and take
    down every consumer, including the launchers' own banner probe. The sentinel is
    deliberately not a plausible version: a real-looking number here would be a second
    writer again, silently wrong instead of visibly unknown.
    """
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        _raise_package_not_found,
    )
    reloaded = importlib.reload(claude_bridge)
    try:
        assert reloaded.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(claude_bridge)


def _raise_package_not_found(_name: str) -> str:
    """Stand in for ``importlib.metadata.version`` on an unregistered distribution."""
    raise importlib.metadata.PackageNotFoundError(_name)


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


def test_launchers_do_not_set_the_claude_code_context_window():
    """The harness installer is the single writer for CLAUDE_CODE_MAX_CONTEXT_TOKENS.

    It is a Claude Code env var the bridge never reads, and setting it here would do
    nothing: Claude Code resolves a recognized model id to that model's own window and
    never consults the var, measured by driving the TUI ``/context`` panel with the var
    at 333000 and observing an unchanged 200k window. A launcher-local export therefore
    states a window nothing acts on, in a second place, inviting drift against the
    harness value. The window is raised by arming the 1M model path instead — see the
    companion test below and D-CONTEXT-003.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("claude-codex", "claude-grok"):
        text = (repo_root / name).read_text()
        offenders = [
            line
            for line in text.splitlines()
            if "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in line and not line.lstrip().startswith("#")
        ]
        assert not offenders, (
            f"{name} sets CLAUDE_CODE_MAX_CONTEXT_TOKENS ({offenders!r}); the harness "
            f"installer owns it globally, and it is inert for a recognized model id"
        )


def test_launchers_arm_the_one_million_context_model():
    """Both launchers must default ANTHROPIC_MODEL to the ``[1m]``-suffixed model id.

    ANTHROPIC_AUTH_TOKEN puts a bridge session in API-billing mode, where claude-opus-5
    resolves to its 200k base window and CLAUDE_CODE_AUTO_COMPACT_WINDOW (350k) is never
    the binding term. The ``[1m]`` suffix arms the 1M path, restoring the harness window;
    dropping it silently halves every bridge session's context. The ``:-`` default form is
    load-bearing — a bare assignment would clobber an operator's own ANTHROPIC_MODEL.
    See D-CONTEXT-003.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("claude-codex", "claude-grok"):
        text = (repo_root / name).read_text()
        exports = [
            line.strip()
            for line in text.splitlines()
            if "ANTHROPIC_MODEL" in line and not line.lstrip().startswith("#")
        ]
        assert exports == ['export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-opus-5[1m]}"'], (
            f"{name} must default ANTHROPIC_MODEL to the [1m]-suffixed id exactly once; "
            f"found {exports!r}"
        )
