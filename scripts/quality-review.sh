#!/usr/bin/env bash
#
# Run the full branch-review quality gate with the one environment tweak this
# src-layout project needs. Use this instead of calling
# `claude-protocol quality run --phase branch-review` by hand.
#
# Why this wrapper exists
# -----------------------
# The branch-review roster includes grimp (dependency metrics: Ca/Ce/Instability).
# claude-protocol runs grimp IN-PROCESS under its own globally-installed
# interpreter, which does NOT carry this project's `.venv` editable install -- so
# `grimp.build_graph("claude_bridge")` fails with "Could not find package
# 'claude_bridge' in your Python path" and the whole gate blocks on it.
#
# The fix is an ambient PYTHONPATH=src: it puts the src-layout package on
# claude-protocol's interpreter path, and it survives claude-protocol's subprocess
# env scrubbing (which drops only .env-absorbed keys, never ambient ones -- so a
# project `.env` would NOT reach the grimp subprocess). `uv run` does not help
# either: the global console script uses its own interpreter regardless. See
# DECISIONS.md D-QUALITY-003 for the full rationale and rejected alternatives.
#
# Every other tool in the roster (bandit, pip-audit, lizard, pytest-gremlins,
# vulture, deptry, import-linter, hypothesis, pytest-cov, gitleaks, godfile,
# jscpd) is unaffected by PYTHONPATH=src -- for those it is a harmless no-op.
#
#     scripts/quality-review.sh    # full branch-review gate (extra args forwarded to the CLI)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

exec env PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" \
  claude-protocol quality run --phase branch-review "$@"
