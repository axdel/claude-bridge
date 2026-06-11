#!/usr/bin/env bash
#
# Run mutation testing with the only invocation that reports an accurate kill
# rate for this project. Use this instead of calling `pytest --gremlins` by hand.
#
# Why this wrapper exists
# -----------------------
# pytest-gremlins' default coverage-guided test selection builds a degenerate
# line->test map on this environment (Python 3.14): each mutated line resolves
# to a SINGLE covering test, even when many tests exercise it. That one test
# rarely asserts the mutated behavior, so the mutant is scored a false survivor
# and the kill rate is under-reported (observed: 50% where the true rate is
# 100%). The operative remedy is `--gremlin-no-coverage-filter`, which bypasses
# the map and runs the full test set per mutant. `--no-cov` alone does NOT fix
# the selection -- it only avoids redundant coverage instrumentation overhead
# during the mutant runs (pyproject's addopts forces `--cov` on every pytest
# invocation).
#
# Accuracy vs speed
# -----------------
# `--gremlin-no-coverage-filter` runs the entire collected test suite against
# every mutant. To narrow the kill-test universe (much faster, still accurate),
# pass the relevant test path(s) as trailing arguments -- they restrict which
# tests pytest collects:
#
#     scripts/mutate.sh                       # mutate changed source vs HEAD, full suite
#     scripts/mutate.sh tests/test_stream.py  # same, but only test_stream.py as kill tests
#     scripts/mutate.sh --all                 # mutate the whole src tree (scheduled sweep)
#
# Target: >=85% kill rate on changed source files (zero survivors for auth code).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mode="diff"
if [[ "${1:-}" == "--all" ]]; then
  mode="all"
  shift
fi

# Mutation targets are package source files under src/ (which holds no tests).
# Both git commands exit 0 on empty output, so a no-change tree falls through to
# the empty-targets guard below instead of aborting under `set -e`/`pipefail`.
# An unscoped run mutates the whole tree -- slower, and noisy with unrelated
# survivors -- so diff mode narrows to source files changed vs HEAD.
if [[ "$mode" == "all" ]]; then
  targets="$(git ls-files -- 'src/*.py' | paste -sd, -)"
else
  targets="$(git diff --name-only HEAD -- 'src/*.py' | paste -sd, -)"
fi

if [[ -z "$targets" ]]; then
  echo "No changed Python source files vs HEAD -- nothing to mutate."
  echo "Run 'scripts/mutate.sh --all' to mutate the whole src tree."
  exit 0
fi

echo "Mutating: $targets"

exec uv run pytest --no-cov --gremlins \
  --gremlin-targets="$targets" \
  --gremlin-no-coverage-filter \
  --gremlin-parallel \
  --gremlin-cache \
  "$@"
