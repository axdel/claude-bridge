#!/usr/bin/env bash
# Claude Bridge — use Claude Code with any LLM provider
# https://github.com/axdel/claude-bridge
#
# Usage:
#   ./start.sh                    # auto mode (passthrough + failover)
#   ./start.sh --provider openai  # direct mode (always use OpenAI)
#   ./start.sh --port 9090        # custom port
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LLM_BRIDGE_PORT:-9999}"

# The bridge's one runtime dependency (httpx[http2], the HTTP/2 data plane) lives in the
# project venv that `uv sync` provisions at ${SCRIPT_DIR}/.venv — the system python3 does
# not have it. Run the bridge through that interpreter and fail loud with the fix if it is
# missing (D-RUNTIME-003). Because claude_bridge then resolves from the venv's editable
# install and httpx from its site-packages, no PYTHONPATH is needed and the bridge starts
# in full isolated mode (-I): no cwd, inherited PYTHONPATH, or per-user site dir on
# sys.path, so a hostile sitecustomize.py / claude_bridge/ / httpx.py in the working
# directory cannot run ahead of the real bridge (search-path injection, CWE-427).
BRIDGE_PY="${SCRIPT_DIR}/.venv/bin/python"
if [[ ! -x "$BRIDGE_PY" ]]; then
    echo "start.sh: ${SCRIPT_DIR}/.venv not found — run 'uv sync' in ${SCRIPT_DIR} first." >&2
    exit 1
fi

# Parse --port from args for the env var message
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--port" ]]; then
        PORT="$arg"
    fi
    prev_arg="$arg"
done

cat <<'BANNER'

      _                 _            _          _     _
  ___| | __ _ _   _  __| | ___      | |__  _ __(_) __| | __ _  ___
 / __| |/ _` | | | |/ _` |/ _ \ ___ | '_ \| '__| |/ _` |/ _` |/ _ \
| (__| | (_| | |_| | (_| |  __/|___|| |_) | |  | | (_| | (_| |  __/
 \___|_|\__,_|\__,_|\__,_|\___|     |_.__/|_|  |_|\__,_|\__, |\___|
                                                         |___/
BANNER
echo " port:${PORT}"
echo ""
echo "  To use with Claude Code:"
echo ""
echo "    export ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}"
echo "    export ANTHROPIC_AUTH_TOKEN=bridge-placeholder"
echo "    unset ANTHROPIC_API_KEY"
echo "    claude"
echo ""

exec "$BRIDGE_PY" -I -m claude_bridge --port "$PORT" "$@"
