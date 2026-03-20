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

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m claude_bridge --port "$PORT" "$@"
