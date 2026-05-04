#!/usr/bin/env bash
# Start the L4 VM, wait for SSH, drop into the interactive REPL on the VM.
# Run this ~5 min before the TA session. Type prompts at the > marker.
# When the demo is done, type 'exit' to leave the REPL, then run scripts/demo-stop.sh.
#
# Usage:
#   ./scripts/demo-start.sh             # default --max-tokens 8
#   ./scripts/demo-start.sh 16          # custom max-tokens

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f .l4-config.env ]] || { echo "ERROR: .l4-config.env missing"; exit 1; }
# shellcheck disable=SC1091
source .l4-config.env
[[ -n "${GCLOUD_PATH:-}" ]] && export PATH="$GCLOUD_PATH:$PATH"

MAX_TOKENS="${1:-32}"
ZONE="${PREFERRED_ZONE:-us-east1-c}"

echo "[demo] starting $VM_NAME in $ZONE..."
gcloud compute instances start "$VM_NAME" --zone="$ZONE" >/dev/null

IP=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
echo "[demo] waiting for SSH at $IP..."
until nc -zw3 "$IP" 22 2>/dev/null; do sleep 3; done
echo "[demo] SSH up. Launching interactive REPL (warmup ~3 min)..."
echo

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="cd ~/CS265 && ./bin/llm --interactive --max-tokens $MAX_TOKENS"

echo
echo "[demo] REPL exited. To stop the VM, run: ./scripts/demo-stop.sh"
