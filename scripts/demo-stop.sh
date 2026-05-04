#!/usr/bin/env bash
# Stop the L4 VM after the TA session. Run when the demo is done.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f .l4-config.env ]] || { echo "ERROR: .l4-config.env missing"; exit 1; }
# shellcheck disable=SC1091
source .l4-config.env
[[ -n "${GCLOUD_PATH:-}" ]] && export PATH="$GCLOUD_PATH:$PATH"

ZONE="${PREFERRED_ZONE:-us-east1-c}"

echo "[demo] stopping $VM_NAME..."
gcloud compute instances stop "$VM_NAME" --zone="$ZONE" --quiet >/dev/null
gcloud compute instances list --filter="name=$VM_NAME" --format="value(name,status)"
