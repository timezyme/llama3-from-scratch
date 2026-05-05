#!/usr/bin/env bash
# Creates a custom GCP Machine Image from your fully-provisioned VM.
# Run this once after you've successfully run provision_l4.sh and
# verified everything is working.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f .l4-config.env ]] || { echo "ERROR: .l4-config.env missing."; exit 1; }
# shellcheck disable=SC1091
source .l4-config.env

: "${VM_NAME:?missing in .l4-config.env}"
CUSTOM_IMAGE_NAME="${VM_NAME}-image"

echo "Checking for existing VM $VM_NAME..."
EXISTING=$(gcloud compute instances list --filter="name=$VM_NAME" --format="value(zone.basename(),status)" 2>/dev/null || true)

if [[ -z "$EXISTING" ]]; then
    echo "ERROR: VM $VM_NAME not found. Run provision_l4.sh first to set up the VM."
    exit 1
fi

ZONE=$(echo "$EXISTING" | head -1 | awk '{print $1}')
STATUS=$(echo "$EXISTING" | head -1 | awk '{print $2}')

if [[ "$STATUS" != "TERMINATED" ]]; then
    echo "Stopping VM $VM_NAME in zone $ZONE to ensure data consistency before imaging..."
    gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
fi

echo "Creating custom image $CUSTOM_IMAGE_NAME from disk $VM_NAME in zone $ZONE..."
echo "This will take a few minutes..."
gcloud compute images create "$CUSTOM_IMAGE_NAME" \
    --source-disk="$VM_NAME" \
    --source-disk-zone="$ZONE" \
    --force

echo "=========================================================================="
echo "Custom image $CUSTOM_IMAGE_NAME created successfully."
echo ""
echo "Next time you run provision_l4.sh, it will automatically use this image."
echo "Startup time will be reduced from ~15-20 minutes to ~1 minute, as all"
echo "drivers, weights, and packages are pre-installed."
echo ""
echo "You can now safely delete the underlying VM to save costs:"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo "=========================================================================="
