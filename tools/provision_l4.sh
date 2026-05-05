#!/usr/bin/env bash
# One-time: provision GCP L4 spot VM and set up the project.
# Idempotent — reruns reuse an existing VM, skip downloaded weights, etc.
#
# Reads config from `.l4-config.env` at repo root. See `.l4-config.env.example`.
# Reads `HUGGINGFACE_TOKEN` from `.env` at repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- preflight ----
[[ -f .l4-config.env ]] || { echo "ERROR: .l4-config.env missing. Copy .l4-config.env.example and fill it in."; exit 1; }
# shellcheck disable=SC1091
source .l4-config.env
[[ -f .env ]] || { echo "ERROR: .env missing (need HUGGINGFACE_TOKEN=hf_... at repo root)"; exit 1; }
grep -q '^HUGGINGFACE_TOKEN=' .env || { echo "ERROR: .env missing HUGGINGFACE_TOKEN=..."; exit 1; }

: "${VM_NAME:?missing in .l4-config.env}"
: "${CUDA_VER:?missing in .l4-config.env}"
: "${ARCH:?missing in .l4-config.env}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
[[ "$PROVISIONING_MODEL" == "SPOT" || "$PROVISIONING_MODEL" == "STANDARD" ]] || \
    { echo "ERROR: PROVISIONING_MODEL must be SPOT or STANDARD"; exit 1; }
[[ -n "${GCLOUD_PATH:-}" ]] && export PATH="$GCLOUD_PATH:$PATH"
command -v gcloud >/dev/null || { echo "ERROR: gcloud not on PATH (set GCLOUD_PATH in .l4-config.env)"; exit 1; }

# ---- find or create VM ----
ZONE=""
EXISTING=$(gcloud compute instances list --filter="name=$VM_NAME" --format="value(zone.basename(),status)" 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
    ZONE=$(echo "$EXISTING" | head -1 | awk '{print $1}')
    STATUS=$(echo "$EXISTING" | head -1 | awk '{print $2}')
    echo "[provision] reusing VM $VM_NAME in $ZONE (status=$STATUS)"
    [[ "$STATUS" == "TERMINATED" ]] && gcloud compute instances start "$VM_NAME" --zone="$ZONE"
else
    # Check if a custom image exists to skip the 15+ minute setup
    CUSTOM_IMAGE_NAME="${VM_NAME}-image"
    if gcloud compute images describe "$CUSTOM_IMAGE_NAME" --format="value(name)" 2>/dev/null >/dev/null; then
        IMAGE_ARGS=("--image=$CUSTOM_IMAGE_NAME")
        echo "[provision] Found custom image $CUSTOM_IMAGE_NAME! Fast boot enabled."
    else
        IMAGE_ARGS=(
            "--image-family=common-cu129-ubuntu-2204-nvidia-580"
            "--image-project=deeplearning-platform-release"
            "--metadata=install-nvidia-driver=True"
        )
    fi

    ZONES=(${PREFERRED_ZONE:-us-east1-c} us-east1-d us-central1-a us-central1-b us-central1-c us-central1-f us-west1-a us-west4-a)
    for Z in "${ZONES[@]}"; do
        echo "[provision] trying zone $Z..."
        if gcloud compute instances create "$VM_NAME" \
            --zone="$Z" --machine-type=g2-standard-4 \
            --accelerator=type=nvidia-l4,count=1 \
            --maintenance-policy=TERMINATE --provisioning-model="$PROVISIONING_MODEL" \
            "${IMAGE_ARGS[@]}" \
            --boot-disk-size=80GB --boot-disk-type=pd-balanced \
            --scopes=default; then
            ZONE="$Z"; break
        else
            echo "[provision]   no capacity / error in $Z, falling through..."
        fi
    done
    [[ -n "$ZONE" ]] || { echo "ERROR: no L4 capacity ($PROVISIONING_MODEL) in any tried zone"; exit 1; }
    echo "[provision] created in $ZONE ($PROVISIONING_MODEL)"
fi

# ---- wait for SSH ----
IP=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
echo "[provision] waiting for SSH at $IP..."
until nc -zw3 "$IP" 22 2>/dev/null; do sleep 5; done

# ---- push tools/ + .env ----
echo "[provision] pushing tools/, .env, and L4 config..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="mkdir -p ~/CS265"
gcloud compute scp --zone="$ZONE" --recurse tools/ "$VM_NAME":~/CS265/
gcloud compute scp --zone="$ZONE" .env "$VM_NAME":~/CS265/.env
gcloud compute scp --zone="$ZONE" .l4-config.env "$VM_NAME":~/CS265/.l4-config.env
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="chmod 600 ~/CS265/.env"

# ---- remote setup (idempotent) ----
echo "[provision] running setup on VM (apt, venv, weights, dump, fixtures)..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='
set -euo pipefail
cd ~/CS265
if [ -f .l4-config.env ]; then
    # shellcheck disable=SC1091
    source .l4-config.env
fi
CUDA_VER="${CUDA_VER:-12.9}"
echo "[setup] waiting for nvidia driver..."
until nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; do sleep 5; done
echo "[setup] apt..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential python3-pip python3-venv
echo "[setup] venv..."
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet huggingface_hub safetensors numpy
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
echo "[setup] loading HF token from .env..."
set -a; source .env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN}"
[ -n "$HF_TOKEN" ] || { echo "ERROR: HF token empty"; exit 1; }
echo "[setup] downloader..."
if [ ! -f assets/llama3/model-00004-of-00004.safetensors ]; then
    python3 tools/llama3_downloader.py --out ./assets/llama3/
else
    echo "  model shards present; skipping"
fi
echo "[setup] dumper..."
if [ ! -f assets/llama3/dump/embeddings.bin ]; then
    python3 tools/dumper.py
else
    echo "  dump present; skipping"
fi
echo "[setup] token.model..."
if [ ! -f assets/llama3/token.model ]; then
    python3 tools/gen_token_model.py
else
    echo "  token.model present; skipping"
fi
echo "[setup] m2m3 fixtures..."
if [ ! -f tests/data/m2m3/embeddings_hello.bin ]; then
    python3 tools/gen_m2m3_fixtures.py
else
    echo "  m2m3 fixtures present; skipping"
fi
echo "[setup] shell env..."
if ! grep -q "CS265 L4 environment" ~/.bashrc; then
    {
        echo ""
        echo "# CS265 L4 environment"
        echo "export PATH=/usr/local/cuda-${CUDA_VER}/bin:\$PATH"
        echo "if [ -f \"\$HOME/CS265/.env\" ]; then"
        echo "    set -a"
        echo "    . \"\$HOME/CS265/.env\""
        echo "    set +a"
        echo "fi"
    } >> ~/.bashrc
else
    echo "  ~/.bashrc already configured"
fi
echo "[setup] OK"
'

echo
echo "[provision] DONE  VM=$VM_NAME  zone=$ZONE  ip=$IP"
echo "[provision] next: ./tools/test_l4.sh"
