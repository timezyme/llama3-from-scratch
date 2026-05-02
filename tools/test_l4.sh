#!/usr/bin/env bash
# Recurring: push current source to L4, build, run M1 + M2-3, stop VM.
#
# Reads config from `.l4-config.env`. Detects VM zone via gcloud (handles
# zone changes after re-provisioning).
#
# Usage:
#   ./tools/test_l4.sh             # full cycle, stop VM at end
#   ./tools/test_l4.sh --quick     # skip full_forward_* and layer_streaming_smoke
#   ./tools/test_l4.sh --clean     # force make clean before build
#   ./tools/test_l4.sh --no-stop   # leave VM running for follow-up work

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STOP_AT_END=1
QUICK=0
FORCE_CLEAN=0

usage() {
    cat <<'USAGE'
Usage: ./tools/test_l4.sh [--quick] [--clean] [--no-stop]

  --quick     Run M1 plus fast M2-3 tests; skip full_forward_* and layer_streaming_smoke.
  --clean     Force make clean before building.
  --no-stop   Leave the VM running after the test cycle.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1 ;;
        --clean) FORCE_CLEAN=1 ;;
        --no-stop) STOP_AT_END=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown flag: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

[[ -f .l4-config.env ]] || { echo "ERROR: .l4-config.env missing. Run ./tools/provision_l4.sh first."; exit 1; }
# shellcheck disable=SC1091
source .l4-config.env
: "${VM_NAME:?missing in .l4-config.env}"
: "${CUDA_VER:?missing in .l4-config.env}"
: "${ARCH:?missing in .l4-config.env}"
[[ -n "${GCLOUD_PATH:-}" ]] && export PATH="$GCLOUD_PATH:$PATH"
command -v gcloud >/dev/null || { echo "ERROR: gcloud not on PATH"; exit 1; }

clean_fingerprint() {
    {
        shasum Makefile config.h
        find include kernel -type f \( -name '*.h' -o -name '*.cuh' \) |
            LC_ALL=C sort |
            while IFS= read -r f; do shasum "$f"; done
    } | shasum | awk '{print $1}'
}

CLEAN_FINGERPRINT="$(clean_fingerprint)"

# ---- locate VM ----
ZONE_STATUS=$(gcloud compute instances list --filter="name=$VM_NAME" --format="value(zone.basename(),status)" 2>/dev/null | head -1)
[[ -n "$ZONE_STATUS" ]] || { echo "ERROR: VM $VM_NAME not found. Run ./tools/provision_l4.sh first."; exit 1; }
ZONE=$(echo "$ZONE_STATUS" | awk '{print $1}')
STATUS=$(echo "$ZONE_STATUS" | awk '{print $2}')

# ---- start if stopped ----
if [[ "$STATUS" == "TERMINATED" ]]; then
    echo "[test] starting $VM_NAME in $ZONE..."
    gcloud compute instances start "$VM_NAME" --zone="$ZONE"
fi

# ---- wait for SSH ----
IP=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
echo "[test] waiting for SSH at $IP..."
until nc -zw3 "$IP" 22 2>/dev/null; do sleep 5; done

# ---- push source (no assets/, no .venv/, no build artifacts) ----
echo "[test] pushing source..."
gcloud compute scp --zone="$ZONE" --recurse --scp-flag=-p \
    Makefile main.cpp config.h include src kernel tests \
    "$VM_NAME":~/CS265/ >/dev/null

# ---- build + test on VM ----
MODE="full"
[[ $QUICK -eq 1 ]] && MODE="quick"
echo "[test] build + run on $VM_NAME ($ZONE, mode=$MODE)..."
EXIT=0
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
set -e
cd ~/CS265
export PATH=/usr/local/cuda-${CUDA_VER}/bin:\$PATH
export CUDA_PATH=/usr/local/cuda-${CUDA_VER}

JOBS=\$(nproc)
echo \"[remote] nvcc=\$(command -v nvcc)\"
echo \"[remote] jobs=\$JOBS arch=${ARCH} cuda_path=\$CUDA_PATH\"

STAMP=build/.l4-clean-fingerprint
if [ ${FORCE_CLEAN} -eq 1 ]; then
    echo '[remote] make clean (--clean)'
    make clean
elif [ ! -f \"\$STAMP\" ] || [ \"\$(cat \"\$STAMP\")\" != '${CLEAN_FINGERPRINT}' ]; then
    echo '[remote] make clean (Makefile/header fingerprint changed)'
    make clean
else
    echo '[remote] reusing existing build directory'
fi

make -j\"\$JOBS\" ARCH=${ARCH} CUDA_PATH=\"\$CUDA_PATH\" all tests tests_m2m3
mkdir -p build
printf '%s\n' '${CLEAN_FINGERPRINT}' > \"\$STAMP\"

echo
echo '=== M1 (7 tests) ==='
M1_FAIL=0
for i in 1 2 3 4 5 6 7; do
    OUT=\$(./bin/tests \$i 2>&1 | tail -1)
    echo \"  test \$i: \$OUT\"
    echo \"\$OUT\" | grep -q PASSED || M1_FAIL=\$((M1_FAIL+1))
done

echo
if [ ${QUICK} -eq 1 ]; then
    echo '=== M2-3 quick (skipping full_forward_* and layer_streaming_smoke) ==='
    M2_TESTS=\$(./bin/tests_m2m3 --list | grep -Ev '^  (full_forward_|layer_streaming_smoke$)')
else
    echo '=== M2-3 (all tests) ==='
    M2_TESTS=\$(./bin/tests_m2m3 --list)
fi
M2_FAIL=0
M2_COUNT=0
for t in \$M2_TESTS; do
    M2_COUNT=\$((M2_COUNT+1))
    OUT=\$(./bin/tests_m2m3 \$t 2>&1 | grep -E '^(PASS|FAIL)' | tail -1)
    printf '  %-40s %s\n' \"\$t\" \"\$OUT\"
    echo \"\$OUT\" | grep -q '^PASS' || M2_FAIL=\$((M2_FAIL+1))
done

echo
echo \"=== SUMMARY: M1 fail=\$M1_FAIL  M2-3 fail=\$M2_FAIL  M2-3 ran=\$M2_COUNT ===\"
[ \$M1_FAIL -eq 0 ] && [ \$M2_FAIL -eq 0 ]
" || EXIT=$?

# ---- stop VM (unless --no-stop) ----
if [[ $STOP_AT_END -eq 1 ]]; then
    echo "[test] stopping $VM_NAME..."
    gcloud compute instances stop "$VM_NAME" --zone="$ZONE" --quiet >/dev/null
    echo "[test] stopped."
else
    echo "[test] VM left RUNNING (--no-stop). Stop with: gcloud compute instances stop $VM_NAME --zone=$ZONE"
fi

exit $EXIT
