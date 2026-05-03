#!/usr/bin/env bash
# Recurring: push current source to L4, build, run a named test lane, stop VM.
#
# Reads config from `.l4-config.env`. Detects VM zone via gcloud (handles
# zone changes after re-provisioning).
#
# Usage:
#   ./tools/test_l4.sh             # quick cycle, stop VM at end
#   ./tools/test_l4.sh --perf      # TODO #1 KV-cache perf/audit lane
#   ./tools/test_l4.sh --full      # final full regression gate
#   ./tools/test_l4.sh --clean     # force make clean before build
#   ./tools/test_l4.sh --no-stop   # leave VM running for follow-up work

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STOP_AT_END=1
LANE="quick"
FORCE_CLEAN=0
PERF_TIMEOUT_SECONDS=300

usage() {
    cat <<'USAGE'
Usage: ./tools/test_l4.sh [--unit|--quick|--perf|--full] [--clean] [--no-stop]

  --unit      Build and run only the 7 M1 grading tests.
  --quick     Build and run M1 plus fast M2-3 tests. Default lane.
  --perf      Build and run the TODO #1 KV-cache performance/audit lane only.
  --full      Build and run M1 plus every M2-3 test. Final gate only.
  --clean     Force make clean before building.
  --no-stop   Leave the VM running after the test cycle.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit) LANE="unit" ;;
        --quick) LANE="quick" ;;
        --perf) LANE="perf" ;;
        --full) LANE="full" ;;
        --lane)
            [[ $# -ge 2 ]] || { echo "ERROR: --lane requires unit|quick|perf|full" >&2; exit 1; }
            LANE="$2"
            shift
            ;;
        --clean) FORCE_CLEAN=1 ;;
        --no-stop) STOP_AT_END=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown flag: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

case "$LANE" in
    unit|quick|perf|full) ;;
    *) echo "ERROR: unknown lane: $LANE" >&2; usage >&2; exit 1 ;;
esac

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
echo "[test] build + run on $VM_NAME ($ZONE, lane=$LANE)..."
EXIT=0
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
set -e
cd ~/CS265
export PATH=/usr/local/cuda-${CUDA_VER}/bin:\$PATH
export CUDA_PATH=/usr/local/cuda-${CUDA_VER}

JOBS=\$(nproc)
echo \"[remote] nvcc=\$(command -v nvcc)\"
echo \"[remote] jobs=\$JOBS arch=${ARCH} cuda_path=\$CUDA_PATH\"
echo \"[remote] lane=${LANE}\"

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

case '${LANE}' in
    unit)
        BUILD_TARGETS='all tests'
        ;;
    perf)
        BUILD_TARGETS='all'
        ;;
    quick|full)
        BUILD_TARGETS='all tests tests_m2m3'
        ;;
esac

make -j\"\$JOBS\" ARCH=${ARCH} CUDA_PATH=\"\$CUDA_PATH\" \$BUILD_TARGETS
mkdir -p build
printf '%s\n' '${CLEAN_FINGERPRINT}' > \"\$STAMP\"

run_m1() {
    echo
    echo '=== M1 (7 tests) ==='
    M1_FAIL=0
    for i in 1 2 3 4 5 6 7; do
        OUT=\$(./bin/tests \$i 2>&1 | tail -1)
        echo \"  test \$i: \$OUT\"
        echo \"\$OUT\" | grep -q PASSED || M1_FAIL=\$((M1_FAIL+1))
    done
}

run_m2m3() {
    local mode=\"\$1\"
    echo
    if [ \"\$mode\" = quick ]; then
        echo '=== M2-3 quick (skipping full_forward_* and layer_streaming_smoke) ==='
        M2_TESTS=\$(./bin/tests_m2m3 --list | grep -Ev '^  (full_forward_|layer_streaming_smoke$)')
    else
        echo '=== M2-3 full (all tests; final gate only) ==='
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
}

run_kv_perf() {
    echo
    echo '=== PERF: TODO #1 KV cache resident 8-token audit ==='
    LOG=build/l4-kv-cache-perf.log
    rm -f \"\$LOG\"

    set +e
    /usr/bin/time -f '[perf] wall_seconds=%e' \
        timeout ${PERF_TIMEOUT_SECONDS}s \
        ./bin/llm --max-tokens 8 'The capital of France is' >\"\$LOG\" 2>&1
    STATUS=\$?
    set -e

    tail -120 \"\$LOG\"
    if [ \$STATUS -ne 0 ]; then
        echo \"FAIL kv_cache_perf: command exited with status \$STATUS\"
        exit \$STATUS
    fi

    FORBIDDEN=\$(grep -E '^[[:space:]]+(layer\\.load_disk_to_host|layer\\.h2d_weights|layer\\.unload)[[:space:]]' \"\$LOG\" || true)
    if [ -n \"\$FORBIDDEN\" ]; then
        echo
        echo 'FAIL kv_cache_perf: resident decode hit forbidden streaming timers'
        printf '%s\n' \"\$FORBIDDEN\"
        exit 1
    fi

    echo
    echo '=== PERF SUMMARY FIELDS ==='
    grep -E '^[[:space:]]+(weights\\.load_all_resident_bf16|step\\.prefill|step\\.decode|generate\\.total|lm_head\\.cpu)[[:space:]]' \"\$LOG\" || true
    echo 'PASS kv_cache_perf'
}

M1_FAIL=0
M2_FAIL=0
M2_COUNT=0

case '${LANE}' in
    unit)
        run_m1
        echo
        echo \"=== SUMMARY: lane=unit  M1 fail=\$M1_FAIL ===\"
        [ \$M1_FAIL -eq 0 ]
        ;;
    quick)
        run_m1
        run_m2m3 quick
        echo
        echo \"=== SUMMARY: lane=quick  M1 fail=\$M1_FAIL  M2-3 fail=\$M2_FAIL  M2-3 ran=\$M2_COUNT ===\"
        [ \$M1_FAIL -eq 0 ] && [ \$M2_FAIL -eq 0 ]
        ;;
    perf)
        run_kv_perf
        ;;
    full)
        run_m1
        run_m2m3 full
        echo
        echo \"=== SUMMARY: lane=full  M1 fail=\$M1_FAIL  M2-3 fail=\$M2_FAIL  M2-3 ran=\$M2_COUNT ===\"
        [ \$M1_FAIL -eq 0 ] && [ \$M2_FAIL -eq 0 ]
        ;;
esac
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
