# Runbook: Testing on GCP L4

## One-time setup

```bash
cp .l4-config.env.example .l4-config.env   # edit values
# .env at repo root must contain HUGGINGFACE_TOKEN=hf_...
./tools/provision_l4.sh
```

Provisions the L4 VM (iterates zones for capacity), pushes `tools/`, downloads weights from HuggingFace, runs the dumper, generates `token.model`, and generates M2-3 fixtures. Idempotent — safe to rerun.

`PROVISIONING_MODEL` in `.l4-config.env` selects:
- `SPOT` (default, ~$0.20–0.30/hr) — preemptible; GCP can reclaim mid-build.
- `STANDARD` (~$0.72/hr) — no preemption; use when a clean ~15 min provision matters.

## Fast boot via custom image

After the first successful warm provision **and** a `tools/test_l4.sh` build (so `bin/llm` is on disk), capture the boot disk as a reusable image:

```bash
./tools/test_l4.sh --quick --no-stop       # rsync source, build, leave VM up
./tools/create_custom_image.sh             # stops VM, snapshots boot disk -> cs265-l4-image
gcloud compute instances delete cs265-l4 --zone=<zone>   # image is now the source of truth
```

Now `./tools/provision_l4.sh` boots from the image in ~60s — driver, venv, weights, dumps, and `bin/llm` are all pre-baked. Idle cost: ~$2/month for the image, no compute billing.

**Region pinning gotcha**: a custom image lives in the source disk's region. Set `PREFERRED_ZONE` in `.l4-config.env` to a zone in that region (e.g., `us-east1-d` if the image was created in `us-east1`); otherwise GCP must copy the image cross-region on the first boot, adding several minutes to "fast" boot.

## Recurring test cycle

```bash
./tools/test_l4.sh             # default quick lane: M1+fast M2-3, stop VM
./tools/test_l4.sh --unit      # M1 only
./tools/test_l4.sh --quick     # same as default; M1+fast M2-3
./tools/test_l4.sh --perf      # TODO #1 KV-cache perf/audit lane
./tools/test_l4.sh --full      # final full regression gate only
./tools/test_l4.sh --clean     # force make clean before building
./tools/test_l4.sh --no-stop   # leave VM running for follow-up
```

Detects the VM's current zone automatically. Exits non-zero if any lane fails.
Quick is the default development lane; it skips `full_forward_*` and
`layer_streaming_smoke` so expensive full-model checks do not run by accident.
Use `--full` only as a final acceptance gate.

The `--perf` lane is the inner loop for TODO #1. It builds `bin/llm`, runs
8-token resident KV-cache generation for `"The capital of France is"`, captures
the log in `build/l4-kv-cache-perf.log`, prints resident load/prefill/decode
timers, and fails if the resident path reports old streaming timers such as
`layer.load_disk_to_host`, `layer.h2d_weights`, or `layer.unload`.

## Demo lifecycle (code review)

For the code-review demo, use the wrapper scripts instead of raw gcloud. Both auto-detect the VM's current zone (no need to pass `--zone`):

```bash
./scripts/demo-start.sh [N]     # start L4, SSH in, run bin/llm --interactive --max-tokens ${N:-32}
./scripts/demo-stop.sh          # stop the VM after the demo
```

`demo-start.sh` warms up the resident BF16 weights (~3 min) once, then each prompt is ~3s. Ready signal: `[interactive] ready. max-tokens per prompt: N.`

## Tear-down

If you have a custom image, prefer **delete** (no idle cost; re-provision rebuilds the VM from the image in ~60s):

```bash
source .l4-config.env
[[ -n "${GCLOUD_PATH:-}" ]] && export PATH="$GCLOUD_PATH:$PATH"
gcloud compute instances delete "$VM_NAME" --zone=<zone> --quiet
```

Without an image, prefer **stop** instead of delete (preserves disk + setup; ~$8/month idle for 80 GB pd-balanced):

```bash
./scripts/demo-stop.sh
```

## Troubleshooting

See `docs/learnings.md` (auto-injected) for known gotchas: nvcc PATH, disk pressure, M1/M2-3 fixture sources, spot preemption.
