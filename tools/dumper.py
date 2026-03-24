#!/usr/bin/env python3
"""Dump Llama safetensors weights into simple binary files for the C++ loader.

Reads the HuggingFace safetensors shards (via model.safetensors.index.json),
converts each tensor to a target dtype, and writes it as a flat binary file
with a 280-byte header that the C++ LlamaDumpLoader can parse directly.

Output structure:
  <out>/embeddings.bin           — embedding table
  <out>/layer_00/*.bin           — per-layer weights
  <out>/global/*.bin             — non-layer weights (e.g. final norm)
  <out>/manifest.json            — index of all dumped tensors

Usage:
  python tools/dumper.py --model-dir assets/llama3 --out assets/llama3/dump --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open

# Binary header format (little-endian):
#   256s = tensor name (null-padded), I = uint32 dtype code,
#   I = uint32 ndims, Q = uint64 shape[0], Q = uint64 shape[1]
HEADER_FMT = "<256sIIQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 280 bytes

# Maps torch dtypes to the integer codes stored in the dump header.
DTYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

# Maps CLI --dtype strings to torch dtypes.
TARGET_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model dir, output dir, and target dtype."""
    parser = argparse.ArgumentParser(
        description="Dump Llama weights to binary files and emit manifest.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("assets/llama3"),
        help="Directory containing model.safetensors.index.json and shards.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/llama3/dump"),
        help="Output directory for binary dumps and manifest.json",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Output floating-point type for dumped tensors.",
    )
    return parser.parse_args()


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    """Load the tensor-name -> shard-filename mapping from the HF index JSON."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index file not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("index json missing valid 'weight_map'")
    return weight_map


def sanitize_name(name: str) -> str:
    """Replace dots and slashes with underscores for filesystem-safe filenames."""
    safe = name.replace("/", "_").replace(".", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def tensor_output_path(out_dir: Path, tensor_name: str) -> Path:
    """Determine the output file path based on tensor name.

    - Embedding table -> <out>/embeddings.bin
    - Layer tensors   -> <out>/layer_XX/<sanitized_name>.bin
    - Other tensors   -> <out>/global/<sanitized_name>.bin
    """
    if tensor_name == "model.embed_tokens.weight":
        return out_dir / "embeddings.bin"

    # Check if this is a per-layer tensor (e.g. "model.layers.0.self_attn.q_proj.weight")
    parts = tensor_name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        try:
            layer_idx = int(parts[2])
        except ValueError:
            layer_idx = -1
        if layer_idx >= 0:
            return (
                out_dir
                / f"layer_{layer_idx:02d}"
                / f"{sanitize_name(tensor_name)}.bin"
            )

    return out_dir / "global" / f"{sanitize_name(tensor_name)}.bin"


def normalize_tensor_dtype(tensor: torch.Tensor, target: torch.dtype) -> torch.Tensor:
    """Cast tensor to the target dtype on CPU, ensuring contiguous memory layout."""
    if not torch.is_floating_point(tensor):
        raise TypeError(f"unsupported non-floating tensor dtype: {tensor.dtype}")
    return tensor.to(dtype=target, device="cpu", copy=False).contiguous()


def tensor_shape_2d(shape: torch.Size) -> Tuple[int, int, int]:
    """Extract (ndims, dim0, dim1) from a 1D or 2D tensor shape.
    For 1D tensors, dim1 is 0."""
    ndims = len(shape)
    if ndims == 1:
        return ndims, int(shape[0]), 0
    if ndims == 2:
        return ndims, int(shape[0]), int(shape[1])
    raise ValueError(f"only 1D/2D tensors are supported, got shape={tuple(shape)}")


def tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    """Get the raw bytes of a tensor in C (row-major) order."""
    return tensor.view(torch.uint8).numpy().tobytes(order="C")


def write_tensor_file(
    tensor_name: str,
    tensor: torch.Tensor,
    output_path: Path,
) -> None:
    """Write a single tensor as a binary dump file (280-byte header + raw payload)."""
    if len(tensor_name.encode("utf-8")) > 255:
        raise ValueError(f"tensor name too long for fixed header: {tensor_name}")

    ndims, dim0, dim1 = tensor_shape_2d(tensor.shape)
    dtype_code = DTYPE_CODE.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"unsupported tensor dtype for dump: {tensor.dtype}")

    # Pack the fixed-size header: name (256B, null-padded) + metadata fields.
    name_field = tensor_name.encode("utf-8").ljust(256, b"\x00")
    header = struct.pack(HEADER_FMT, name_field, dtype_code, ndims, dim0, dim1)
    payload = tensor_payload_bytes(tensor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(header)
        f.write(payload)


def dump_all_tensors(model_dir: Path, out_dir: Path, target_dtype: torch.dtype) -> None:
    """Main dump loop: iterate over all safetensor shards, convert and write each tensor.

    Also produces manifest.json with metadata for every dumped tensor.
    Verifies that the total tensor count matches the weight map (no missing tensors).
    """
    weight_map = load_weight_map(model_dir)

    # Group tensors by shard so we only open each shard file once.
    shard_to_tensors: Dict[str, List[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        shard_to_tensors.setdefault(shard_name, []).append(tensor_name)

    manifest_records: List[dict] = []
    tensor_count = 0

    for shard_name in sorted(shard_to_tensors):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"shard file not found: {shard_path}")

        # Open the safetensors shard and dump each tensor it contains.
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for tensor_name in sorted(shard_to_tensors[shard_name]):
                tensor = sf.get_tensor(tensor_name)
                tensor = normalize_tensor_dtype(tensor, target_dtype)
                output_path = tensor_output_path(out_dir, tensor_name)
                write_tensor_file(tensor_name, tensor, output_path)

                manifest_records.append(
                    {
                        "tensor_name": tensor_name,
                        "output_path": str(output_path),
                        "dtype_code": DTYPE_CODE[tensor.dtype],
                        "shape": list(tensor.shape),
                        "source_shard": shard_name,
                    }
                )
                tensor_count += 1
                print(
                    f"[dump] {tensor_name} shape={tuple(tensor.shape)} "
                    f"dtype={str(tensor.dtype)} -> {output_path}"
                )

    # Sanity check: every tensor in the weight map should have been dumped.
    if tensor_count != len(weight_map):
        raise RuntimeError(
            f"manifest coverage mismatch: dumped={tensor_count}, expected={len(weight_map)}"
        )

    # Write the manifest JSON describing the dump format and all tensor files.
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": {
            "header_size": HEADER_SIZE,
            "header_struct": "tensor_name[256],dtype_code[u32],ndims[u32],shape0[u64],shape1[u64]",
            "endianness": "little",
        },
        "tensor_count": tensor_count,
        "records": manifest_records,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[dump] wrote manifest: {manifest_path} ({tensor_count} tensors)")


def main() -> int:
    args = parse_args()
    target_dtype = TARGET_DTYPE[args.dtype]
    dump_all_tensors(args.model_dir, args.out, target_dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
