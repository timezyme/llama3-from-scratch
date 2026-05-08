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

# This is the recipe for the 280-byte header we stick in front of every binary file.
# It tells the C++ loader: "here's the tensor name, what number type it uses,
# how many dimensions it has, and how big each dimension is."
HEADER_FMT = "<256sIIQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 280 bytes

# Simple lookup: torch dtype -> integer code we write into the header.
# The C++ side reads this integer to know how to interpret the raw bytes.
DTYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

# Translates the user's --dtype flag ("bf16", etc.) into the torch type we convert to.
TARGET_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    # Read command-line flags: where the model lives, where to write output, what dtype to use.
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
    # HuggingFace splits model weights across multiple shard files.
    # The index JSON is the table of contents -- it maps each tensor name
    # (like "model.layers.0.self_attn.q_proj.weight") to the shard file that holds it.
    # We load that map so we know which file to open for each tensor.
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
    # Tensor names have dots and slashes (e.g. "model.layers.0.self_attn.q_proj.weight").
    # We swap those for underscores so the name works as a filename on disk.
    safe = name.replace("/", "_").replace(".", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def tensor_output_path(out_dir: Path, tensor_name: str) -> Path:
    # Decide where on disk this tensor should go.
    # We sort tensors into three buckets:
    #   - The embedding table gets its own top-level file.
    #   - Layer weights go into per-layer folders (layer_00/, layer_01/, ...).
    #   - Everything else (final norm, lm_head) goes into a "global" folder.
    if tensor_name == "model.embed_tokens.weight":
        return out_dir / "embeddings.bin"

    # If the name looks like "model.layers.5.something", it belongs in layer_05/.
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

    # Not an embedding and not a layer weight -- put it in the global folder.
    return out_dir / "global" / f"{sanitize_name(tensor_name)}.bin"


def normalize_tensor_dtype(tensor: torch.Tensor, target: torch.dtype) -> torch.Tensor:
    # Convert the tensor to our target type (usually BF16).
    # .contiguous() makes sure the bytes are laid out in a single flat block in memory,
    # which is important because we're about to dump them raw to a file.
    if not torch.is_floating_point(tensor):
        raise TypeError(f"unsupported non-floating tensor dtype: {tensor.dtype}")
    return tensor.to(dtype=target, device="cpu", copy=False).contiguous()


def tensor_shape_2d(shape: torch.Size) -> Tuple[int, int, int]:
    # Pull out the dimensions so we can write them into the header.
    # 1D tensors (like bias vectors) get dim1 = 0 as a placeholder.
    # 2D tensors (like weight matrices) get both dims.
    # We don't handle 3D+ because Llama weights are all 1D or 2D.
    ndims = len(shape)
    if ndims == 1:
        return ndims, int(shape[0]), 0
    if ndims == 2:
        return ndims, int(shape[0]), int(shape[1])
    raise ValueError(f"only 1D/2D tensors are supported, got shape={tuple(shape)}")


def tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    # Turn the tensor into raw bytes -- just the numbers, no metadata.
    # view(uint8) reinterprets the data as plain bytes (no conversion happens).
    # Row-major order ("C") means rows are stored one after another, which is
    # the layout the C++ loader expects.
    return tensor.view(torch.uint8).numpy().tobytes(order="C")


def write_tensor_file(
    tensor_name: str,
    tensor: torch.Tensor,
    output_path: Path,
) -> None:
    # Write one tensor to disk as: [280-byte header][raw number bytes]
    # The header tells the C++ loader what's in the file without parsing anything fancy.

    # Make sure the name fits in the 256-byte slot we have in the header.
    if len(tensor_name.encode("utf-8")) > 255:
        raise ValueError(f"tensor name too long for fixed header: {tensor_name}")

    ndims, dim0, dim1 = tensor_shape_2d(tensor.shape)
    dtype_code = DTYPE_CODE.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"unsupported tensor dtype for dump: {tensor.dtype}")

    # Build the header: pad the name to exactly 256 bytes, then pack in
    # the dtype code, number of dimensions, and the size of each dimension.
    name_field = tensor_name.encode("utf-8").ljust(256, b"\x00")
    header = struct.pack(HEADER_FMT, name_field, dtype_code, ndims, dim0, dim1)
    payload = tensor_payload_bytes(tensor)

    # Create the output folder if it doesn't exist, then write header + data.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(header)
        f.write(payload)


def dump_all_tensors(model_dir: Path, out_dir: Path, target_dtype: torch.dtype) -> None:
    # This is the main function that does all the work.
    # It loops through every tensor in the model, converts it, and writes it to disk.
    # At the end it writes a manifest.json so you can see what got dumped.

    weight_map = load_weight_map(model_dir)

    # Group tensors by which shard file they live in.
    # This way we only open each big shard file once instead of reopening it
    # for every tensor inside it.
    shard_to_tensors: Dict[str, List[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        shard_to_tensors.setdefault(shard_name, []).append(tensor_name)

    manifest_records: List[dict] = []
    tensor_count = 0

    for shard_name in sorted(shard_to_tensors):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"shard file not found: {shard_path}")

        # Open one shard file and pull out each tensor it contains.
        # For each tensor: convert to our target dtype, then write to its own binary file.
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for tensor_name in sorted(shard_to_tensors[shard_name]):
                tensor = sf.get_tensor(tensor_name)
                tensor = normalize_tensor_dtype(tensor, target_dtype)
                output_path = tensor_output_path(out_dir, tensor_name)
                write_tensor_file(tensor_name, tensor, output_path)

                # Keep a record of what we dumped so we can write the manifest later.
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

    # Make sure we didn't skip any tensors. If this count doesn't match,
    # something went wrong and the C++ loader would be missing weights.
    if tensor_count != len(weight_map):
        raise RuntimeError(
            f"manifest coverage mismatch: dumped={tensor_count}, expected={len(weight_map)}"
        )

    # Write a JSON file listing every tensor we dumped, its shape, dtype, and path.
    # This isn't needed at runtime -- it's just handy for debugging and inspection.
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
    # Parse flags, pick the target dtype, and kick off the dump.
    args = parse_args()
    target_dtype = TARGET_DTYPE[args.dtype]
    dump_all_tensors(args.model_dir, args.out, target_dtype)
    return 0


if __name__ == "__main__":
    # Using SystemExit so the script's return code gets forwarded to the shell.
    raise SystemExit(main())
