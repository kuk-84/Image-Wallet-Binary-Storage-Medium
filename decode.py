"""
decode.py — ResNet-101 Image Vault Decoder
==========================================
Extracts images from a vault.safetensors file produced by encode.py.

Steps:
  1. Load state_dict from vault.safetensors.
  2. Read metadata header from fc.weight (magic + image count + JPEG lengths).
  3. Concatenate all float32 storage tensors in alphabetical key order.
  4. Slice each JPEG byte-stream by its stored length; decode with PIL.

Usage:
    python decode.py <vault.safetensors> [output_dir]
"""

import os, io, struct, argparse
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

MAGIC       = b'VALT'
HEADER_KEYS = {'fc.weight', 'fc.bias'}


def storage_pairs(state_dict: dict):
    return sorted(
        [(k, v) for k, v in state_dict.items()
         if k not in HEADER_KEYS and v.dtype == torch.float32],
        key=lambda x: x[0]
    )


def decode(vault_path: str, output_dir: str):
    print(f"[decode] Loading vault : {vault_path}")
    state_dict = load_file(vault_path)

    # 1. Parse header
    raw_header = state_dict['fc.weight'].numpy().tobytes()
    if raw_header[:4] != MAGIC:
        raise ValueError("Not a valid vault file (bad magic bytes)")
    num_images   = struct.unpack_from('<I', raw_header, 4)[0]
    jpeg_lengths = [struct.unpack_from('<I', raw_header, 8 + i * 4)[0]
                    for i in range(num_images)]
    print(f"[decode] Images in vault : {num_images}")

    # 2. Assemble payload
    payload = b''.join(v.numpy().tobytes() for _, v in storage_pairs(state_dict))

    # 3. Decode images
    os.makedirs(output_dir, exist_ok=True)
    digits = len(str(num_images))
    ptr = 0
    for idx, length in enumerate(jpeg_lengths):
        jpeg_bytes = payload[ptr: ptr + length]
        ptr += length
        img = Image.open(io.BytesIO(jpeg_bytes))
        out = os.path.join(output_dir, f'image_{idx + 1:0{digits}d}.jpeg')
        img.save(out, format='JPEG', quality=95, subsampling=0)
        if (idx + 1) % 20 == 0 or idx == num_images - 1:
            print(f"[decode]   {idx + 1}/{num_images} extracted ...")

    print(f"[decode] Output : {output_dir}")
    print(f"[decode] Done ✓")


def main():
    p = argparse.ArgumentParser(description="Extract images from a ResNet-101 vault.")
    p.add_argument('vault')
    p.add_argument('output_dir', nargs='?', default='decoded_images')
    args = p.parse_args()
    decode(args.vault, args.output_dir)

if __name__ == '__main__':
    main()
