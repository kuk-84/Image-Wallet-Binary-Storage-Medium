import os, sys, io, struct, argparse
import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch
import torchvision.models as models

MAGIC= b'VALT'
HEADER_KEYS={'fc.weight', 'fc.bias'}
DEFAULT_QUALITY=95
SUPPORTED_EXTS ={'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def jpeg_encode(img: 'Image.Image', quality: int) -> bytes:
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=quality,
                            subsampling=0, optimize=True)
    return buf.getvalue()


def storage_pairs(state_dict: dict):
    """All float32 tensors except fc, sorted alphabetically."""
    return sorted(
        [(k, v) for k, v in state_dict.items()
         if k not in HEADER_KEYS and v.dtype == torch.float32],
        key=lambda x: x[0]
    )


def encode(image_dir: str, output_path: str, quality: int = DEFAULT_QUALITY):
    print(f"[encode] Source directory : {image_dir}")
    print(f"[encode] JPEG quality     : {quality}")

    # 1. Blank ResNet-101
    print("[encode] Building ResNet-101 skeleton ...")
    model = models.resnet101(weights=None)
    state_dict = model.state_dict()

    pairs = storage_pairs(state_dict)
    capacity = sum(v.numel() * 4 for _, v in pairs)
    print(f"[encode] Storage capacity : {capacity / 1e6:.2f} MB")

    # 2. JPEG-compress images
    all_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )
    jpeg_buffers, used, skipped = [], 0, 0
    for fname in all_files:
        encoded = jpeg_encode(Image.open(os.path.join(image_dir, fname)), quality)
        if used + len(encoded) > capacity:
            skipped += 1
            continue
        jpeg_buffers.append(encoded)
        used += len(encoded)
    print(f"[encode] Images stored    : {len(jpeg_buffers)}  (skipped {skipped})")

    # 3. Pack bytes → float32 tensors (alphabetical key order)
    raw = b''.join(jpeg_buffers)
    padded = raw + b'\x00' * (capacity - len(raw))
    src = np.frombuffer(padded, dtype=np.uint8)

    ptr = 0
    for key, tensor in pairs:
        n = tensor.numel() * 4
        chunk = src[ptr: ptr + n].copy().view(np.float32).reshape(tensor.shape)
        state_dict[key] = torch.from_numpy(chunk)
        ptr += n

    # 4. Header in fc.weight  [magic(4) | N(4) | len_0(4) | ... | len_N-1(4)]
    header = MAGIC + struct.pack('<I', len(jpeg_buffers))
    for b in jpeg_buffers:
        header += struct.pack('<I', len(b))
    fc_w = state_dict['fc.weight']
    fc_cap = fc_w.numel() * 4
    padded_h = header + b'\x00' * (fc_cap - len(header))
    state_dict['fc.weight'] = torch.from_numpy(
        np.frombuffer(padded_h, dtype=np.float32).copy()
    ).reshape(fc_w.shape)

    # 5. Save
    save_file(state_dict, output_path)
    vault_mb = os.path.getsize(output_path) / 1e6
    E = used / (vault_mb * 1e6)
    print(f"[encode] Vault saved      : {output_path}  ({vault_mb:.2f} MB)")
    print(f"[encode] Efficiency E     : {E:.4f}")
    print(f"[encode] Done ✓")


def main():
    p = argparse.ArgumentParser(description="Pack images into a ResNet-101 vault.")
    p.add_argument('image_dir')
    p.add_argument('output', nargs='?', default='vault.safetensors')
    p.add_argument('--quality', type=int, default=DEFAULT_QUALITY)
    args = p.parse_args()
    encode(args.image_dir, args.output, args.quality)

if __name__ == '__main__':
    main()
