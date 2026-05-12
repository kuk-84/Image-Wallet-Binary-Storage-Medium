"""
check_psnr.py — Compare decoded images vs originals and report PSNR

The decoded images (image_001.jpeg ...) correspond to the first N files
from the original directory in alphabetical order. This script correctly
pairs them by index rather than filename.

Usage: python check_psnr.py <original_dir> <decoded_dir>
"""

import os
import sys
import numpy as np
from PIL import Image


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


def main():
    if len(sys.argv) < 3:
        print("Usage: python check_psnr.py <original_dir> <decoded_dir>")
        sys.exit(1)

    orig_dir    = sys.argv[1]
    decoded_dir = sys.argv[2]

    orig_files = sorted(
        f for f in os.listdir(orig_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    decoded_files = sorted(
        f for f in os.listdir(decoded_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )

    # Decoded images correspond to first N originals (alphabetical order)
    n = len(decoded_files)
    orig_files = orig_files[:n]   # only compare the originals that were stored

    print(f"Originals found  : {len(sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))))}")
    print(f"Decoded found    : {n}")
    print(f"Pairs to compare : {n}\n")
    print(f"{'#':<5} {'Original':<30} {'Decoded':<25} {'PSNR':>15}")
    print("-" * 80)

    psnrs = []
    for i in range(n):
        orig_path    = os.path.join(orig_dir,    orig_files[i])
        decoded_path = os.path.join(decoded_dir, decoded_files[i])

        o = np.array(Image.open(orig_path).convert('RGB'))
        d = np.array(Image.open(decoded_path).convert('RGB'))

        if o.shape != d.shape:
            print(f"{i+1:<5} {orig_files[i]:<30} {decoded_files[i]:<25} {'size mismatch':>15}")
            continue

        p = psnr(o, d)
        psnrs.append(p)
        p_str = f"{p:.2f} dB" if p != float('inf') else "inf (lossless)"
        print(f"{i+1:<5} {orig_files[i]:<30} {decoded_files[i]:<25} {p_str:>15}")

    if psnrs:
        finite = [p for p in psnrs if p != float('inf')]
        print("-" * 80)
        print(f"\nSummary ({len(psnrs)} pairs compared):")
        if finite:
            print(f"  Average PSNR : {np.mean(finite):.2f} dB")
            print(f"  Min PSNR     : {min(finite):.2f} dB")
            print(f"  Max PSNR     : {max(finite):.2f} dB")
        print(f"  Lossless     : {len(psnrs) - len(finite)}/{len(psnrs)} images")

        print("\nPSNR Quality Guide:")
        print("  > 50 dB  → Excellent (virtually indistinguishable)")
        print("  40-50 dB → Very good")
        print("  30-40 dB → Good, minor artifacts visible")
        print("  < 30 dB  → Noticeable degradation")


if __name__ == '__main__':
    main()
