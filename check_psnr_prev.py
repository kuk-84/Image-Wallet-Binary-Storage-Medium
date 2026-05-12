"""
check_psnr.py — Compare decoded images vs originals and report PSNR
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

    orig_files    = sorted(f for f in os.listdir(orig_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    decoded_files = sorted(f for f in os.listdir(decoded_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    n = min(len(orig_files), len(decoded_files))
    print(f"Comparing {n} image pairs...\n")
    print(f"{'#':<5} {'Original':<25} {'Decoded':<25} {'PSNR':>10}")
    print("-" * 70)

    psnrs = []
    for i in range(n):
        o = np.array(Image.open(os.path.join(orig_dir,    orig_files[i])).convert('RGB'))
        d = np.array(Image.open(os.path.join(decoded_dir, decoded_files[i])).convert('RGB'))

        if o.shape != d.shape:
            print(f"{i+1:<5} {orig_files[i]:<25} {decoded_files[i]:<25} {'size mismatch':>10}")
            continue

        p = psnr(o, d)
        psnrs.append(p)
        p_str = f"{p:.2f} dB" if p != float('inf') else "inf (lossless)"
        print(f"{i+1:<5} {orig_files[i]:<25} {decoded_files[i]:<25} {p_str:>15}")

    if psnrs:
        finite = [p for p in psnrs if p != float('inf')]
        print("-" * 70)
        print(f"\nSummary ({len(psnrs)} images):")
        print(f"  Average PSNR : {np.mean(finite):.2f} dB" if finite else "  All lossless!")
        print(f"  Min PSNR     : {min(finite):.2f} dB"     if finite else "")
        print(f"  Max PSNR     : {max(finite):.2f} dB"     if finite else "")
        print(f"  Lossless     : {len(psnrs) - len(finite)}/{len(psnrs)} images")

if __name__ == '__main__':
    main()
