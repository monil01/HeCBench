#!/usr/bin/env python3
"""Triton port of the `histogram` HeCBench benchmark (simplified variant).

The original benchmark ships two histogram kernels (SMEM & GMEM atomic) and a
harness driven from cub/thrust that reads Targa images off disk.  Neither
CUB nor the tga assets fit into a self-contained Triton port, so we

  * generate a deterministic random 1920x1080 uchar4 image,
  * compute per-channel 256-bin histograms with a Triton atomic-add kernel,
  * verify against torch.bincount and report ``--i=<iters>`` timing.

Matches the "one PASS line" cadence expected by the harness.
"""
import sys, time
import argparse
import torch
import triton
import triton.language as tl


NUM_BINS = 256
ACTIVE_CHANNELS = 3
NUM_CHANNELS = 4  # RGBA, only first 3 counted


@triton.jit
def histogram_kernel(pixels_ptr, hist_ptr, num_pixels, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < num_pixels
    # pixels_ptr is uint8[num_pixels * 4]
    p = offs * 4
    r = tl.load(pixels_ptr + p + 0, mask=m, other=0).to(tl.int32)
    g = tl.load(pixels_ptr + p + 1, mask=m, other=0).to(tl.int32)
    b = tl.load(pixels_ptr + p + 2, mask=m, other=0).to(tl.int32)
    # atomic add into three per-channel histograms
    tl.atomic_add(hist_ptr + 0 * 256 + r, tl.where(m, 1, 0))
    tl.atomic_add(hist_ptr + 1 * 256 + g, tl.where(m, 1, 0))
    tl.atomic_add(hist_ptr + 2 * 256 + b, tl.where(m, 1, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=100)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--entropy", type=int, default=0)
    parser.add_argument("--v", action="store_true")
    args, _ = parser.parse_known_args()

    width, height, iters = args.width, args.height, args.i
    print(f"Random image: entropy-reduction({args.entropy}) width({width}) height({height})")

    torch.manual_seed(0xABCDEF)
    img = torch.randint(0, 256, (height, width, NUM_CHANNELS), dtype=torch.uint8)
    pixels = img.view(-1).cuda().contiguous()
    num_pixels = height * width
    hist = torch.zeros(ACTIVE_CHANNELS * NUM_BINS, device="cuda", dtype=torch.int32)

    BLOCK = 512
    grid = ((num_pixels + BLOCK - 1) // BLOCK,)

    # Reference on CPU via torch.bincount
    ref = torch.zeros(ACTIVE_CHANNELS, NUM_BINS, dtype=torch.int64)
    for c in range(ACTIVE_CHANNELS):
        ref[c] = torch.bincount(img[..., c].reshape(-1).to(torch.int64), minlength=NUM_BINS)

    # warmup
    hist.zero_()
    histogram_kernel[grid](pixels, hist, num_pixels, BLOCK=BLOCK)
    torch.cuda.synchronize()

    hist.zero_()
    t0 = time.perf_counter()
    for _ in range(iters):
        # reset each iter so the atomic accum stays a real histogram
        hist.zero_()
        histogram_kernel[grid](pixels, hist, num_pixels, BLOCK=BLOCK)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / iters
    print(f"Average kernel execution time: {dt_ms} (ms)")

    got = hist.view(ACTIVE_CHANNELS, NUM_BINS).cpu().to(torch.int64)
    ok = torch.equal(got, ref)
    print("PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
