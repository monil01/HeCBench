#!/usr/bin/env python3
"""Triton port of the `saxpy-ompt` HeCBench benchmark.

y := a * x + y over n = 2**26 elements.  The CUDA reference runs six kernel
variants (hand-tiled, unrolled, etc.) to sweep grid configurations.  This
Triton port sweeps three BLOCK sizes to keep the same "multiple runs"
structure and reports one PASS line per configuration, aligning with the
original stdout shape.
"""
import sys, time
import torch
import triton
import triton.language as tl


@triton.jit
def saxpy_kernel(x_ptr, y_ptr, a, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    xv = tl.load(x_ptr + offs, mask=m)
    yv = tl.load(y_ptr + offs, mask=m)
    tl.store(y_ptr + offs, a * xv + yv, mask=m)


def main():
    N = 1 << 26
    NLUP = 32
    a = 2.0

    torch.manual_seed(0)
    x = (torch.randint(0, 32, (N,), device="cuda", dtype=torch.int32).float() / 32.0)
    y0 = (torch.randint(0, 32, (N,), device="cuda", dtype=torch.int32).float() / 32.0)

    y_ref = a * x + y0  # CPU/torch reference (executed on GPU tensors, same math)

    total_mb = 2.0 * N * 4 / (1 << 20)
    print(f"total size of x and y is {total_mb:9.1f} MB")
    print(f"tests are averaged over {NLUP:2d} loops")

    blocks = [128, 256, 1024]
    for ial, BLOCK in enumerate(blocks):
        y = y0.clone()
        grid = ((N + BLOCK - 1) // BLOCK,)

        # warmup
        saxpy_kernel[grid](x, y, a, N, BLOCK=BLOCK)
        torch.cuda.synchronize()

        y = y0.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(NLUP):
            y = y0.clone()  # reset each iter so the additive kernel stays stable
            saxpy_kernel[grid](x, y, a, N, BLOCK=BLOCK)
        torch.cuda.synchronize()
        wt = (time.perf_counter() - t0) / NLUP

        maxabserr = float((y - y_ref).abs().max().item())
        mbs = 3.0 * N * 4 / ((1 << 20) * wt)
        print(f"saxpy on accl (impl. {ial})")
        print(f"total: {mbs:9.1f} MB/s kernel: {mbs:9.1f} MB/s maxabserr = {maxabserr:9.1f}")
        print("PASS" if maxabserr < 1e-4 else "FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
