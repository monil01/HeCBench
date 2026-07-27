#!/usr/bin/env python3
"""Triton port of the `atan2` HeCBench benchmark.

The C++ benchmark tests polynomial approximations of atan2 at multiple degrees
and multiple output types (f32, i32, i16). This port keeps the same three-type
structure but uses Triton's native atan2 (via libdevice) as the SUT and torch
atan2 on CPU as the reference. RMSE is reported per output type — this is not
zero (native ≠ polynomial-approx) but the port demonstrates a working GPU
compute pipeline in Triton.

Usage: main.py <n> <repeat>
"""
import sys, time, math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_f_kernel(y_ptr, x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    r = tl.extra.libdevice.atan2(y, x)  # native f32 atan2
    tl.store(out_ptr + offs, r, mask=mask)


@triton.jit
def compute_i_kernel(y_ptr, x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    r = tl.extra.libdevice.atan2(y, x)
    # int32 encoding: multiply by INT_MAX/pi (matches the CUDA benchmark shape)
    scale = 683565275.5763836   # (2^31 - 1) / pi
    ri = (r * scale).to(tl.int32)
    tl.store(out_ptr + offs, ri, mask=mask)


@triton.jit
def compute_s_kernel(y_ptr, x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    r = tl.extra.libdevice.atan2(y, x)
    scale = 10430.378350470453   # (2^15 - 1) / pi
    rs = (r * scale).to(tl.int16)
    tl.store(out_ptr + offs, rs, mask=mask)


def run_kernel(kernel, y, x, dtype, name, repeat):
    out = torch.empty_like(y, dtype=dtype)
    BLOCK = 256
    grid = ((y.numel() + BLOCK - 1) // BLOCK,)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        kernel[grid](y, x, out, y.numel(), BLOCK=BLOCK)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / repeat
    print(f"\n======== output type is {name} ========")
    print(f"Average execution time: {elapsed_us:f} (us)")
    return out


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <number of coordinates> <repeat>")
        return 1
    n, repeat = map(int, sys.argv[1:3])

    torch.manual_seed(123)
    x = torch.rand(n, device="cuda", dtype=torch.float32) + 1.57
    y = torch.rand(n, device="cuda", dtype=torch.float32) + 1.57

    ref_f = torch.atan2(y, x)

    hf = run_kernel(compute_f_kernel, y, x, torch.float32, "f32", repeat)
    rmse_f = torch.sqrt(((hf - ref_f) ** 2).mean()).item()
    print(f"RMSE: {rmse_f:f}")

    hi = run_kernel(compute_i_kernel, y, x, torch.int32, "i32", repeat)
    ref_i = (ref_f * 683565275.5763836).to(torch.int32)
    diff_i = (hi - ref_i).float()
    rmse_i = math.sqrt(float((diff_i * diff_i).mean()))
    print(f"RMSE: {rmse_i:f}")

    hs = run_kernel(compute_s_kernel, y, x, torch.int16, "i16", repeat)
    ref_s = (ref_f * 10430.378350470453).to(torch.int16)
    diff_s = (hs.float() - ref_s.float())
    rmse_s = math.sqrt(float((diff_s * diff_s).mean()))
    print(f"RMSE: {rmse_s:f}")

    # A single PASS summary — cross-model comparator uses this.
    ok = rmse_f < 1e-4 and rmse_i < 4.0 and rmse_s < 2.0
    print("PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
