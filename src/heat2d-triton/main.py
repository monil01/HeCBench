#!/usr/bin/env python3
"""Triton port of the `heat2d` HeCBench benchmark.

Periodic 2D Laplace update:
    out = norm * in + delta * (E + W + N + S neighbors, wrap-around)

Runs niter iterations and cross-checks against a torch-only CPU reference
that uses the exact same wrap-around 5-point stencil.
"""
import sys, time
import torch
import triton
import triton.language as tl


NTX = 16
NTY = 16


@triton.jit
def lapl_kernel(out_ptr, in_ptr, delta, norm, Lx, Ly, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < Lx * Ly
    x = off % Lx
    y = off // Lx
    xp = (x + 1) % Lx
    xm = (x + Lx - 1) % Lx
    yp = (y + 1) % Ly
    ym = (y + Ly - 1) % Ly
    v00 = tl.load(in_ptr + y * Lx + x, mask=m, other=0.0)
    v0p = tl.load(in_ptr + y * Lx + xp, mask=m, other=0.0)
    v0m = tl.load(in_ptr + y * Lx + xm, mask=m, other=0.0)
    vp0 = tl.load(in_ptr + yp * Lx + x, mask=m, other=0.0)
    vm0 = tl.load(in_ptr + ym * Lx + x, mask=m, other=0.0)
    tl.store(out_ptr + y * Lx + x, norm * v00 + delta * (v0p + v0m + vp0 + vm0), mask=m)


def cpu_iter(inp: torch.Tensor, delta: float, norm: float) -> torch.Tensor:
    up = torch.roll(inp, shifts=-1, dims=0)
    dn = torch.roll(inp, shifts=1, dims=0)
    rt = torch.roll(inp, shifts=-1, dims=1)
    lf = torch.roll(inp, shifts=1, dims=1)
    return norm * inp + delta * (up + dn + rt + lf)


def main():
    if len(sys.argv) != 4:
        print(f" Usage: {sys.argv[0]} LX LY NITER")
        return 1
    Lx = int(sys.argv[1]); Ly = int(sys.argv[2]); niter = int(sys.argv[3])
    if Lx % NTX or Ly % NTY:
        print(f"Array length LX and LY must be a multiple of block size {NTX} and {NTY}, respectively")
        return 1

    sigma = 0.01
    xdelta = sigma / (1.0 + 4.0 * sigma)
    xnorm  = 1.0 / (1.0 + 4.0 * sigma)

    print(f" Ly,Lx = {Ly},{Lx}")
    print(f" niter = {niter}")

    torch.manual_seed(123)
    buf = torch.zeros(Ly, Lx, dtype=torch.float32)
    xs = torch.randint(0, Lx, (Lx // 16,))
    ys = torch.randint(0, Ly, (Ly // 16,))
    for x in xs.tolist():
        buf[:, x] = 1.0
    for y in ys.tolist():
        buf[y, :] = 1.0

    # CPU reference (torch on CPU)
    h_in = buf.clone()
    for _ in range(niter):
        h_out = cpu_iter(h_in, xdelta, xnorm)
        h_in, h_out = h_out, h_in

    # GPU
    d_in = buf.cuda().contiguous()
    d_out = torch.empty_like(d_in)

    total = Lx * Ly
    BLOCK = NTX * NTY
    grid = ((total + BLOCK - 1) // BLOCK,)

    # warmup
    lapl_kernel[grid](d_out, d_in, xdelta, xnorm, Lx, Ly, BLOCK=BLOCK)
    torch.cuda.synchronize()

    d_in = buf.cuda().contiguous()
    d_out = torch.empty_like(d_in)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(niter):
        lapl_kernel[grid](d_out, d_in, xdelta, xnorm, Lx, Ly, BLOCK=BLOCK)
        d_in, d_out = d_out, d_in
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / niter

    bw = Lx * Ly * 4 * 2.0 / (elapsed_us * 1e3)
    p = (Lx * Ly * 6.0) / (elapsed_us * 1e3)
    print(f"Device: iters = {niter:8d}, (Lx,Ly) = {Lx:6d}, {Ly:6d}, "
          f"t = {elapsed_us:8.1f} usec/iter, BW = {bw:6.3f} GB/s, P = {p:6.3f} Gflop/s")

    d_res = d_in.cpu()
    err = (h_in - d_res).abs().max().item()
    ok = err < 1e-2
    if not ok:
        idx = int((h_in - d_res).abs().argmax())
        print(f"Mismatch at {idx} cpu={h_in.view(-1)[idx].item()} gpu={d_res.view(-1)[idx].item()}")
    print("PASS" if ok else "FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
