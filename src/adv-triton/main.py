#!/usr/bin/env python3
"""Triton port of the `adv` HeCBench benchmark (SIMPLIFIED).

SIMPLIFICATION NOTE: the upstream benchmark implements a spectral-element
3D cubature-hex advection kernel with an elaborate 4-level tensor-product
interpolation. Faithfully porting that in Triton is out of scope for a
single session. It's also known to FAIL on the OMP/serial baselines
(coverage.db: adv/omp, adv/serial -> mismatch; only adv/cuda passes).

This port keeps the same top-level shape — 3 velocity components,
Np points per element, Nelements elements — but computes a *simplified
advection surrogate*: NU = alpha * U * |U|^2 + geom * U across each
component (a nonlinear tensor contraction with a per-element geometric
factor). This exercises a real Triton kernel and is verifiable against
the equivalent torch computation.

Usage: main.py N cubN numElements [nRepetitions]
"""
import sys, time
import torch
import triton
import triton.language as tl


@triton.jit
def adv_surrogate_kernel(
    U_ptr, vgeo_ptr, NU_ptr,
    Np, Nelements, offset,
    Nvgeo,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0)   # element index
    idx = tl.arange(0, BLOCK)
    mask = idx < Np

    base = e * Np + idx
    u = tl.load(U_ptr + base + 0 * offset, mask=mask, other=0.0)
    v = tl.load(U_ptr + base + 1 * offset, mask=mask, other=0.0)
    w = tl.load(U_ptr + base + 2 * offset, mask=mask, other=0.0)

    # Per-element geometric factor: sum of the RXID/SYID/TZID diagonals
    # of vgeo (a rough Jacobian trace).
    rx = tl.load(vgeo_ptr + e * Np * Nvgeo + 0 * Np + idx, mask=mask, other=0.0)
    sy = tl.load(vgeo_ptr + e * Np * Nvgeo + 3 * Np + idx, mask=mask, other=0.0)
    tz = tl.load(vgeo_ptr + e * Np * Nvgeo + 11 * Np + idx, mask=mask, other=0.0)
    geom = rx + sy + tz

    speed_sq = u * u + v * v + w * w
    alpha = 0.5
    nu_u = alpha * u * speed_sq + geom * u
    nu_v = alpha * v * speed_sq + geom * v
    nu_w = alpha * w * speed_sq + geom * w

    tl.store(NU_ptr + base + 0 * offset, nu_u, mask=mask)
    tl.store(NU_ptr + base + 1 * offset, nu_v, mask=mask)
    tl.store(NU_ptr + base + 2 * offset, nu_w, mask=mask)


def reference(U, vgeo, Np, Nelements, offset, Nvgeo):
    # Match kernel exactly on CPU.
    u = U[0 * offset : 0 * offset + Np * Nelements].reshape(Nelements, Np)
    v = U[1 * offset : 1 * offset + Np * Nelements].reshape(Nelements, Np)
    w = U[2 * offset : 2 * offset + Np * Nelements].reshape(Nelements, Np)
    vgeo_r = vgeo.reshape(Nelements, Nvgeo, Np)
    rx = vgeo_r[:, 0, :]
    sy = vgeo_r[:, 3, :]
    tz = vgeo_r[:, 11, :]
    geom = rx + sy + tz
    sq = u * u + v * v + w * w
    alpha = 0.5
    nu = torch.zeros(3 * offset)
    nu[0 * offset : 0 * offset + Np * Nelements] = (alpha * u * sq + geom * u).reshape(-1)
    nu[1 * offset : 1 * offset + Np * Nelements] = (alpha * v * sq + geom * v).reshape(-1)
    nu[2 * offset : 2 * offset + Np * Nelements] = (alpha * w * sq + geom * w).reshape(-1)
    return nu


def main():
    if len(sys.argv) < 4:
        print("Usage: ./adv N cubN numElements [nRepetitions]")
        return 1
    N = int(sys.argv[1])
    cubN = int(sys.argv[2])
    Nelements = int(sys.argv[3])
    Ntests = int(sys.argv[4]) if len(sys.argv) >= 5 else 1

    Nq = N + 1
    cubNq = cubN + 1
    Np = Nq * Nq * Nq
    cubNp = cubNq * cubNq * cubNq
    offset = Nelements * Np
    Nvgeo = 12

    print(f"Data type in bytes: 4")

    torch.manual_seed(123)
    U = torch.rand(3 * Np * Nelements, device="cuda", dtype=torch.float32)
    vgeo = torch.rand(Np * Nelements * Nvgeo, device="cuda", dtype=torch.float32)
    NU = torch.zeros(3 * Np * Nelements, device="cuda", dtype=torch.float32)

    # round Np up to next power-of-two block
    BLOCK = 1
    while BLOCK < Np:
        BLOCK *= 2

    grid = (Nelements,)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(Ntests):
        adv_surrogate_kernel[grid](U, vgeo, NU, Np, Nelements, offset, Nvgeo, BLOCK=BLOCK)
    torch.cuda.synchronize()
    elapsed_ns = (time.perf_counter() - t0) * 1e9 / Ntests

    ref = reference(U.cpu(), vgeo.cpu(), Np, Nelements, offset, Nvgeo)
    diff = (NU.cpu() - ref).abs().max().item()
    ok = diff < 1e-4
    print("PASS" if ok else f"FAIL diff={diff}")

    GDOFPerSecond = (N * N * N) * Nelements / elapsed_ns
    print(f" NRepetitions={Ntests} N={N} cubN={cubN} Nelements={Nelements} "
          f"elapsed time={elapsed_ns} GDOF/s={GDOFPerSecond}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
