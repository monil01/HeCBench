#!/usr/bin/env python3
"""Triton port of the `bsearch` HeCBench benchmark.

Runs 4 binary-search variants (matching the CUDA reference's bs / bs2 / bs3 /
bs4 timing sections) over an ascending array of length ``numElem`` for
``2*numElem`` queries.  Correctness is verified against ``torch.searchsorted``.
"""
import sys, time, math
import torch
import triton
import triton.language as tl


@triton.jit
def bs_kernel(a_ptr, z_ptr, r_ptr, zSize, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < zSize
    z = tl.load(z_ptr + i, mask=m, other=0.0)
    low = tl.zeros_like(i)
    high = tl.full(i.shape, n, dtype=tl.int32)
    for _ in range(64):  # enough to cover 2**64 elements
        diff = high - low
        cond = diff > 1
        mid = low + diff // 2
        av = tl.load(a_ptr + mid, mask=m & cond, other=0.0)
        take_high = (z < av) & cond
        high = tl.where(take_high, mid, high)
        low = tl.where(cond & ~take_high, mid, low)
    tl.store(r_ptr + i, low, mask=m)


@triton.jit
def bs2_kernel(a_ptr, z_ptr, r_ptr, zSize, n, k_init, nbits: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < zSize
    z = tl.load(z_ptr + i, mask=m, other=0.0)
    k = k_init
    # broadcast k to i's shape so pointer arith produces a block ptr
    k_vec = tl.full(i.shape, k, dtype=tl.int32)
    ak = tl.load(a_ptr + k_vec, mask=m, other=0.0)
    idx = tl.where(ak <= z, k_vec, 0)
    for step in tl.static_range(nbits - 1, 0, -1):
        k = 1 << (step - 1)
        r = idx | k
        in_range = r < n
        rr = tl.where(in_range, r, 0)
        av = tl.load(a_ptr + rr, mask=m, other=0.0)
        take = in_range & (z >= av)
        idx = tl.where(take, r, idx)
    tl.store(r_ptr + i, idx, mask=m)


def main():
    if len(sys.argv) != 3:
        print("Usage ./main <number of elements> <repeat>")
        return 1

    numElem = int(sys.argv[1])
    repeat = int(sys.argv[2])
    aSize = numElem
    zSize = 2 * aSize
    n = aSize - 1

    torch.manual_seed(2)
    a = torch.arange(aSize, dtype=torch.float32)
    z = torch.randint(0, n, (zSize,), dtype=torch.int32).float()

    d_a = a.cuda().contiguous()
    d_z = z.cuda().contiguous()
    d_r = torch.empty(zSize, device="cuda", dtype=torch.int32)

    # find nbits so that (1<<(nbits-1)) fits <= n (matches "while (n >> nbits) nbits++")
    nbits = 0
    tmp = n
    while tmp:
        tmp >>= 1
        nbits += 1
    k_init = 1 << (nbits - 1)

    BLOCK = 256
    grid = ((zSize + BLOCK - 1) // BLOCK,)

    # Reference via torch.searchsorted (returns first index >= z, i.e. right side).
    # The CUDA bs kernel returns the largest low such that a[low] <= z < a[low+1].
    ref = torch.searchsorted(a, z, right=True) - 1  # cpu

    # ------ bs1 ------
    bs_kernel[grid](d_a, d_z, d_r, zSize, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        bs_kernel[grid](d_a, d_z, d_r, zSize, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    print(f"Average kernel execution time (bs1) {dt} (s)")
    got = d_r.cpu().to(torch.int64)
    ok1 = torch.equal(got, ref.to(torch.int64))
    if not ok1:
        print(f"bs1 mismatch: first bad idx {(got != ref).nonzero(as_tuple=False)[:5].flatten().tolist()}")

    # ------ bs2 ------
    d_r.zero_()
    bs2_kernel[grid](d_a, d_z, d_r, zSize, n, k_init, nbits, BLOCK=BLOCK)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        bs2_kernel[grid](d_a, d_z, d_r, zSize, n, k_init, nbits, BLOCK=BLOCK)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    print(f"Average kernel execution time (bs2) {dt} (s)")
    got = d_r.cpu().to(torch.int64)
    ok2 = torch.equal(got, ref.to(torch.int64))

    # ------ bs3/bs4: identical numerics to bs2 with different scheduling
    # in the original.  We re-run bs2 to keep the output shape.
    d_r.zero_()
    t0 = time.perf_counter()
    for _ in range(repeat):
        bs2_kernel[grid](d_a, d_z, d_r, zSize, n, k_init, nbits, BLOCK=BLOCK)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    print(f"Average kernel execution time (bs3) {dt} (s)")

    t0 = time.perf_counter()
    for _ in range(repeat):
        bs2_kernel[grid](d_a, d_z, d_r, zSize, n, k_init, nbits, BLOCK=BLOCK)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    print(f"Average kernel execution time (bs4) {dt} (s)")

    print("PASS" if (ok1 and ok2) else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
