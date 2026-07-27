#!/usr/bin/env python3
"""Triton port of the `all-pairs-distance` HeCBench benchmark.

Computes an all-pairs Hamming distance matrix over 224 instances x 4096
byte-valued attributes. Matches the three CUDA kernel variants (k1 atomic
scatter, k2 shared-mem reduction, k3 CUB block reduction) with functionally
equivalent Triton kernels — same PASS/FAIL structure, three PASS lines.

Usage: main.py <iterations>
"""
import sys, time
import torch
import triton
import triton.language as tl

INSTANCES = 224
ATTRIBUTES = 4096


@triton.jit
def k1_atomic(data_ptr, dist_ptr, THREADS: tl.constexpr,
              INSTANCES: tl.constexpr, ATTRIBUTES: tl.constexpr):
    gx = tl.program_id(0)
    gy = tl.program_id(1)
    idx = tl.arange(0, THREADS)

    row_x = data_ptr + gx * ATTRIBUTES
    row_y = data_ptr + gy * ATTRIBUTES
    # process 4 attrs per thread across ATTRIBUTES, mimic char4 access
    step = THREADS * 4
    i = idx * 4
    acc = tl.zeros((THREADS,), dtype=tl.int32)
    for base in range(0, ATTRIBUTES, step):
        off = base + i
        mask = off < ATTRIBUTES
        # unroll 4 lanes
        a0 = tl.load(row_x + off + 0, mask=mask & (off + 0 < ATTRIBUTES), other=0)
        b0 = tl.load(row_y + off + 0, mask=mask & (off + 0 < ATTRIBUTES), other=0)
        a1 = tl.load(row_x + off + 1, mask=mask & (off + 1 < ATTRIBUTES), other=0)
        b1 = tl.load(row_y + off + 1, mask=mask & (off + 1 < ATTRIBUTES), other=0)
        a2 = tl.load(row_x + off + 2, mask=mask & (off + 2 < ATTRIBUTES), other=0)
        b2 = tl.load(row_y + off + 2, mask=mask & (off + 2 < ATTRIBUTES), other=0)
        a3 = tl.load(row_x + off + 3, mask=mask & (off + 3 < ATTRIBUTES), other=0)
        b3 = tl.load(row_y + off + 3, mask=mask & (off + 3 < ATTRIBUTES), other=0)
        cnt = ((a0 != b0).to(tl.int32) + (a1 != b1).to(tl.int32)
               + (a2 != b2).to(tl.int32) + (a3 != b3).to(tl.int32))
        acc += cnt
    # scatter-add into distance[INSTANCES*gx + gy] once per thread
    tl.atomic_add(dist_ptr + INSTANCES * gx + gy, tl.sum(acc))


@triton.jit
def k2_shared(data_ptr, dist_ptr, THREADS: tl.constexpr,
              INSTANCES: tl.constexpr, ATTRIBUTES: tl.constexpr):
    gx = tl.program_id(0)
    gy = tl.program_id(1)
    idx = tl.arange(0, THREADS)
    row_x = data_ptr + gx * ATTRIBUTES
    row_y = data_ptr + gy * ATTRIBUTES
    step = THREADS * 4
    i = idx * 4
    acc = tl.zeros((THREADS,), dtype=tl.int32)
    for base in range(0, ATTRIBUTES, step):
        off = base + i
        m = off + 3 < ATTRIBUTES
        a0 = tl.load(row_x + off + 0, mask=m, other=0)
        b0 = tl.load(row_y + off + 0, mask=m, other=0)
        a1 = tl.load(row_x + off + 1, mask=m, other=0)
        b1 = tl.load(row_y + off + 1, mask=m, other=0)
        a2 = tl.load(row_x + off + 2, mask=m, other=0)
        b2 = tl.load(row_y + off + 2, mask=m, other=0)
        a3 = tl.load(row_x + off + 3, mask=m, other=0)
        b3 = tl.load(row_y + off + 3, mask=m, other=0)
        acc += ((a0 != b0).to(tl.int32) + (a1 != b1).to(tl.int32)
                + (a2 != b2).to(tl.int32) + (a3 != b3).to(tl.int32))
    total = tl.sum(acc)
    # k2 wrote distance[INSTANCES*gy + gx]
    tl.store(dist_ptr + INSTANCES * gy + gx, total)


@triton.jit
def k3_reduce(data_ptr, dist_ptr, THREADS: tl.constexpr,
              INSTANCES: tl.constexpr, ATTRIBUTES: tl.constexpr):
    gx = tl.program_id(0)
    gy = tl.program_id(1)
    idx = tl.arange(0, THREADS)
    row_x = data_ptr + gx * ATTRIBUTES
    row_y = data_ptr + gy * ATTRIBUTES
    step = THREADS * 4
    i = idx * 4
    acc = tl.zeros((THREADS,), dtype=tl.int32)
    for base in range(0, ATTRIBUTES, step):
        off = base + i
        m = off + 3 < ATTRIBUTES
        a0 = tl.load(row_x + off + 0, mask=m, other=0)
        b0 = tl.load(row_y + off + 0, mask=m, other=0)
        a1 = tl.load(row_x + off + 1, mask=m, other=0)
        b1 = tl.load(row_y + off + 1, mask=m, other=0)
        a2 = tl.load(row_x + off + 2, mask=m, other=0)
        b2 = tl.load(row_y + off + 2, mask=m, other=0)
        a3 = tl.load(row_x + off + 3, mask=m, other=0)
        b3 = tl.load(row_y + off + 3, mask=m, other=0)
        acc += ((a0 != b0).to(tl.int32) + (a1 != b1).to(tl.int32)
                + (a2 != b2).to(tl.int32) + (a3 != b3).to(tl.int32))
    total = tl.sum(acc)
    tl.store(dist_ptr + INSTANCES * gy + gx, total)


def cpu_reference(data_int: torch.Tensor) -> torch.Tensor:
    """data_int: [INSTANCES, ATTRIBUTES] int32. Returns [INSTANCES, INSTANCES] int32."""
    d = data_int
    # We want distance[i, j] = sum over k of (d[i,k] != d[j,k]).
    # But the CUDA CPU baseline uses distance[i + INSTANCES*j] i.e. col-major.
    # Compute both, use the natural [i, j].
    # Note: original memcmp does bytewise identical compare so no orientation
    # bug — as long as the (i,j) pair is unique and consistent.
    n = d.shape[0]
    diff = (d.unsqueeze(1) != d.unsqueeze(0)).sum(dim=2).to(torch.int32)
    return diff


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <iterations>")
        return 1
    iterations = int(sys.argv[1])

    torch.manual_seed(2)
    # random() % 3 -> values in {0,1,2}
    data = torch.randint(0, 3, (INSTANCES, ATTRIBUTES),
                         device="cuda", dtype=torch.int8)

    # CPU distance (using int version, values are equal <=> chars are equal)
    t0 = time.perf_counter()
    cpu_dist = cpu_reference(data.to(torch.int32).cpu())
    cpu_us = (time.perf_counter() - t0) * 1e6
    print(f"CPU time: {cpu_us:f} (us)")

    THREADS = 128
    grid = (INSTANCES, INSTANCES)
    dist_dev = torch.zeros((INSTANCES, INSTANCES), device="cuda", dtype=torch.int32)
    # data as flat uint8 (kernels treat as bytes)
    data_flat = data.reshape(-1).contiguous().view(torch.uint8)

    # ---- k1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        dist_dev.zero_()
        k1_atomic[grid](data_flat, dist_dev, THREADS=THREADS,
                        INSTANCES=INSTANCES, ATTRIBUTES=ATTRIBUTES)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(f"Average kernel execution time: {us:f} (us)")
    # k1 wrote distance[INSTANCES*gx + gy] -> [gx, gy]
    ok1 = torch.equal(dist_dev.cpu(), cpu_dist)
    print("PASS" if ok1 else "FAIL")

    # ---- k2
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        dist_dev.zero_()
        k2_shared[grid](data_flat, dist_dev, THREADS=THREADS,
                        INSTANCES=INSTANCES, ATTRIBUTES=ATTRIBUTES)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(f"Average kernel execution time: {us:f} (us)")
    # k2 wrote distance[INSTANCES*gy + gx] -> [gy, gx]  (transposed)
    ok2 = torch.equal(dist_dev.t().contiguous().cpu(), cpu_dist)
    print("PASS" if ok2 else "FAIL")

    # ---- k3
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        dist_dev.zero_()
        k3_reduce[grid](data_flat, dist_dev, THREADS=THREADS,
                        INSTANCES=INSTANCES, ATTRIBUTES=ATTRIBUTES)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(f"Average kernel execution time: {us:f} (us)")
    ok3 = torch.equal(dist_dev.t().contiguous().cpu(), cpu_dist)
    print("PASS" if ok3 else "FAIL")

    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
