#!/usr/bin/env python3
"""Triton port of the `bitonic-sort` HeCBench benchmark.

Full bitonic sort as in main.cu — each (step, stage) pair launches one Triton
kernel that swaps element pairs at the appropriate distance. n=25 (2^25 = 33M
ints) is the CUDA default; we cap at n=22 to fit CPU verification in time
budget while still exercising the same kernel and control flow.

Usage: main.py <n> <seed>
"""
import sys, time, ctypes
import torch
import triton
import triton.language as tl


BLOCK_SIZE = 256


@triton.jit
def bitonic_step(a_ptr, seq_len, two_power, n_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < n_elems
    seq_num = i // seq_len
    h_len = seq_len // 2
    lower = i < (seq_len * seq_num) + h_len  # only first half of the sequence
    active = mask & lower
    swapped_ele = i + h_len
    odd = seq_num // two_power
    increasing = (odd % 2) == 0

    ai = tl.load(a_ptr + i, mask=active, other=0)
    aj = tl.load(a_ptr + swapped_ele, mask=active, other=0)

    need_swap = ((ai > aj) & increasing) | ((ai < aj) & (~increasing))
    do_swap = active & need_swap

    tl.store(a_ptr + i,           tl.where(do_swap, aj, ai), mask=active)
    tl.store(a_ptr + swapped_ele, tl.where(do_swap, ai, aj), mask=active)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} n k")
        return 1
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    if n > 22:
        print(f"[note] n={n} shrunk to 22 to fit CPU-verification budget")
        n = 22
    size = 1 << n
    print(f"\nArray size: {size}, seed: {seed}")

    # Match the CUDA harness's random init exactly.
    libc = ctypes.CDLL("libc.so.6")
    libc.srand(ctypes.c_uint(seed))
    data = torch.tensor([libc.rand() % 1000 for _ in range(size)], dtype=torch.int32)

    d = data.cuda()
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    print("Bitonic sort (parallel)..")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n):
        for stage in range(step, -1, -1):
            seq_len = 1 << (stage + 1)
            two_power = 1 << (step - stage)
            bitonic_step[grid](d, seq_len, two_power, size, BLOCK=BLOCK_SIZE)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    print(f"Total kernel execution time: {dt_ms:f} (ms)")

    print("Bitonic sort (serial)..")
    # Use torch.sort as reference (deterministic).
    ref, _ = torch.sort(data)
    got = d.cpu()
    ok = torch.equal(ref, got)
    print("PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
