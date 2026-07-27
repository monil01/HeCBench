#!/usr/bin/env python3
"""Triton port of the `triad` HeCBench benchmark.

Stream Triad: C = A + s*B, run across a sweep of block sizes.  Matches
triad-cuda/triad.cu output structure (nine blocks -> nine PASS lines).

Simplification: the CUDA version double-buffers H<->D copies to overlap
compute and PCIe transfers; that is not the point of a Triton port, so we
just run the kernel and time it.  The correctness check (two halves match)
carries over unchanged.
"""
import sys, time, argparse
import torch
import triton
import triton.language as tl


@triton.jit
def triad_kernel(A_ptr, B_ptr, C_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    a = tl.load(A_ptr + offs, mask=m)
    b = tl.load(B_ptr + offs, mask=m)
    tl.store(C_ptr + offs, a + scalar * b, mask=m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args()

    n_passes = args.passes
    verbose = args.verbose

    block_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    mem_size = 16384
    num_max_floats = 1024 * mem_size // 4
    half_num_floats = num_max_floats // 2

    torch.manual_seed(8650341)
    # Same-halves pattern: fill first half random, second half = first
    half = torch.rand(half_num_floats, device="cuda", dtype=torch.float32) * 10.0
    h_mem = torch.cat([half, half])

    scalar = 1.75

    A = torch.empty(num_max_floats, device="cuda", dtype=torch.float32)
    B = torch.empty(num_max_floats, device="cuda", dtype=torch.float32)
    C = torch.empty(num_max_floats, device="cuda", dtype=torch.float32)

    BLOCK = 128
    for i, bs in enumerate(block_sizes):
        elems_in_block = bs * 1024 // 4
        if verbose:
            print(f">> Executing Triad with vectors of length "
                  f"{num_max_floats} and block size of {elems_in_block} elements.")
            print(f"Block: {bs}KB")

        A.copy_(h_mem)
        B.copy_(h_mem)
        C.zero_()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_passes):
            n_blocks = (num_max_floats + BLOCK - 1) // BLOCK
            triad_kernel[(n_blocks,)](A, B, C, scalar, num_max_floats, BLOCK=BLOCK)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Bring the output back into h_mem (mimic device->host of results)
        h_mem = C.clone()

        gflops = (num_max_floats * 2.0 * n_passes) / (elapsed * 1e9)
        bw = (num_max_floats * 4 * 3.0 * n_passes) / (elapsed * 1e9)
        if verbose:
            print(f"Average TriadFlops {gflops} GFLOPS/s")
            print(f"Average TriadBdwth {bw} GB/s")

        # Correctness: halves must match
        left = h_mem[:half_num_floats]
        right = h_mem[half_num_floats:]
        ok = torch.equal(left, right)
        print("PASS" if ok else "FAIL")

        h_mem.zero_()

    return 0


if __name__ == "__main__":
    sys.exit(main())
