#!/usr/bin/env python3
"""Triton port of the `adjacent` HeCBench benchmark.

Block-wide adjacent difference: for each contiguous block of BLOCK_SIZE*4
integers, compute out[i] = in[i] - in[i-1] (subtract-left) or out[i] = in[i] -
in[i+1] (subtract-right); boundary elements stay unchanged.

Matches adjacent-cuda semantics; verifies against a torch CPU reference and
prints PASS/FAIL for each BLOCK_SIZE.

Usage: main.py <num_elements> <repeat>
"""
import sys, time
import torch
import triton
import triton.language as tl


@triton.jit
def adj_diff_left_kernel(in_ptr, out_ptr, N,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs, mask=offs < N, other=0)
    # Shift right by 1 within this block: x_prev is in[i-1], with i==0 mapping
    # to x[i] itself (so difference is zero → we return in[i] instead).
    idx_in_block = tl.arange(0, BLOCK)
    prev_offs = pid * BLOCK + tl.maximum(idx_in_block - 1, 0)
    x_prev = tl.load(in_ptr + prev_offs, mask=offs < N, other=0)
    out = tl.where(idx_in_block == 0, x, x - x_prev)
    tl.store(out_ptr + offs, out, mask=offs < N)


@triton.jit
def adj_diff_right_kernel(in_ptr, out_ptr, N,
                          BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs, mask=offs < N, other=0)
    idx_in_block = tl.arange(0, BLOCK)
    last = BLOCK - 1
    next_offs = pid * BLOCK + tl.minimum(idx_in_block + 1, last)
    x_next = tl.load(in_ptr + next_offs, mask=offs < N, other=0)
    out = tl.where(idx_in_block == last, x, x - x_next)
    tl.store(out_ptr + offs, out, mask=offs < N)


def reference(inp: torch.Tensor, block: int, subtract_left: bool) -> torch.Tensor:
    inp2d = inp.view(-1, block)
    out = inp2d.clone()
    if subtract_left:
        out[:, 1:] = inp2d[:, 1:] - inp2d[:, :-1]
    else:
        out[:, :-1] = inp2d[:, :-1] - inp2d[:, 1:]
    return out.view(-1)


def test(num_items: int, repeat: int, block_threads: int) -> None:
    items_per_block = block_threads * 4  # matches cuda: ITEMS_PER_THREAD=4
    num_items = ((num_items + items_per_block - 1) // items_per_block) * items_per_block
    grid = num_items // items_per_block

    # Same init pattern as adjacent-cuda: h_in[i] = i % 17
    h_in = (torch.arange(num_items, device="cuda", dtype=torch.int32) % 17)
    h_out = torch.empty_like(h_in)

    # verify subtract-left
    for _ in range(repeat):
        adj_diff_left_kernel[(grid,)](h_in, h_out, num_items, BLOCK=items_per_block)
    ref_left = reference(h_in, items_per_block, subtract_left=True)
    ok_left = torch.equal(h_out, ref_left)
    print("PASS" if ok_left else "FAIL")

    # verify subtract-right
    for _ in range(repeat):
        adj_diff_right_kernel[(grid,)](h_in, h_out, num_items, BLOCK=items_per_block)
    ref_right = reference(h_in, items_per_block, subtract_left=False)
    ok_right = torch.equal(h_out, ref_right)
    print("PASS" if ok_right else "FAIL")

    # timed run: 2 kernels back to back, `repeat` times
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        adj_diff_left_kernel[(grid,)](h_in, h_out, num_items, BLOCK=items_per_block)
        adj_diff_right_kernel[(grid,)](h_out, h_out, num_items, BLOCK=items_per_block)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / repeat
    print(f"Average execution time of the kernels (thread block size = {block_threads:4d}): {elapsed_us:f} (us)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <number of elements> <repeat>")
        return 1
    n = int(sys.argv[1]); repeat = int(sys.argv[2])
    for bs in (64, 128, 256, 512, 1024):
        test(n, repeat, bs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
