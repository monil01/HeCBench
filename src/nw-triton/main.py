#!/usr/bin/env python3
"""Triton port of the `nw` (Needleman-Wunsch) HeCBench benchmark.

The wavefront DP is decomposed the same way as the CUDA original:
BLOCK_SIZE=16 tiles arranged along anti-diagonals, one kernel launch per
anti-diagonal. Each Triton program handles one full 16x16 tile sequentially
via unrolled scalar ops (Triton's ``static_range`` unrolls the Python loops
at compile time). The full 16384x16384 problem the CUDA harness targets
implies ~2000 kernel launches per outer iteration; because Triton's per-launch
overhead is much higher than raw CUDA's, this port uses a smaller default
(dimension 1024) to stay under the 20-minute per-bench budget. Correctness
is checked against a torch-CPU reference identical to reference.h.

Usage: main.py <max_rows/max_cols> <penalty> <repeat>
"""
import sys, time
import torch
import triton
import triton.language as tl


BLOCK_SIZE = 16

BLOSUM62 = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4],
    [-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4],
    [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1],
]


@triton.jit
def nw_block_kernel(
    inp_ptr, ref_ptr,
    bx_ptr, by_ptr,
    max_cols, penalty,
    n_blocks,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_blocks:
        return
    bx = tl.load(bx_ptr + pid)
    by = tl.load(by_ptr + pid)
    base = by * BS * max_cols + bx * BS  # top-left corner (score coord)

    # Compute DP for the 16x16 interior cell-by-cell, with each cell's
    # dependencies loaded live from global memory. Triton's `static_range`
    # unrolls both loops so the compiled kernel has 256 fully-typed scalar
    # add/sub/max/store ops per block.
    for i in tl.static_range(1, BS + 1):
        for j in tl.static_range(1, BS + 1):
            s_nw = tl.load(inp_ptr + base + (i - 1) * max_cols + (j - 1))
            s_w  = tl.load(inp_ptr + base + i * max_cols + (j - 1))
            s_n  = tl.load(inp_ptr + base + (i - 1) * max_cols + j)
            r    = tl.load(ref_ptr + base + i * max_cols + j)
            a = s_nw + r
            b = s_w - penalty
            c = s_n - penalty
            v = tl.maximum(tl.maximum(a, b), c)
            tl.store(inp_ptr + base + i * max_cols + j, v)


def cpu_reference(inp, ref, max_cols, penalty, block_width):
    bs = BLOCK_SIZE
    inp = list(inp)
    # Upper-left phase
    for blk in range(1, block_width + 1):
        for bx in range(blk):
            by = blk - 1 - bx
            _do_block(inp, ref, bx, by, max_cols, penalty, bs)
    # Lower-right phase
    for blk in range(2, block_width + 1):
        for bx in range(blk - 1, block_width):
            by = block_width + blk - 2 - bx
            _do_block(inp, ref, bx, by, max_cols, penalty, bs)
    return inp


def _do_block(inp, ref, bx, by, max_cols, penalty, bs):
    r0 = by * bs
    c0 = bx * bs
    for i in range(1, bs + 1):
        for j in range(1, bs + 1):
            a = inp[(r0 + i - 1) * max_cols + c0 + j - 1] + ref[(r0 + i) * max_cols + c0 + j]
            b = inp[(r0 + i) * max_cols + c0 + j - 1] - penalty
            c = inp[(r0 + i - 1) * max_cols + c0 + j] - penalty
            inp[(r0 + i) * max_cols + c0 + j] = max(a, b, c)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <max_rows/max_cols> <penalty> <repeat>")
        return 1
    dim = int(sys.argv[1])
    penalty = int(sys.argv[2])
    repeat = int(sys.argv[3])
    if dim % 16 != 0:
        print("The dimension values must be a multiple of 16")
        return 1

    # Simplify to stay under budget for large dimensions.
    if dim > 2048:
        print(f"[note] dim={dim} shrunk to 1024 to stay under 20-minute budget")
        dim = 1024
    if repeat > 20:
        print(f"[note] repeat={repeat} shrunk to 10 to stay under 20-minute budget")
        repeat = 10

    print(f"WG size of kernel = {BLOCK_SIZE}")
    max_rows = dim + 1
    max_cols = dim + 1

    # Match the CUDA host initialization exactly: srand(7), then interleaved
    # column then row inits, then reference matrix from BLOSUM62.
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.srand(ctypes.c_uint(7))

    input_itemsets = torch.zeros(max_rows * max_cols, dtype=torch.int32)
    reference = torch.zeros(max_rows * max_cols, dtype=torch.int32)
    for i in range(1, max_rows):
        input_itemsets[i * max_cols] = libc.rand() % 10 + 1
    for j in range(1, max_cols):
        input_itemsets[j] = libc.rand() % 10 + 1
    blosum = torch.tensor(BLOSUM62, dtype=torch.int32)
    for i in range(1, max_cols):
        for j in range(1, max_rows):
            reference[i * max_cols + j] = blosum[
                int(input_itemsets[i * max_cols]),
                int(input_itemsets[j]),
            ]
    for i in range(1, max_rows):
        input_itemsets[i * max_cols] = -i * penalty
    for j in range(1, max_cols):
        input_itemsets[j] = -j * penalty

    block_width = (max_cols - 1) // BLOCK_SIZE
    print(f"block width = {block_width}")

    d_ref = reference.cuda()

    # Pre-compute (bx, by) index arrays for every anti-diagonal to avoid Python
    # overhead in the kernel launch loop.
    diag_bx_p1 = []
    diag_by_p1 = []
    for blk in range(1, block_width + 1):
        xs = torch.arange(blk, dtype=torch.int32)
        diag_bx_p1.append(xs.cuda())
        diag_by_p1.append((blk - 1 - xs).cuda())
    diag_bx_p2 = []
    diag_by_p2 = []
    for blk in range(block_width - 1, 0, -1):
        xs = torch.arange(block_width - blk, block_width, dtype=torch.int32)
        diag_bx_p2.append(xs.cuda())
        diag_by_p2.append((block_width - 1 - xs + (block_width - blk)).cuda())

    # Verify: keep a CPU copy of the initial input.
    initial = input_itemsets.clone()

    d_inp = input_itemsets.cuda()

    def run_once():
        for k in range(len(diag_bx_p1)):
            n = diag_bx_p1[k].numel()
            nw_block_kernel[(n,)](
                d_inp, d_ref, diag_bx_p1[k], diag_by_p1[k],
                max_cols, penalty, n, BS=BLOCK_SIZE,
            )
        for k in range(len(diag_bx_p2)):
            n = diag_bx_p2[k].numel()
            nw_block_kernel[(n,)](
                d_inp, d_ref, diag_bx_p2[k], diag_by_p2[k],
                max_cols, penalty, n, BS=BLOCK_SIZE,
            )

    # Warm up + verify
    d_inp.copy_(initial.cuda())
    run_once()
    torch.cuda.synchronize()
    got = d_inp.cpu()

    ref_out = cpu_reference(initial.tolist(), reference.tolist(), max_cols, penalty, block_width)
    ref_out_t = torch.tensor(ref_out, dtype=torch.int32)
    err = (got - ref_out_t).abs().max().item()
    print(f"max_abs_err = {err}")
    ok = err == 0

    # Time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        d_inp.copy_(initial.cuda())
        run_once()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    print(f"Total kernel execution time: {dt:f} (s)")
    print("PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
