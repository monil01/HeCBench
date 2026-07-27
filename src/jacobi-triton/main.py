#!/usr/bin/env python3
"""Triton port of the `jacobi` HeCBench benchmark.

2D Jacobi relaxation on an NxN grid with sinusoidal Dirichlet boundary
conditions.  Uses a Triton kernel to compute the new field values and per-
program error contributions; the reduction is finished on the host with a
single tensor sum.
"""
import sys, math, time
import torch
import triton
import triton.language as tl


N = 2048


@triton.jit
def jacobi_step_kernel(f_ptr, fold_ptr, err_ptr,
                       N: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_j = tl.program_id(0)  # rows tile idx
    pid_i = tl.program_id(1)  # cols tile idx
    off_j = pid_j * BLOCK_M + tl.arange(0, BLOCK_M)
    off_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)

    # base row & column masks
    inside_j = (off_j >= 1) & (off_j <= N - 2)
    inside_i = (off_i >= 1) & (off_i <= N - 2)
    inside = inside_j[:, None] & inside_i[None, :]

    j2 = off_j[:, None]
    i2 = off_i[None, :]
    m_c = (j2 >= 0) & (j2 < N) & (i2 >= 0) & (i2 < N)
    center = tl.load(fold_ptr + j2 * N + i2, mask=m_c, other=0.0)
    m_l = (j2 >= 0) & (j2 < N) & ((i2 - 1) >= 0) & ((i2 - 1) < N)
    left = tl.load(fold_ptr + j2 * N + (i2 - 1), mask=m_l, other=0.0)
    m_r = (j2 >= 0) & (j2 < N) & ((i2 + 1) >= 0) & ((i2 + 1) < N)
    right = tl.load(fold_ptr + j2 * N + (i2 + 1), mask=m_r, other=0.0)
    m_d = ((j2 - 1) >= 0) & ((j2 - 1) < N) & (i2 >= 0) & (i2 < N)
    down = tl.load(fold_ptr + (j2 - 1) * N + i2, mask=m_d, other=0.0)
    m_u = ((j2 + 1) >= 0) & ((j2 + 1) < N) & (i2 >= 0) & (i2 < N)
    up = tl.load(fold_ptr + (j2 + 1) * N + i2, mask=m_u, other=0.0)

    new_val = 0.25 * (left + right + up + down)
    df = new_val - center
    err_tile = tl.where(inside, df * df, 0.0)
    err_sum = tl.sum(err_tile)

    # write out (only write inside points; leave boundary untouched)
    out_val = tl.where(inside, new_val, center)
    tl.store(f_ptr + j2 * N + i2, out_val, mask=m_c)

    tl.atomic_add(err_ptr, err_sum)


def initialize_data() -> torch.Tensor:
    j = torch.arange(N, dtype=torch.float32).view(N, 1).expand(N, N)  # j varies along rows
    i = torch.arange(N, dtype=torch.float32).view(1, N).expand(N, N)
    f = torch.zeros(N, N, dtype=torch.float32)
    boundary_x = (i * 2 * math.pi / (N - 1)).sin()  # for j==0 / j==N-1
    boundary_y = (j * 2 * math.pi / (N - 1)).sin()  # for i==0 / i==N-1
    # layout in file is IDX(i,j) = i + j*N, so f[j,i] with row=j, col=i in row-major.
    f[0, :] = boundary_x[0, :]
    f[-1, :] = boundary_x[-1, :]
    f[:, 0] = boundary_y[:, 0]
    f[:, -1] = boundary_y[:, -1]
    return f


def main():
    t_start = time.perf_counter()

    f = initialize_data().cuda().contiguous()
    f_old = f.clone()
    err = torch.zeros(1, device="cuda", dtype=torch.float32)

    tol = 1e-5
    max_iters = 10000
    BLOCK_M, BLOCK_N = 16, 16
    grid = (N // BLOCK_M, N // BLOCK_N)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    num_iters = 0
    error = float("inf")
    while error > tol and num_iters < max_iters:
        err.zero_()
        jacobi_step_kernel[grid](f, f_old, err, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        f, f_old = f_old, f
        error = math.sqrt(float(err.item()) / (N * N))
        if num_iters % 1000 == 0:
            print(f"Error after iteration {num_iters} = {error}")
        num_iters += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    avg = elapsed / num_iters
    print(f"Average execution time per iteration: {avg} (s)")
    if error <= tol and num_iters < max_iters:
        print("PASS")
    else:
        print("FAIL")
        return 1

    total = time.perf_counter() - t_start
    print(f"Total elapsed time: {total:.4g} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
