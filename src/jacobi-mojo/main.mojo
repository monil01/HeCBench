# Mojo GPU port of the HeCBench `jacobi` benchmark.
#
# 5-point Jacobi relaxation with sinusoidal Dirichlet boundary conditions.
# The upstream CUDA kernel uses shared memory, warp shuffles and
# atomicAdd for the error reduction; none of those primitives are usable
# from Mojo 1.0.0b2 device kernels, so this port:
#   * uses a plain global-memory 5-point stencil kernel
#   * writes the per-cell squared error to a scratch buffer and sums it
#     on the host each iteration
# Grid size is reduced (N=256) so the host-side reduction is cheap.
#
# Usage: main.mojo

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.math import sqrt, sin
from std.time import perf_counter_ns


alias N: Int = 256
alias BLOCK: Int = 16
alias PI: Float32 = 3.14159265358979
alias MAX_ITERS: Int = 20000
alias TOL: Float32 = 1.0e-4


def initialize_host(f: UnsafePointer[Float32, MutAnyOrigin]):
    for j in range(N):
        for i in range(N):
            if i == 0 or i == N - 1:
                f[i + j * N] = sin(Float32(j) * 2.0 * PI / Float32(N - 1))
            elif j == 0 or j == N - 1:
                f[i + j * N] = sin(Float32(i) * 2.0 * PI / Float32(N - 1))
            else:
                f[i + j * N] = 0.0


def jacobi_kernel(
        f:     UnsafePointer[Float32, MutAnyOrigin],
        f_old: UnsafePointer[Float32, MutAnyOrigin],
        err:   UnsafePointer[Float32, MutAnyOrigin]):
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y
    if i >= N or j >= N:
        return
    var idx = i + j * N
    if i >= 1 and i <= N - 2 and j >= 1 and j <= N - 2:
        var v = Float32(0.25) * (f_old[idx + 1] + f_old[idx - 1]
                                 + f_old[idx + N] + f_old[idx - N])
        f[idx] = v
        var d = v - f_old[idx]
        err[idx] = d * d
    else:
        # Preserve the boundary value from f_old
        f[idx] = f_old[idx]
        err[idx] = 0.0


def main() raises:
    var ctx = DeviceContext()

    var d_f     = ctx.enqueue_create_buffer[DType.float32](N * N)
    var d_f_old = ctx.enqueue_create_buffer[DType.float32](N * N)
    var d_err   = ctx.enqueue_create_buffer[DType.float32](N * N)

    with d_f.map_to_host() as hf, d_f_old.map_to_host() as ho:
        initialize_host(hf.unsafe_ptr())
        initialize_host(ho.unsafe_ptr())

    var grid_x = N // BLOCK
    var grid_y = N // BLOCK
    var grid = (grid_x, grid_y)
    var block = (BLOCK, BLOCK)

    ctx.synchronize()
    var t0 = perf_counter_ns()
    var num_iters: Int = 0
    var error: Float32 = 1.0e10

    # Track buffer identity through swaps by owning two references.
    while error > TOL and num_iters < MAX_ITERS:
        ctx.enqueue_function[func=jacobi_kernel](
            d_f.unsafe_ptr(), d_f_old.unsafe_ptr(), d_err.unsafe_ptr(),
            grid_dim=grid, block_dim=block)
        ctx.synchronize()

        # Reduce error on the host.
        var s: Float32 = 0.0
        with d_err.map_to_host() as he:
            for k in range(N * N):
                s += he[k]
        error = sqrt(s / Float32(N * N))

        # swap by copying d_f -> d_f_old on the device (dev-side copy would be
        # ideal; a host-round-trip works too).
        with d_f.map_to_host() as hf, d_f_old.map_to_host() as ho:
            for k in range(N * N):
                ho[k] = hf[k]

        if num_iters % 1000 == 0:
            print("Error after iteration", num_iters, "=", error)
        num_iters += 1

    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("Average execution time per iteration:",
          Float64(t1 - t0) * 1e-9 / Float64(num_iters), "(s)")

    print("Converged in", num_iters, "iterations; final error =", error)
    if error <= TOL and num_iters < MAX_ITERS:
        print("PASS")
    else:
        print("FAIL")
