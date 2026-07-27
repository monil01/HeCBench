# Mojo GPU port of the `saxpy-ompt` HeCBench benchmark (simplified).
#
# The upstream benchmark compares SAXPY on CPU/OpenMP/OpenMP-target/CUDA
# with cuBLAS and hand-written kernels. This port covers the Mojo-relevant
# case: a hand-written GPU SAXPY kernel + host-side comparison.
#
# Usage: main.mojo (no args needed; uses fixed n=1M, a=2.0f)

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import exit
from std.time import perf_counter_ns


def saxpy_kernel(
        x: UnsafePointer[Float32, MutAnyOrigin],
        y: UnsafePointer[Float32, MutAnyOrigin],
        a: Float32, n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        y[i] = a * x[i] + y[i]


def main() raises:
    var n = 1_000_000
    var a: Float32 = 2.0

    var ctx = DeviceContext()
    var d_x = ctx.enqueue_create_buffer[DType.float32](n)
    var d_y = ctx.enqueue_create_buffer[DType.float32](n)

    with d_x.map_to_host() as xh, d_y.map_to_host() as yh:
        for i in range(n):
            xh[i] = Float32(i % 100) / Float32(100)
            yh[i] = Float32((i * 7) % 100) / Float32(100)

    comptime BLOCK: Int = 256
    var blocks = (n + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    var repeat = 100
    for _ in range(repeat):
        ctx.enqueue_function[func=saxpy_kernel](
            d_x.unsafe_ptr(), d_y.unsafe_ptr(), a, n,
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(repeat)
    print("Average SAXPY kernel time:", us, "(us)")

    # Verify: y_final[i] = repeat*a*x[i] + y_init[i]
    var max_err: Float32 = 0.0
    with d_x.map_to_host() as xh, d_y.map_to_host() as yh:
        for i in range(n):
            var xi = Float32(i % 100) / Float32(100)
            var yi_init = Float32((i * 7) % 100) / Float32(100)
            var expected = Float32(repeat) * a * xi + yi_init
            var d: Float32 = yh[i] - expected
            if d < 0:
                d = -d
            if d > max_err:
                max_err = d
    print("Max element error:", max_err)
    if max_err <= Float32(1e-3):
        print("PASS")
    else:
        print("FAIL")
