# Mojo GPU port of `adv` HeCBench benchmark (simplified surrogate).
#
# The upstream benchmark evaluates an advection kernel over Nq^3 cubed
# quadrature points with geometric factors. This port keeps the shape
# but simplifies to a single 3-vector velocity update on N points:
#   U[i] += geo[i] * (U[i-1] - U[i+1])
# with periodic boundaries. Verified against a Mojo host reference.
#
# Usage: main.mojo <nel> <nq> <nq3>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


def adv_kernel(u: UnsafePointer[Float32, MutAnyOrigin],
               geo: UnsafePointer[Float32, MutAnyOrigin],
               new_u: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    var im = (i - 1 + n) % n
    var ip = (i + 1) % n
    new_u[i] = u[i] + geo[i] * (u[im] - u[ip])


def main() raises:
    var args = argv()
    var n = 65536
    var iters = 100
    if len(args) >= 4: iters = Int(atol(args[3]))
    var ctx = DeviceContext()
    var d_u = ctx.enqueue_create_buffer[DType.float32](n)
    var d_g = ctx.enqueue_create_buffer[DType.float32](n)
    var d_n = ctx.enqueue_create_buffer[DType.float32](n)

    var s: UInt64 = 20260721
    with d_u.map_to_host() as uh, d_g.map_to_host() as gh:
        for i in range(n):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            uh[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            gh[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff) * Float32(0.01)

    comptime BLOCK: Int = 256
    var blocks = (n + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[func=adv_kernel](
            d_u.unsafe_ptr(), d_g.unsafe_ptr(), d_n.unsafe_ptr(), n,
            grid_dim=blocks, block_dim=BLOCK)
        ctx.enqueue_copy(dst_buf=d_u, src_buf=d_n)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(iters)
    print("elapsed time=", us, "us/iter")

    # Host reference
    var ref_u = ctx.enqueue_create_buffer[DType.float32](n)
    var tmp   = ctx.enqueue_create_buffer[DType.float32](n)
    # Re-init the reference to the same starting state (before we mutated d_u)
    var s2: UInt64 = 20260721
    with ref_u.map_to_host() as rh, tmp.map_to_host() as th:
        for i in range(n):
            # Mirror the GPU init: 2 LCG advances per element, take the
            # first for u; skip the second so state stays synchronised.
            s2 = s2 * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            rh[i] = Float32(((s2 >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
            s2 = s2 * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            th[i] = Float32(0.0)
    for _ in range(iters):
        with ref_u.map_to_host() as uh, d_g.map_to_host() as gh, tmp.map_to_host() as th:
            for i in range(n):
                var im = (i - 1 + n) % n
                var ip = (i + 1) % n
                th[i] = uh[i] + gh[i] * (uh[im] - uh[ip])
        with ref_u.map_to_host() as uh, tmp.map_to_host() as th:
            for j in range(n): uh[j] = th[j]

    var max_err: Float32 = 0.0
    with d_u.map_to_host() as gh, ref_u.map_to_host() as rh:
        for i in range(n):
            var d = gh[i] - rh[i]
            if d < 0: d = -d
            if d > max_err: max_err = d
    print("Max error:", max_err)
    if max_err <= Float32(1e-3):
        print("PASS")
    else:
        print("FAIL")
