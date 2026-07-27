# Mojo GPU port of the `softmax` HeCBench benchmark.
#
# One thread per row: compute max, exp-and-sum, normalize. Same as the
# CUDA source's simple `softMax` kernel (not the warp-shuffled softMax2).
# Verified against a host-side reference to within 1e-5.
#
# Usage: main.mojo <numSlice> <sliceSize> <select> <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.math import exp as _exp
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


def softmax_kernel(
        src:  UnsafePointer[Float32, MutAnyOrigin],
        dst:  UnsafePointer[Float32, MutAnyOrigin],
        num_slice: Int, slice_size: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= num_slice:
        return
    var base = i * slice_size
    var max_v: Float32 = src[base]
    for j in range(slice_size):
        var v = src[base + j]
        if v > max_v:
            max_v = v
    var sum_v: Float32 = 0.0
    for j in range(slice_size):
        sum_v = sum_v + _exp(src[base + j] - max_v)
    for j in range(slice_size):
        dst[base + j] = _exp(src[base + j] - max_v) / sum_v


def softmax_cpu(
        src:  UnsafePointer[Float32, MutAnyOrigin],
        dst:  UnsafePointer[Float32, MutAnyOrigin],
        num_slice: Int, slice_size: Int):
    for i in range(num_slice):
        var base = i * slice_size
        var max_v: Float32 = src[base]
        for j in range(slice_size):
            var v = src[base + j]
            if v > max_v:
                max_v = v
        var sum_v: Float32 = 0.0
        for j in range(slice_size):
            sum_v = sum_v + _exp(src[base + j] - max_v)
        for j in range(slice_size):
            dst[base + j] = _exp(src[base + j] - max_v) / sum_v


def main() raises:
    var args = argv()
    if len(args) != 5:
        print("Usage:", args[0], "<numSlice> <sliceSize> <select> <repeat>")
        exit(1)
    var num_slice   = Int(atol(args[1]))
    var slice_size  = Int(atol(args[2]))
    var _select     = Int(atol(args[3]))
    var repeat      = Int(atol(args[4]))
    var total = num_slice * slice_size

    var ctx = DeviceContext()
    var d_src = ctx.enqueue_create_buffer[DType.float32](total)
    var d_dst = ctx.enqueue_create_buffer[DType.float32](total)
    var ref_dst = ctx.enqueue_create_buffer[DType.float32](total)

    # Deterministic init via LCG (same pattern as adam-mojo)
    var s: UInt64 = 20260722
    with d_src.map_to_host() as sh:
        for i in range(total):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            sh[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)

    comptime BLOCK: Int = 256
    var blocks = (num_slice + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=softmax_kernel](
            d_src.unsafe_ptr(), d_dst.unsafe_ptr(), num_slice, slice_size,
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var elapsed_ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(repeat)
    print("Average kernel execution time", elapsed_ms, "(ms)")

    # Reference on first 1024 rows (full compute is 100000 rows × 784 cols)
    var subset = 1024 if num_slice > 1024 else num_slice
    with d_src.map_to_host() as sh, ref_dst.map_to_host() as rh:
        softmax_cpu(sh.unsafe_ptr(), rh.unsafe_ptr(), subset, slice_size)

    var max_err: Float32 = 0.0
    with d_dst.map_to_host() as gh, ref_dst.map_to_host() as rh:
        for i in range(subset * slice_size):
            var d: Float32 = gh[i] - rh[i]
            if d < 0:
                d = -d
            if d > max_err:
                max_err = d
    print("Max element error over first", subset, "rows:", max_err)
    if max_err <= Float32(1e-5):
        print("PASS")
    else:
        print("FAIL")
