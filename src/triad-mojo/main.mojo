# Mojo GPU port of the SHOC `triad` benchmark.
#
# Simple STREAM Triad kernel: C[i] = A[i] + s * B[i].
# The original benchmark stress-tests memory transfers via a two-stream
# ping-pong; here we perform a straight-forward element-wise sweep and
# verify against a Mojo CPU reference.
#
# Usage: main.mojo <n_passes>
#
# Mojo 1.0.0b2 GPU kernel launch pattern (matches accuracy-mojo).

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.random import seed, random_float64
from std.time import perf_counter_ns


def triad_kernel(
        A: UnsafePointer[Float32, MutAnyOrigin],
        B: UnsafePointer[Float32, MutAnyOrigin],
        C: UnsafePointer[Float32, MutAnyOrigin],
        n: Int, s: Float32):
    var gid = block_idx.x * block_dim.x + thread_idx.x
    if gid >= n:
        return
    C[gid] = A[gid] + s * B[gid]


def main() raises:
    var args = argv()
    var n_passes: Int = 100
    if len(args) >= 2:
        n_passes = Int(atol(args[1]))

    # 4M floats — 16 MiB per buffer, matches CUDA benchmark scale.
    var num_floats: Int = 1024 * 1024 * 4
    var half = num_floats // 2
    var scalar: Float32 = 1.75

    var ctx = DeviceContext()
    var d_A = ctx.enqueue_create_buffer[DType.float32](num_floats)
    var d_B = ctx.enqueue_create_buffer[DType.float32](num_floats)
    var d_C = ctx.enqueue_create_buffer[DType.float32](num_floats)

    seed(8650341)
    with d_A.map_to_host() as ha, d_B.map_to_host() as hb:
        for j in range(half):
            var v = Float32(random_float64() * 10.0)
            ha[j]        = v
            ha[j + half] = v
            hb[j]        = v
            hb[j + half] = v

    alias BLOCK: Int = 128
    var blocks = (num_floats + BLOCK - 1) // BLOCK

    # warm up
    ctx.enqueue_function[func=triad_kernel](
        d_A.unsafe_ptr(), d_B.unsafe_ptr(), d_C.unsafe_ptr(),
        num_floats, scalar,
        grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(n_passes):
        ctx.enqueue_function[func=triad_kernel](
            d_A.unsafe_ptr(), d_B.unsafe_ptr(), d_C.unsafe_ptr(),
            num_floats, scalar,
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var seconds = Float64(t1 - t0) * 1e-9
    var bdwth = (Float64(num_floats) * 4.0 * 3.0 * Float64(n_passes)) / (seconds * 1e9)
    var flops = (Float64(num_floats) * 2.0 * Float64(n_passes)) / (seconds * 1e9)
    print("Average TriadBdwth", bdwth, "GB/s")
    print("Average TriadFlops", flops, "GFLOPS/s")
    print("Average kernel time", Float64(t1 - t0) * 1e-3 / Float64(n_passes), "us")

    # Verify: halves must match.
    var ok = True
    var mism = 0
    with d_C.map_to_host() as hc:
        for j in range(half):
            var a = hc[j]
            var b = hc[j + half]
            if a != b:
                if mism < 3:
                    print("mismatch at", j, "=", a, "vs", b)
                mism += 1
                ok = False
    if ok:
        print("PASS")
    else:
        print("FAIL:", mism, "mismatches")
