# Mojo GPU port of `all-pairs-distance` HeCBench benchmark.
#
# Hamming-distance-like all-pairs kernel over a 512-instance x 100-attribute
# character matrix. The upstream CUDA kernel uses atomicAdd across a 2D
# team/thread space; Mojo 1.0.0b2 doesn't expose an atomicAdd usable from
# a device kernel this way, so we use a variant with one thread per
# (gx, gy) pair that walks all attributes serially — same result, no
# atomics needed. Compared against a host-side reference for correctness.
#
# Usage: main.mojo <iterations>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns

alias INSTANCES: Int = 512
alias ATTRIBUTES: Int = 100


def apd_kernel(
        data:     UnsafePointer[UInt8, MutAnyOrigin],
        distance: UnsafePointer[Int32, MutAnyOrigin]):
    var gx = Int(block_idx.x)
    var gy = Int(block_idx.y)
    if gx >= INSTANCES or gy >= INSTANCES:
        return
    var cnt: Int32 = 0
    for i in range(ATTRIBUTES):
        if data[i + ATTRIBUTES * gx] != data[i + ATTRIBUTES * gy]:
            cnt = cnt + Int32(1)
    distance[INSTANCES * gx + gy] = cnt


def main() raises:
    var args = argv()
    var iters = Int(atol(args[1])) if len(args) >= 2 else 1

    var ctx = DeviceContext()
    var d_data = ctx.enqueue_create_buffer[DType.uint8](INSTANCES * ATTRIBUTES)
    var d_dist = ctx.enqueue_create_buffer[DType.int32](INSTANCES * INSTANCES)
    var ref_dist = ctx.enqueue_create_buffer[DType.int32](INSTANCES * INSTANCES)

    var s: UInt64 = 20260721
    with d_data.map_to_host() as h:
        for i in range(INSTANCES * ATTRIBUTES):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            h[i] = UInt8(Int(s >> 33) % 4)  # 4 symbols

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[func=apd_kernel](
            d_data.unsafe_ptr(), d_dist.unsafe_ptr(),
            grid_dim=(INSTANCES, INSTANCES), block_dim=1)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(iters)
    print("Average kernel execution time:", us, "(us)")

    # Host reference
    with d_data.map_to_host() as h, ref_dist.map_to_host() as rd:
        for gx in range(INSTANCES):
            for gy in range(INSTANCES):
                var cnt: Int32 = 0
                for i in range(ATTRIBUTES):
                    if h[i + ATTRIBUTES * gx] != h[i + ATTRIBUTES * gy]:
                        cnt = cnt + Int32(1)
                rd[INSTANCES * gx + gy] = cnt

    var ok = True
    with d_dist.map_to_host() as gh, ref_dist.map_to_host() as rh:
        for i in range(INSTANCES * INSTANCES):
            if gh[i] != rh[i]:
                if ok:
                    print("Mismatch at", i, "gpu=", gh[i], "cpu=", rh[i])
                ok = False
    if ok:
        print("PASS")
    else:
        print("FAIL")
