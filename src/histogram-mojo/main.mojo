# Mojo GPU port of the `histogram` HeCBench benchmark (simplified).
#
# 256-bin histogram of uint8 values. Mojo 1.0.0b2 does not expose an
# atomicAdd usable from a device kernel, so this port uses per-thread
# partial histograms: each of NTILE threads walks a slice of the input
# and writes 256 counts to its own tile in an output buffer of size
# NTILE * 256. The host then sums the tiles into the final histogram.
# Verified against a host-side reference on the same input.
#
# Usage: main.mojo <iters>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


comptime NUM_BINS: Int = 256
comptime NTILE:    Int = 1024  # partial histograms


def hist_kernel(
        data:    UnsafePointer[UInt8, MutAnyOrigin],
        partials: UnsafePointer[Int32, MutAnyOrigin],
        n: Int32):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= NTILE: return
    # Each thread owns a strided slice of the input
    var base = tid * NUM_BINS
    for b in range(NUM_BINS):
        partials[base + b] = Int32(0)
    var i = tid
    while i < Int(n):
        var v = Int(data[i])
        partials[base + v] = partials[base + v] + Int32(1)
        i = i + NTILE


def main() raises:
    var args = argv()
    var iters = Int(atol(args[1])) if len(args) > 1 else 50
    var width = 1920
    var height = 1080
    var num_pixels = width * height

    print("Random image: width(", width, ") height(", height, ")")

    var ctx = DeviceContext()
    var d_data = ctx.enqueue_create_buffer[DType.uint8](num_pixels)
    var d_part = ctx.enqueue_create_buffer[DType.int32](NTILE * NUM_BINS)

    var s: UInt64 = UInt64(0xABCDEF)
    with d_data.map_to_host() as h:
        for i in range(num_pixels):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            h[i] = UInt8(Int(s >> 32) & 0xff)

    # CPU reference
    var ref_bins = List[Int64](length=NUM_BINS, fill=Int64(0))
    with d_data.map_to_host() as h:
        for i in range(num_pixels):
            var v = Int(h[i])
            ref_bins[v] = ref_bins[v] + Int64(1)

    comptime BLOCK: Int = 128
    var blocks = (NTILE + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[func=hist_kernel](
            d_data.unsafe_ptr(), d_part.unsafe_ptr(), Int32(num_pixels),
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(iters)
    print("Average kernel execution time:", ms, "(ms)")

    # Reduce partials on host
    var got = List[Int64](length=NUM_BINS, fill=Int64(0))
    with d_part.map_to_host() as ph:
        for t in range(NTILE):
            var base = t * NUM_BINS
            for b in range(NUM_BINS):
                got[b] = got[b] + Int64(ph[base + b])

    var ok = True
    for b in range(NUM_BINS):
        if got[b] != ref_bins[b]:
            if ok:
                print("Mismatch at bin", b, "gpu=", got[b], "cpu=", ref_bins[b])
            ok = False
    if ok:
        print("PASS")
    else:
        print("FAIL")
