# Mojo GPU port of the `sobol` HeCBench benchmark (simplified variant).
# Matches sobol-triton/sobol-rust simplifications: synthesised direction
# vectors (seed pattern, top bit of k-th vector = 2^(31-k)), Gray-code
# XOR loop. CPU reference reproduces the same values for byte-exact match.
#
# Usage: main.mojo <n_vectors> <n_dimensions> <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


def sobol_kernel(
        dirs:  UnsafePointer[Int32,   MutAnyOrigin],
        result: UnsafePointer[Float32, MutAnyOrigin],
        n_vec: Int, n_dim: Int):
    var i   = block_idx.x * block_dim.x + thread_idx.x
    var dim = block_idx.y
    if i >= n_vec:
        return
    var g: UInt32 = UInt32(i) ^ (UInt32(i) >> 1)
    var x: UInt32 = 0
    for k in range(32):
        var bit: UInt32 = (g >> UInt32(k)) & UInt32(1)
        var mask: UInt32 = UInt32(0) - bit
        var v: UInt32 = UInt32(dirs[dim * 32 + k])
        x = x ^ (mask & v)
    result[dim * n_vec + i] = Float32(x) * Float32(2.3283064e-10)


def main() raises:
    var args = argv()
    if len(args) != 4:
        print("Usage:", args[0], "<n_vectors> <n_dimensions> <repeat>")
        exit(1)
    var n_vec  = Int(atol(args[1]))
    var n_dim  = Int(atol(args[2]))
    var repeat = Int(atol(args[3]))

    var ctx = DeviceContext()
    print("Allocating GPU memory...")

    var d_dirs = ctx.enqueue_create_buffer[DType.int32](n_dim * 32)
    var d_out  = ctx.enqueue_create_buffer[DType.float32](n_dim * n_vec)

    print("Initializing direction numbers...")
    var seed_state: UInt64 = 0xC0FFEE
    with d_dirs.map_to_host() as h:
        for dim in range(n_dim):
            for k in range(32):
                seed_state = seed_state ^ (seed_state << 13)
                seed_state = seed_state ^ (seed_state >> 7)
                seed_state = seed_state ^ (seed_state << 17)
                var low30: UInt32 = UInt32(seed_state) & UInt32((1 << 30) - 1)
                var msb:   UInt32 = UInt32(1) << UInt32(31 - k)
                h[dim * 32 + k] = Int32(low30 | msb)

    comptime BLOCK: Int = 128
    var grid_x = (n_vec + BLOCK - 1) // BLOCK

    print("Executing QRNG on GPU...")
    # warmup
    ctx.enqueue_function[func=sobol_kernel](
        d_dirs.unsafe_ptr(), d_out.unsafe_ptr(), n_vec, n_dim,
        grid_dim=(grid_x, n_dim), block_dim=BLOCK)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=sobol_kernel](
            d_dirs.unsafe_ptr(), d_out.unsafe_ptr(), n_vec, n_dim,
            grid_dim=(grid_x, n_dim), block_dim=BLOCK)
    ctx.synchronize()
    var kt_ns = perf_counter_ns() - t0
    print("Average kernel execution time:", Float64(kt_ns) / 1e9 / Float64(repeat), "(s)")

    # CPU reference + comparison inline within map_to_host blocks
    print("Checking results (subset)...")
    var n_cpu = 16 if n_dim > 16 else n_dim
    var l1_diff: Float64 = 0.0
    var l1_ref:  Float64 = 0.0
    with d_dirs.map_to_host() as dh, d_out.map_to_host() as gh:
        for dim in range(n_cpu):
            for i in range(n_vec):
                var g: UInt32 = UInt32(i) ^ (UInt32(i) >> 1)
                var x: UInt32 = 0
                for k in range(32):
                    var bit: UInt32 = (g >> k) & 1
                    var mask: UInt32 = UInt32(0) - bit
                    var v: UInt32 = UInt32(dh[dim * 32 + k])
                    x = x ^ (mask & v)
                var ref_val: Float32 = Float32(x) * Float32(2.3283064e-10)
                var gpu_val: Float32 = gh[dim * n_vec + i]
                var d: Float32 = gpu_val - ref_val
                if d < 0:
                    d = -d
                l1_diff = l1_diff + Float64(d)
                var ac: Float32 = ref_val
                if ac < 0:
                    ac = -ac
                l1_ref = l1_ref + Float64(ac)

    var l1err: Float64 = l1_diff / l1_ref if l1_ref > 0 else l1_diff
    print("L1-Error:", l1err)
    if l1err < 1e-6:
        print("PASS")
    else:
        print("FAIL")
