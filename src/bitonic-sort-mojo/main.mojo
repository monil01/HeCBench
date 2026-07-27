# Mojo GPU port of the `bitonic-sort` HeCBench benchmark.
#
# Array size = 2^n. log^2(n) kernel launches, one per (step, stage).
# Each thread swaps within a bitonic subsequence. Verified against a
# Mojo host-side sort. Matches the CUDA source's simple kernel — no
# atomics, no transcendentals — so a clean fit for Mojo 1.0.0b2.
#
# Usage: main.mojo <log2_size> <seed>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


def bitonic_kernel(
        a:        UnsafePointer[Int32, MutAnyOrigin],
        seq_len:   Int32,
        two_power: Int32):
    var i: Int32 = Int32(block_dim.x * block_idx.x + thread_idx.x)
    var seq_num: Int32 = i // seq_len
    var swapped_ele: Int32 = -1
    var h_len: Int32 = seq_len // 2
    if i < seq_len * seq_num + h_len:
        swapped_ele = i + h_len
    var odd: Int32 = seq_num // two_power
    var increasing: Bool = (odd % Int32(2)) == Int32(0)
    if swapped_ele != Int32(-1):
        var ai: Int32 = a[Int(i)]
        var aj: Int32 = a[Int(swapped_ele)]
        var need_swap: Bool = (ai > aj and increasing) or (ai < aj and not increasing)
        if need_swap:
            a[Int(i)] = aj
            a[Int(swapped_ele)] = ai


def main() raises:
    var args = argv()
    if len(args) != 3:
        print("Usage:", args[0], "<log2_size> <seed>")
        exit(1)
    var n_bits = Int(atol(args[1]))
    var seed_v = Int(atol(args[2]))
    var size = Int(1 << n_bits)
    print("Array size:", size, ", seed:", seed_v)

    var ctx = DeviceContext()
    var d_a = ctx.enqueue_create_buffer[DType.int32](size)

    # Deterministic host init via LCG
    var s: UInt64 = UInt64(seed_v)
    with d_a.map_to_host() as h:
        for i in range(size):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            h[i] = Int32(Int(s >> 33) & 0x7FFFFFFF)

    # Golden expected: same values sorted ascending
    var expected = ctx.enqueue_create_buffer[DType.int32](size)
    with d_a.map_to_host() as h, expected.map_to_host() as e:
        for i in range(size):
            e[i] = h[i]
        # Insertion sort — O(n^2) but fine for verification of small n
        for i in range(1, size):
            var k = i
            while k > 0 and e[k-1] > e[k]:
                var t = e[k-1]
                e[k-1] = e[k]
                e[k] = t
                k = k - 1

    comptime BLOCK: Int = 256
    var grid = (size + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for step in range(n_bits):
        var stage = step
        while stage >= 0:
            var seq_len: Int32 = Int32(1 << (stage + 1))
            var two_power: Int32 = Int32(1 << (step - stage))
            ctx.enqueue_function[func=bitonic_kernel](
                d_a.unsafe_ptr(), seq_len, two_power,
                grid_dim=grid, block_dim=BLOCK)
            stage = stage - 1
    ctx.synchronize()
    var elapsed_ms = Float64(perf_counter_ns() - t0) / 1e6
    print("Total kernel execution time:", elapsed_ms, "(ms)")

    var ok = True
    with d_a.map_to_host() as h, expected.map_to_host() as e:
        for i in range(size):
            if h[i] != e[i]:
                if ok:
                    print("Mismatch at", i, ": gpu=", h[i], "cpu=", e[i])
                ok = False
    if ok:
        print("PASS")
    else:
        print("FAIL")
