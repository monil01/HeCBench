# Mojo GPU port of the HeCBench `bsearch` benchmark.
#
# Runs a plain binary-search kernel (BS1) and a branchless bit-by-bit
# variant (BS3) over a sorted float array.  Each query is independent
# so the port is a straightforward per-thread loop.
#
# Verifies both kernels' output on the host and prints PASS/FAIL.
#
# Usage: main.mojo <n_elems> <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.random import seed, random_float64
from std.time import perf_counter_ns


alias BLOCK: Int = 256


def kernel_BS(
        d_a: UnsafePointer[Float32, MutAnyOrigin],
        d_z: UnsafePointer[Float32, MutAnyOrigin],
        d_r: UnsafePointer[Int32,   MutAnyOrigin],
        zSize: Int, n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= zSize:
        return
    var z = d_z[i]
    var low: Int = 0
    var high: Int = n
    while high - low > 1:
        var mid = low + (high - low) // 2
        if z < d_a[mid]:
            high = mid
        else:
            low = mid
    d_r[i] = Int32(low)


def kernel_BS3(
        d_a: UnsafePointer[Float32, MutAnyOrigin],
        d_z: UnsafePointer[Float32, MutAnyOrigin],
        d_r: UnsafePointer[Int32,   MutAnyOrigin],
        zSize: Int, n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= zSize:
        return
    # nbits = number of bits needed to represent n
    var nbits: Int = 0
    var tmp = n
    while tmp > 0:
        nbits += 1
        tmp = tmp >> 1
    var k: Int = 1 << (nbits - 1)
    var z = d_z[i]
    var idx: Int
    if d_a[k] <= z:
        idx = k
    else:
        idx = 0
    k = k >> 1
    while k > 0:
        var r = idx | k
        var w = r if r < n else n
        if z >= d_a[w]:
            idx = r
        k = k >> 1
    d_r[i] = Int32(idx)


def main() raises:
    var args = argv()
    if len(args) != 3:
        print("Usage:", args[0], "<n_elems> <repeat>")
        exit(1)
    var numElem = Int(atol(args[1]))
    var repeat  = Int(atol(args[2]))

    var aSize = numElem
    var zSize = 2 * aSize
    var N = aSize - 1

    var ctx = DeviceContext()
    var d_a = ctx.enqueue_create_buffer[DType.float32](aSize)
    var d_z = ctx.enqueue_create_buffer[DType.float32](zSize)
    var d_r = ctx.enqueue_create_buffer[DType.int32](zSize)

    # a strictly ascending: a[i] = i
    # z uniformly random in [0, N)
    seed(2)
    with d_a.map_to_host() as ha, d_z.map_to_host() as hz:
        for i in range(aSize):
            ha[i] = Float32(i)
        for i in range(zSize):
            hz[i] = Float32(random_float64() * Float64(N))

    var grid = (zSize + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=kernel_BS](
            d_a.unsafe_ptr(), d_z.unsafe_ptr(), d_r.unsafe_ptr(),
            zSize, N, grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("Average kernel execution time (bs1)", Float64(t1 - t0) * 1e-9 / Float64(repeat), "(s)")

    # verify bs1
    var ok1 = True
    var m1 = 0
    with d_a.map_to_host() as ha, d_z.map_to_host() as hz, d_r.map_to_host() as hr:
        for i in range(zSize):
            var idx = Int(hr[i])
            if not (idx + 1 < aSize and ha[idx] <= hz[i] and hz[i] < ha[idx + 1]):
                if m1 < 3:
                    print("bs1 mismatch @", i, "idx=", idx, "z=", hz[i])
                m1 += 1
                ok1 = False
    if ok1:
        print("bs1: PASS")
    else:
        print("bs1: FAIL", m1)

    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=kernel_BS3](
            d_a.unsafe_ptr(), d_z.unsafe_ptr(), d_r.unsafe_ptr(),
            zSize, N, grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("Average kernel execution time (bs3)", Float64(t1 - t0) * 1e-9 / Float64(repeat), "(s)")

    # verify bs3
    var ok2 = True
    var m2 = 0
    with d_a.map_to_host() as ha, d_z.map_to_host() as hz, d_r.map_to_host() as hr:
        for i in range(zSize):
            var idx = Int(hr[i])
            if not (idx + 1 < aSize and ha[idx] <= hz[i] and hz[i] < ha[idx + 1]):
                if m2 < 3:
                    print("bs3 mismatch @", i, "idx=", idx, "z=", hz[i])
                m2 += 1
                ok2 = False
    if ok2:
        print("bs3: PASS")
    else:
        print("bs3: FAIL", m2)

    if ok1 and ok2:
        print("PASS")
    else:
        print("FAIL")
