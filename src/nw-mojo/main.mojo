# Mojo GPU port of `nw` (Needleman-Wunsch) HeCBench benchmark.
#
# Per-anti-diagonal wavefront: one thread per cell (i, j) with i + j == diag.
# Serial in the diag axis, parallel in i. Verified against a host reference.
#
# Usage: main.mojo <dim> <penalty> <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


def nw_diag(
        itemsets:  UnsafePointer[Int32, MutAnyOrigin],
        reference: UnsafePointer[Int32, MutAnyOrigin],
        max_cols: Int32, diag: Int32, penalty: Int32,
        i_min: Int32, i_max: Int32):
    var t = Int(block_idx.x * block_dim.x + thread_idx.x)
    var i = Int(i_min) + t
    if i > Int(i_max):
        return
    var j = Int(diag) - i
    var mc = Int(max_cols)
    var idx = i * mc + j
    var nw_v = itemsets[(i - 1) * mc + (j - 1)] + reference[idx]
    var w    = itemsets[i * mc + (j - 1)] - penalty
    var n    = itemsets[(i - 1) * mc + j] - penalty
    var r = nw_v
    if w > r: r = w
    if n > r: r = n
    itemsets[idx] = r


def nw_host(itemsets: UnsafePointer[Int32, MutAnyOrigin],
            reference: UnsafePointer[Int32, MutAnyOrigin],
            max_cols: Int, penalty: Int32):
    for i in range(1, max_cols):
        for j in range(1, max_cols):
            var idx = i * max_cols + j
            var nw_v = itemsets[(i - 1) * max_cols + (j - 1)] + reference[idx]
            var w    = itemsets[i * max_cols + (j - 1)] - penalty
            var n    = itemsets[(i - 1) * max_cols + j] - penalty
            var r = nw_v
            if w > r: r = w
            if n > r: r = n
            itemsets[idx] = r


def blosum62_at(a: Int, b: Int) -> Int32:
    # BLOSUM62 24x24 substitution matrix — copied from nw-rust/src/main.rs.
    var B: List[Int32] = [
        4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4,
       -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4,
       -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4,
       -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4,
        0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4,
       -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4,
       -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
        0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4,
       -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4,
       -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4,
       -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4,
       -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4,
       -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4,
       -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4,
       -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4,
        1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4,
        0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4,
       -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4,
       -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4,
        0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4,
       -2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4,
       -1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
        0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4,
       -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1
    ]
    return B[a * 24 + b]


def main() raises:
    var args = argv()
    var dim: Int = Int(atol(args[1])) if len(args) > 1 else 256
    var penalty: Int32 = Int32(atol(args[2])) if len(args) > 2 else Int32(10)
    var repeat: Int = Int(atol(args[3])) if len(args) > 3 else 5

    var max_cols = dim + 1
    var max_rows = dim + 1
    var total = max_cols * max_rows

    var ctx = DeviceContext()
    var d_items = ctx.enqueue_create_buffer[DType.int32](total)
    var d_ref   = ctx.enqueue_create_buffer[DType.int32](total)

    # Deterministic init: LCG seeds itemsets first row/col to values in [1,10],
    # then set reference[i*mc+j] = BLOSUM62[itemsets[i*mc]][itemsets[j]],
    # then overwrite first row/col with -i*penalty.
    var s: UInt64 = 7
    with d_items.map_to_host() as ih, d_ref.map_to_host() as rh:
        for k in range(total):
            ih[k] = Int32(0)
            rh[k] = Int32(0)
        # First column
        for i in range(1, max_rows):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ih[i * max_cols] = Int32(Int(s >> 33) % 10 + 1)
        # First row
        for j in range(1, max_cols):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ih[j] = Int32(Int(s >> 33) % 10 + 1)
        # Reference matrix from BLOSUM62
        for i in range(1, max_cols):
            for j in range(1, max_rows):
                var a = Int(ih[i * max_cols])
                var b = Int(ih[j])
                rh[i * max_cols + j] = blosum62_at(a, b)
        # Overwrite first row/col with initial gap penalties
        for i in range(1, max_rows):
            ih[i * max_cols] = Int32(-i) * penalty
        for j in range(1, max_cols):
            ih[j] = Int32(-j) * penalty

    # Snapshot init for later reset + host reference
    var init_p = ctx.enqueue_create_buffer[DType.int32](total)
    with d_items.map_to_host() as ih, init_p.map_to_host() as h:
        for k in range(total): h[k] = ih[k]

    comptime BLOCK: Int = 128
    var n = dim

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(repeat):
        # Reset itemsets to init
        with d_items.map_to_host() as ih, init_p.map_to_host() as h:
            for k in range(total): ih[k] = h[k]
        for diag in range(2, 2 * n + 1):
            var i_min = 1 if diag <= n else diag - n
            var i_max = diag - 1 if diag <= n else n
            var count = i_max - i_min + 1
            var grid = (count + BLOCK - 1) // BLOCK
            ctx.enqueue_function[func=nw_diag](
                d_items.unsafe_ptr(), d_ref.unsafe_ptr(),
                Int32(max_cols), Int32(diag), penalty,
                Int32(i_min), Int32(i_max),
                grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    var s_per = Float64(perf_counter_ns() - t0) / 1e9 / Float64(repeat)
    print("Total kernel execution time:", s_per, "(s)")

    # Host reference
    var ref_out = ctx.enqueue_create_buffer[DType.int32](total)
    with ref_out.map_to_host() as roh, init_p.map_to_host() as ih,\
         d_ref.map_to_host() as rh:
        for k in range(total): roh[k] = ih[k]
        nw_host(roh.unsafe_ptr(), rh.unsafe_ptr(), max_cols, penalty)

    var ok = True
    with d_items.map_to_host() as gh, ref_out.map_to_host() as rh:
        for k in range(total):
            if gh[k] != rh[k]:
                if ok:
                    print("Mismatch at", k, "gpu=", gh[k], "cpu=", rh[k])
                ok = False
    if ok:
        print("PASS")
    else:
        print("FAIL")
