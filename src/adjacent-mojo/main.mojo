# Mojo GPU port of `adjacent` — block-wide adjacent difference.
#
# For each contiguous block of BLOCK_SIZE*4 integers, compute either
#   out[i] = i-1<0 ? in[i] : in[i]-in[i-1]        (subtract-left)
# or
#   out[i] = i+1>=B ? in[i] : in[i]-in[i+1]       (subtract-right)
#
# Verifies against a Mojo CPU reference and prints PASS/FAIL.
#
# Usage: main.mojo <num_elements> <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


def adj_left_kernel(
        d_in:  UnsafePointer[Int32, MutAnyOrigin],
        d_out: UnsafePointer[Int32, MutAnyOrigin],
        n: Int, items_per_block: Int):
    var tid_in_block = thread_idx.x
    var b = block_idx.x
    var base = b * items_per_block
    for k in range(4):
        var idx = tid_in_block * 4 + k
        if idx < items_per_block and base + idx < n:
            var i = base + idx
            if idx == 0:
                d_out[i] = d_in[i]
            else:
                d_out[i] = d_in[i] - d_in[i-1]


def adj_right_kernel(
        d_in:  UnsafePointer[Int32, MutAnyOrigin],
        d_out: UnsafePointer[Int32, MutAnyOrigin],
        n: Int, items_per_block: Int):
    var tid_in_block = thread_idx.x
    var b = block_idx.x
    var base = b * items_per_block
    for k in range(4):
        var idx = tid_in_block * 4 + k
        if idx < items_per_block and base + idx < n:
            var i = base + idx
            if idx == items_per_block - 1:
                d_out[i] = d_in[i]
            else:
                d_out[i] = d_in[i] - d_in[i+1]


def test_size(block_threads: Int, num_items_in: Int, repeat: Int,
              ctx: DeviceContext) raises:
    var items_per_block = block_threads * 4
    var num_items = ((num_items_in + items_per_block - 1) // items_per_block) * items_per_block
    var grid = num_items // items_per_block

    var d_in  = ctx.enqueue_create_buffer[DType.int32](num_items)
    var d_out = ctx.enqueue_create_buffer[DType.int32](num_items)

    with d_in.map_to_host() as h_in:
        for i in range(num_items):
            h_in[i] = Int32(i % 17)

    # subtract-left check
    ctx.enqueue_function[func=adj_left_kernel](
        d_in.unsafe_ptr(), d_out.unsafe_ptr(),
        num_items, items_per_block,
        grid_dim=grid, block_dim=block_threads)
    ctx.synchronize()

    var ok = True
    with d_in.map_to_host() as h_in, d_out.map_to_host() as h_out:
        for b in range(grid):
            for i in range(items_per_block):
                var idx = b * items_per_block + i
                var want: Int32
                if i == 0:
                    want = h_in[idx]
                else:
                    want = h_in[idx] - h_in[idx-1]
                if h_out[idx] != want:
                    ok = False
    if ok:
        print("PASS")
    else:
        print("FAIL")

    # subtract-right check
    ctx.enqueue_function[func=adj_right_kernel](
        d_in.unsafe_ptr(), d_out.unsafe_ptr(),
        num_items, items_per_block,
        grid_dim=grid, block_dim=block_threads)
    ctx.synchronize()

    ok = True
    with d_in.map_to_host() as h_in, d_out.map_to_host() as h_out:
        for b in range(grid):
            for i in range(items_per_block):
                var idx = b * items_per_block + i
                var want: Int32
                if i == items_per_block - 1:
                    want = h_in[idx]
                else:
                    want = h_in[idx] - h_in[idx+1]
                if h_out[idx] != want:
                    ok = False
    if ok:
        print("PASS")
    else:
        print("FAIL")

    # timed 2-kernel back-to-back run
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=adj_left_kernel](
            d_in.unsafe_ptr(), d_out.unsafe_ptr(),
            num_items, items_per_block,
            grid_dim=grid, block_dim=block_threads)
        ctx.enqueue_function[func=adj_right_kernel](
            d_out.unsafe_ptr(), d_out.unsafe_ptr(),
            num_items, items_per_block,
            grid_dim=grid, block_dim=block_threads)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var us = Float64(t1 - t0) * 1e-3 / Float64(repeat)
    print("Average execution time of the kernels (thread block size =",
          block_threads, "):", us, "(us)")


def main() raises:
    var args = argv()
    if len(args) != 3:
        print("Usage:", args[0], "<number of elements> <repeat>")
        exit(1)
    var n      = Int(atol(args[1]))
    var repeat = Int(atol(args[2]))
    var ctx = DeviceContext()
    test_size(64,   n, repeat, ctx)
    test_size(128,  n, repeat, ctx)
    test_size(256,  n, repeat, ctx)
    test_size(512,  n, repeat, ctx)
    test_size(1024, n, repeat, ctx)
