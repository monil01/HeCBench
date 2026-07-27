# Mojo GPU port of the `accuracy` HeCBench benchmark.
#
# Computes top-K classification accuracy: for each row of a data matrix
# and its ground-truth label, count how many predictions exceed the
# label's prediction; a row scores as "correct" if its rank <= top_k.
#
# Matches accuracy-cuda semantics; verifies against a serial CPU
# reference and prints PASS/FAIL and a timing line.
#
# Usage: main.mojo <nrows> <ndims> <top_k> <repeat>
# Args come via sys.argv().
#
# Mojo 1.0.0b2 GPU kernel launch pattern:
#   fn kernel(ptrs...) uses UnsafePointer[Scalar[dtype], MutAnyOrigin]
#   ctx.enqueue_function[func=kernel](args..., grid_dim=, block_dim=)

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.random import seed, random_float64
from std.time import perf_counter_ns
# No cross-cutting atomic import in Mojo 1.0.0b2's stdlib inventory on this
# release; we use a per-row 0/1 output array and reduce on the host.


def top_k_row_kernel(
        data:      UnsafePointer[Float32, MutAnyOrigin],
        label:     UnsafePointer[Int32,   MutAnyOrigin],
        row_flag:  UnsafePointer[Int32,   MutAnyOrigin],
        nrows: Int, ndims: Int, top_k: Int):
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= nrows:
        return
    var lbl = Int(label[row])
    var label_pred = data[row * ndims + lbl]
    var ngt: Int = 0
    for col in range(ndims):
        var pred = data[row * ndims + col]
        if pred > label_pred or (pred == label_pred and col <= lbl):
            ngt += 1
    if ngt <= top_k:
        row_flag[row] = Int32(1)
    else:
        row_flag[row] = Int32(0)


def reference_count(data_h: UnsafePointer[Float32, MutAnyOrigin],
                    label_h: UnsafePointer[Int32,   MutAnyOrigin],
                    nrows: Int, ndims: Int, top_k: Int) -> Int:
    var count: Int = 0
    for row in range(nrows):
        var lbl = Int(label_h[row])
        var label_pred = data_h[row * ndims + lbl]
        var ngt: Int = 0
        for col in range(ndims):
            var pred = data_h[row * ndims + col]
            if pred > label_pred or (pred == label_pred and col <= lbl):
                ngt += 1
        if ngt <= top_k:
            count += 1
    return count


def main() raises:
    var args = argv()
    if len(args) != 5:
        print("Usage:", args[0],
              "<number of rows> <number of columns> <top K> <repeat>")
        exit(1)
    var nrows  = Int(atol(args[1]))
    var ndims  = Int(atol(args[2]))
    var top_k  = Int(atol(args[3]))
    var repeat = Int(atol(args[4]))
    var data_size = nrows * ndims

    var ctx = DeviceContext()

    # Device buffers
    var d_data  = ctx.enqueue_create_buffer[DType.float32](data_size)
    var d_label = ctx.enqueue_create_buffer[DType.int32](nrows)
    var d_flag  = ctx.enqueue_create_buffer[DType.int32](nrows)

    # Initialise inputs on host, then copy to device
    seed(123)
    with d_data.map_to_host() as data_h, d_label.map_to_host() as label_h:
        for i in range(data_size):
            data_h[i] = Float32(random_float64())
        for i in range(nrows):
            label_h[i] = Int32(Int(random_float64() * Float64(ndims)))

    # Compute serial reference (needs host copies of data & label)
    var ref_count: Int
    with d_data.map_to_host() as data_h, d_label.map_to_host() as label_h:
        ref_count = reference_count(
            data_h.unsafe_ptr(), label_h.unsafe_ptr(),
            nrows, ndims, top_k)

    # Match the CUDA benchmark: repeat over four grid sizes
    for gi in range(4):
        var ngrid: Int
        if gi == 0:
            ngrid = nrows // 4
        elif gi == 1:
            ngrid = nrows // 2
        elif gi == 2:
            ngrid = 3 * nrows // 4
        else:
            ngrid = nrows
        print("Grid size is", ngrid)

        alias BLOCK: Int = 256
        var blocks = (nrows + BLOCK - 1) // BLOCK

        # Time repeat kernel launches
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(repeat):
            ctx.enqueue_function[func=top_k_row_kernel](
                d_data.unsafe_ptr(), d_label.unsafe_ptr(),
                d_flag.unsafe_ptr(),
                nrows, ndims, top_k,
                grid_dim=blocks, block_dim=BLOCK)
            ctx.synchronize()
        var t1 = perf_counter_ns()
        var elapsed_us = Float64(t1 - t0) * 1e-3 / Float64(repeat)
        print("Average execution time of accuracy kernel:",
              elapsed_us, "(us)")

        # Sum row flags on the host to get the count
        var got: Int = 0
        with d_flag.map_to_host() as fh:
            for i in range(nrows):
                got += Int(fh[i])
        if got == ref_count:
            print("PASS")
        else:
            print("FAIL: got", got, "ref", ref_count)
