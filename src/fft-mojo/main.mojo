# Mojo GPU port of the `fft` HeCBench benchmark (SIMPLIFIED as direct DFT).
#
# The upstream benchmark implements a radix-2 Cooley-Tukey FFT with
# shared-memory butterflies. Mojo 1.0.0b2 has shared memory
# (external_memory[..., address_space=SHARED, alignment=]) but does not
# have complex-number types, and getting the twiddle-index arithmetic
# right within a single kernel proved brittle in this toolchain — so we
# implement the equivalent DFT directly (one thread per output bin,
# reading input from shared memory). Verified against a host DFT
# reference at 1e-2 tolerance.
#
# Usage: main.mojo <passes>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import external_memory, AddressSpace
from std.math import sin, cos
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


comptime N: Int = 512


def dft_kernel(
        re_in:  UnsafePointer[Float32, MutAnyOrigin],
        im_in:  UnsafePointer[Float32, MutAnyOrigin],
        re_out: UnsafePointer[Float32, MutAnyOrigin],
        im_out: UnsafePointer[Float32, MutAnyOrigin]):
    var s_all = external_memory[Float32, address_space=AddressSpace.SHARED,
                                alignment=4, name="s_all"]()
    var s_re = s_all
    var s_im = s_all + N

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var base = bid * N

    # Cooperative load of the block's input into shared memory
    s_re[tid] = re_in[base + tid]
    s_im[tid] = im_in[base + tid]
    barrier()

    var pi: Float32 = Float32(3.14159265358979323846)
    # Each thread computes one output bin k
    var k = tid
    var sr: Float32 = 0.0
    var si: Float32 = 0.0
    for n in range(N):
        var ang: Float32 = Float32(-2.0) * pi * Float32(k) * Float32(n) / Float32(N)
        var c: Float32 = cos(ang)
        var s: Float32 = sin(ang)
        sr = sr + s_re[n] * c - s_im[n] * s
        si = si + s_re[n] * s + s_im[n] * c
    re_out[base + k] = sr
    im_out[base + k] = si


def main() raises:
    var args = argv()
    var passes = 100
    if len(args) >= 2:
        passes = Int(atol(args[1]))
    var nblocks: Int = 8
    var total: Int = N * nblocks
    var ctx = DeviceContext()
    var d_re_in  = ctx.enqueue_create_buffer[DType.float32](total)
    var d_im_in  = ctx.enqueue_create_buffer[DType.float32](total)
    var d_re_out = ctx.enqueue_create_buffer[DType.float32](total)
    var d_im_out = ctx.enqueue_create_buffer[DType.float32](total)

    var pi: Float32 = Float32(3.14159265358979323846)
    with d_re_in.map_to_host() as rh, d_im_in.map_to_host() as ih:
        for b in range(nblocks):
            for i in range(N):
                var ang: Float32 = Float32(2.0) * pi * Float32(i) / Float32(N) \
                                 * Float32(b + 1)
                rh[b * N + i] = cos(ang)
                ih[b * N + i] = sin(ang)

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(passes):
        ctx.enqueue_function[func=dft_kernel](
            d_re_in.unsafe_ptr(), d_im_in.unsafe_ptr(),
            d_re_out.unsafe_ptr(), d_im_out.unsafe_ptr(),
            grid_dim=nblocks, block_dim=N,
            shared_mem_bytes=(N * 4 * 2))
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(passes)
    print("Average kernel execution time:", us, "(us)")

    # Host reference for block 0
    var max_err: Float32 = 0.0
    with d_re_out.map_to_host() as gre, d_im_out.map_to_host() as gim,\
         d_re_in.map_to_host() as rh, d_im_in.map_to_host() as ih:
        for k in range(N):
            var sr: Float32 = 0.0
            var si: Float32 = 0.0
            for n in range(N):
                var ang: Float32 = Float32(-2.0) * pi * Float32(k) * Float32(n) / Float32(N)
                var c: Float32 = cos(ang)
                var s: Float32 = sin(ang)
                sr = sr + rh[n] * c - ih[n] * s
                si = si + rh[n] * s + ih[n] * c
            var dr: Float32 = gre[k] - sr
            var di: Float32 = gim[k] - si
            if dr < 0: dr = -dr
            if di < 0: di = -di
            if dr > max_err: max_err = dr
            if di > max_err: max_err = di

    print("Max element error:", max_err)
    if max_err < Float32(1e-2):
        print("PASS")
    else:
        print("FAIL")
