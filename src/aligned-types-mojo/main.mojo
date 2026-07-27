# Mojo GPU port of the `aligned-types` HeCBench benchmark (simplified).
#
# The upstream CUDA benchmark compares aligned vs. misaligned struct
# copy throughput across 13 element sizes. Mojo 1.0.0b2 does not
# expose C-style struct alignment control the same way, so this port
# covers the Mojo-relevant subset: element-wise copy kernels for
# 1-byte, 2-byte, 4-byte, and 8-byte types. Each kernel is a simple
# `dst[i] = src[i]`; correctness is verified against a host copy.
#
# Usage: main.mojo (no args needed)

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import exit
from std.time import perf_counter_ns


def copy_u8(src: UnsafePointer[UInt8, MutAnyOrigin],
             dst: UnsafePointer[UInt8, MutAnyOrigin], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[i] = src[i]


def copy_u16(src: UnsafePointer[UInt16, MutAnyOrigin],
              dst: UnsafePointer[UInt16, MutAnyOrigin], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[i] = src[i]


def copy_u32(src: UnsafePointer[UInt32, MutAnyOrigin],
              dst: UnsafePointer[UInt32, MutAnyOrigin], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[i] = src[i]


def copy_u64(src: UnsafePointer[UInt64, MutAnyOrigin],
              dst: UnsafePointer[UInt64, MutAnyOrigin], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[i] = src[i]


def main() raises:
    var ctx = DeviceContext()
    var n = 1_000_000

    var d_src_u8 = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_dst_u8 = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_src_u16 = ctx.enqueue_create_buffer[DType.uint16](n)
    var d_dst_u16 = ctx.enqueue_create_buffer[DType.uint16](n)
    var d_src_u32 = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_dst_u32 = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_src_u64 = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_dst_u64 = ctx.enqueue_create_buffer[DType.uint64](n)

    with d_src_u8.map_to_host() as h:
        for i in range(n): h[i] = UInt8(i & 0xFF)
    with d_src_u16.map_to_host() as h:
        for i in range(n): h[i] = UInt16(i & 0xFFFF)
    with d_src_u32.map_to_host() as h:
        for i in range(n): h[i] = UInt32(i)
    with d_src_u64.map_to_host() as h:
        for i in range(n): h[i] = UInt64(i)

    comptime BLOCK: Int = 256
    var blocks = (n + BLOCK - 1) // BLOCK

    var results = 0
    ctx.synchronize()

    var t0 = perf_counter_ns()
    ctx.enqueue_function[func=copy_u8](d_src_u8.unsafe_ptr(), d_dst_u8.unsafe_ptr(), n,
                                        grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var u8_ns = perf_counter_ns() - t0
    print("uint8   copy:", Float64(u8_ns)/1e3, "(us)")

    t0 = perf_counter_ns()
    ctx.enqueue_function[func=copy_u16](d_src_u16.unsafe_ptr(), d_dst_u16.unsafe_ptr(), n,
                                         grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    print("uint16  copy:", Float64(perf_counter_ns()-t0)/1e3, "(us)")

    t0 = perf_counter_ns()
    ctx.enqueue_function[func=copy_u32](d_src_u32.unsafe_ptr(), d_dst_u32.unsafe_ptr(), n,
                                         grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    print("uint32  copy:", Float64(perf_counter_ns()-t0)/1e3, "(us)")

    t0 = perf_counter_ns()
    ctx.enqueue_function[func=copy_u64](d_src_u64.unsafe_ptr(), d_dst_u64.unsafe_ptr(), n,
                                         grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    print("uint64  copy:", Float64(perf_counter_ns()-t0)/1e3, "(us)")

    # Verify all four
    var ok = True
    with d_dst_u8.map_to_host() as h:
        for i in range(n):
            if h[i] != UInt8(i & 0xFF): ok = False; break
    with d_dst_u16.map_to_host() as h:
        for i in range(n):
            if h[i] != UInt16(i & 0xFFFF): ok = False; break
    with d_dst_u32.map_to_host() as h:
        for i in range(n):
            if h[i] != UInt32(i): ok = False; break
    with d_dst_u64.map_to_host() as h:
        for i in range(n):
            if h[i] != UInt64(i): ok = False; break
    if ok:
        print("[alignedTypes] -> Test Results: 0 Failures")
        print("PASS")
    else:
        print("FAIL")
