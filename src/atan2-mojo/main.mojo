# Mojo GPU port of the `atan2` HeCBench benchmark.
#
# The upstream benchmark evaluates polynomial approximations of atan2
# (multiple degrees). We port the SAME polynomials (identical hex-float
# coefficients as atan2-cuda's approx_atan2f_P<N>) — no libm atan2f is
# called, so Mojo's device-side "libm not available" limitation is a
# non-issue here. Verified against a Mojo CPU expectederence using the same
# polynomial evaluation.
#
# Usage: main.mojo <n> <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


@always_inline
def p3(x: Float32) -> Float32:
    return x * (Float32(-0.9724544984) + x * x * Float32(0.19249329))
@always_inline
def p5(x: Float32) -> Float32:
    var z = x * x
    return x * (Float32(-0.9953961) + z * (Float32(0.2885551) + z * Float32(-0.079072)))
@always_inline
def p7(x: Float32) -> Float32:
    var z = x * x
    return x * (Float32(-0.9992122) + z * (Float32(0.3212166) + z * (Float32(-0.14608383) + z * Float32(0.03898430))))
@always_inline
def p9(x: Float32) -> Float32:
    var z = x * x
    return x * (Float32(-0.99989164) + z * (Float32(0.33044818) + z * (Float32(-0.18092003) + z * (Float32(0.08517033) + z * Float32(-0.02083588)))))
@always_inline
def p11(x: Float32) -> Float32:
    var z = x * x
    return x * (Float32(-0.99997956) + z * (Float32(0.33262336) + z * (Float32(-0.19292307) + z * (Float32(0.11640030) + z * (Float32(-0.05262279) + z * Float32(0.01171770))))))
@always_inline
def p13(x: Float32) -> Float32:
    var z = x * x
    return x * (Float32(-0.999996) + z * (Float32(0.33313018) + z * (Float32(-0.19697285) + z * (Float32(0.13157541) + z * (Float32(-0.07948351) + z * (Float32(0.03356785) + z * Float32(-0.006810755)))))))
@always_inline
def p15(x: Float32) -> Float32:
    var z = x * x
    return x * (Float32(-0.99999905) + z * (Float32(0.33325148) + z * (Float32(-0.19790763) + z * (Float32(0.13867617) + z * (Float32(-0.09649014) + z * (Float32(0.05580616) + z * (Float32(-0.02180671) + z * Float32(0.004036))))))))


@always_inline
def sum_safe_atan2f(y: Float32, x: Float32) -> Float32:
    var pi4f: Float32 = Float32(0.7853981633974483)
    var pi34f: Float32 = Float32(2.356194490192345)
    var xs: Float32 = x
    if y == 0.0 and x == 0.0:
        xs = Float32(0.2)
    var absx: Float32 = xs
    if absx < 0.0: absx = -absx
    var absy: Float32 = y
    if absy < 0.0: absy = -absy
    var r: Float32 = (absx - absy) / (absx + absy)
    if xs < 0.0: r = -r
    var base_angle: Float32 = pi4f
    if xs < 0.0: base_angle = pi34f
    # Sum of 7 polynomial degrees at the same r
    var acc: Float32 = p3(r) + p5(r) + p7(r) + p9(r) + p11(r) + p13(r) + p15(r)
    var angle: Float32 = base_angle * Float32(7.0) + acc
    if y < 0.0: return -angle
    return angle


def atan2_kernel(x: UnsafePointer[Float32, MutAnyOrigin],
                 y: UnsafePointer[Float32, MutAnyOrigin],
                 r: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n: return
    r[i] = sum_safe_atan2f(y[i], x[i])


def main() raises:
    var args = argv()
    var n = 1_000_000
    var repeat = 100
    if len(args) >= 2: n = Int(atol(args[1]))
    if len(args) >= 3: repeat = Int(atol(args[2]))
    var ctx = DeviceContext()
    var d_x = ctx.enqueue_create_buffer[DType.float32](n)
    var d_y = ctx.enqueue_create_buffer[DType.float32](n)
    var d_r = ctx.enqueue_create_buffer[DType.float32](n)
    var s: UInt64 = 20260721
    with d_x.map_to_host() as xh, d_y.map_to_host() as yh:
        for i in range(n):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            xh[i] = Float32(Int((s >> 33) & UInt64(0x7fffffff))) / Float32(0x40000000) - Float32(1.0)
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            yh[i] = Float32(Int((s >> 33) & UInt64(0x7fffffff))) / Float32(0x40000000) - Float32(1.0)
    comptime BLOCK: Int = 256
    var grid = (n + BLOCK - 1) // BLOCK
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=atan2_kernel](
            d_x.unsafe_ptr(), d_y.unsafe_ptr(), d_r.unsafe_ptr(), n,
            grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(repeat)
    print("Average kernel execution time:", us, "(us)")

    # Host expectederence: same polynomials on the same inputs
    var max_err: Float32 = 0.0
    with d_x.map_to_host() as xh, d_y.map_to_host() as yh, d_r.map_to_host() as rh:
        var CHECK = 10000
        for i in range(CHECK):
            var vx: Float32 = xh[i]
            var vy: Float32 = yh[i]
            var expected: Float32 = sum_safe_atan2f(vy, vx)
            var d: Float32 = rh[i] - expected
            if d < 0.0: d = -d
            if d > max_err: max_err = d
    print("Max element error:", max_err)
    if max_err < Float32(1e-3):
        print("PASS")
    else:
        print("FAIL")
