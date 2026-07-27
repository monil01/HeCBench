# Mojo GPU port of the HeCBench `aidw` (Adaptive Inverse Distance Weighted
# interpolation) benchmark.  Ports the non-tiled AIDW_Kernel — the tiled
# variant depends on __shared__ memory which is out of scope for this
# port.  Verifies against an in-Mojo CPU reference of the same algorithm.
#
# Usage: main.mojo <pts_in_1K> <check_flag> <iterations>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.random import seed, random_float64
from std.math import cos, pow, sqrt
from std.time import perf_counter_ns


alias BLOCK: Int = 256
alias R_MIN: Float32 = 0.0
alias R_MAX: Float32 = 2.0
alias A1: Float32 = 1.5
alias A2: Float32 = 2.0
alias A3: Float32 = 2.5
alias A4: Float32 = 3.0
alias A5: Float32 = 3.5


def alpha_for(u_R: Float32) -> Float32:
    var alpha: Float32 = 1.0
    if u_R >= 0.0 and u_R <= 0.1:
        alpha = A1
    if u_R > 0.1 and u_R <= 0.3:
        alpha = A1 * (1.0 - 5.0 * (u_R - 0.1)) + A2 * 5.0 * (u_R - 0.1)
    if u_R > 0.3 and u_R <= 0.5:
        alpha = A3 * 5.0 * (u_R - 0.3) + A1 * (1.0 - 5.0 * (u_R - 0.3))
    if u_R > 0.5 and u_R <= 0.7:
        alpha = A3 * (1.0 - 5.0 * (u_R - 0.5)) + A4 * 5.0 * (u_R - 0.5)
    if u_R > 0.7 and u_R <= 0.9:
        alpha = A5 * 5.0 * (u_R - 0.7) + A4 * (1.0 - 5.0 * (u_R - 0.7))
    if u_R > 0.9 and u_R <= 1.0:
        alpha = A5
    return alpha * 0.5


def aidw_kernel(
        dx: UnsafePointer[Float32, MutAnyOrigin],
        dy: UnsafePointer[Float32, MutAnyOrigin],
        dz: UnsafePointer[Float32, MutAnyOrigin],
        dnum: Int,
        ix: UnsafePointer[Float32, MutAnyOrigin],
        iy: UnsafePointer[Float32, MutAnyOrigin],
        iz: UnsafePointer[Float32, MutAnyOrigin],
        inum: Int,
        area: Float32,
        avg_dist: UnsafePointer[Float32, MutAnyOrigin]):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= inum:
        return
    var r_obs = avg_dist[tid]
    var r_exp = Float32(0.5) / sqrt(Float32(dnum) / area)
    var R_S0 = r_obs / r_exp
    var u_R: Float32 = 0.0
    if R_S0 >= R_MIN:
        u_R = 0.5 - 0.5 * cos(Float32(3.1415926) / R_MAX * (R_S0 - R_MIN))
    if R_S0 >= R_MAX:
        u_R = 1.0
    var alpha = alpha_for(u_R)

    var xi = ix[tid]
    var yi = iy[tid]
    var sum_dn: Float32 = 0.0
    var sum_up: Float32 = 0.0
    for j in range(dnum):
        var ex = xi - dx[j]
        var ey = yi - dy[j]
        var dist = ex * ex + ey * ey
        var t = Float32(1.0) / pow(dist, alpha)
        sum_dn += t
        sum_up += dz[j] * t
    iz[tid] = sum_up / sum_dn


def aidw_cpu(
        dx: UnsafePointer[Float32, MutAnyOrigin],
        dy: UnsafePointer[Float32, MutAnyOrigin],
        dz: UnsafePointer[Float32, MutAnyOrigin],
        dnum: Int,
        ix: UnsafePointer[Float32, MutAnyOrigin],
        iy: UnsafePointer[Float32, MutAnyOrigin],
        iz: UnsafePointer[Float32, MutAnyOrigin],
        inum: Int,
        area: Float32,
        avg_dist: UnsafePointer[Float32, MutAnyOrigin]):
    for tid in range(inum):
        var r_obs = avg_dist[tid]
        var r_exp = Float32(0.5) / sqrt(Float32(dnum) / area)
        var R_S0 = r_obs / r_exp
        var u_R: Float32 = 0.0
        if R_S0 >= R_MIN:
            u_R = 0.5 - 0.5 * cos(Float32(3.1415926) / R_MAX * (R_S0 - R_MIN))
        if R_S0 >= R_MAX:
            u_R = 1.0
        var alpha = alpha_for(u_R)
        var xi = ix[tid]
        var yi = iy[tid]
        var sum_dn: Float32 = 0.0
        var sum_up: Float32 = 0.0
        for j in range(dnum):
            var ex = xi - dx[j]
            var ey = yi - dy[j]
            var dist = ex * ex + ey * ey
            var t = Float32(1.0) / pow(dist, alpha)
            sum_dn += t
            sum_up += dz[j] * t
        iz[tid] = sum_up / sum_dn


def main() raises:
    var args = argv()
    if len(args) != 4:
        print("Usage:", args[0], "<pts_in_1K> <check> <iterations>")
        exit(1)
    var numk = Int(atol(args[1]))
    var check = Int(atol(args[2]))
    var iters = Int(atol(args[3]))
    var dnum = numk * 1024
    var inum = dnum
    var width: Float32 = 2000.0
    var height: Float32 = 2000.0
    var area = width * height
    print("Size = :", numk, "K")
    print("dnum = :", dnum)
    print("inum = :", inum)

    var ctx = DeviceContext()
    var d_dx = ctx.enqueue_create_buffer[DType.float32](dnum)
    var d_dy = ctx.enqueue_create_buffer[DType.float32](dnum)
    var d_dz = ctx.enqueue_create_buffer[DType.float32](dnum)
    var d_ad = ctx.enqueue_create_buffer[DType.float32](dnum)
    var d_ix = ctx.enqueue_create_buffer[DType.float32](inum)
    var d_iy = ctx.enqueue_create_buffer[DType.float32](inum)
    var d_iz = ctx.enqueue_create_buffer[DType.float32](inum)
    # Reference buffer (host-visible)
    var d_ref = ctx.enqueue_create_buffer[DType.float32](inum)

    seed(123)
    with d_dx.map_to_host() as hx, d_dy.map_to_host() as hy, d_dz.map_to_host() as hz:
        for i in range(dnum):
            hx[i] = Float32(random_float64() * 1000.0)
            hy[i] = Float32(random_float64() * 1000.0)
            hz[i] = Float32(random_float64() * 1000.0)
    with d_ix.map_to_host() as hx, d_iy.map_to_host() as hy:
        for i in range(inum):
            hx[i] = Float32(random_float64() * 1000.0)
            hy[i] = Float32(random_float64() * 1000.0)
    with d_ad.map_to_host() as h:
        for i in range(dnum):
            h[i] = Float32(random_float64() * 3.0)

    var blocks = (inum + BLOCK - 1) // BLOCK

    # One warm-up + optional correctness check
    ctx.enqueue_function[func=aidw_kernel](
        d_dx.unsafe_ptr(), d_dy.unsafe_ptr(), d_dz.unsafe_ptr(), dnum,
        d_ix.unsafe_ptr(), d_iy.unsafe_ptr(), d_iz.unsafe_ptr(), inum,
        area, d_ad.unsafe_ptr(),
        grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()

    if check == 1:
        print("Verification enabled")
        # Compute CPU reference into d_ref (mapped)
        with d_dx.map_to_host() as hx, d_dy.map_to_host() as hy, d_dz.map_to_host() as hz, \
             d_ix.map_to_host() as ix, d_iy.map_to_host() as iy, d_ad.map_to_host() as had, \
             d_ref.map_to_host() as href:
            aidw_cpu(hx.unsafe_ptr(), hy.unsafe_ptr(), hz.unsafe_ptr(), dnum,
                     ix.unsafe_ptr(), iy.unsafe_ptr(), href.unsafe_ptr(), inum,
                     area, had.unsafe_ptr())
        var ok = True
        var mism = 0
        with d_iz.map_to_host() as hgpu, d_ref.map_to_host() as href:
            for i in range(inum):
                var g = hgpu[i]
                var r = href[i]
                var diff = g - r
                if diff < 0.0:
                    diff = -diff
                # EPS = 1 in the CUDA reference (loose)
                if diff > 1.0:
                    if mism < 5:
                        print("mismatch", i, g, r)
                    mism += 1
                    ok = False
        if ok:
            print("PASS")
        else:
            print("FAIL:", mism)
    else:
        print("Verification disabled")

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[func=aidw_kernel](
            d_dx.unsafe_ptr(), d_dy.unsafe_ptr(), d_dz.unsafe_ptr(), dnum,
            d_ix.unsafe_ptr(), d_iy.unsafe_ptr(), d_iz.unsafe_ptr(), inum,
            area, d_ad.unsafe_ptr(),
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("Average execution time of AIDW_Kernel      ",
          Float64(t1 - t0) * 1e-9 / Float64(iters), "(s)")
