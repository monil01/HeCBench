# daphne Mojo port: LiDAR-to-image projection surrogate mirroring the Triton
# reference at ../daphne-triton/main.py — synthetic LCG point cloud;
# per-point extrinsic rotation, radial+tangential undistortion, intrinsic
# projection, pt2>2.5 visibility filter. Verified against a Mojo host reference.

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns

comptime R00: Float32 = -0.9111348390579224
comptime R01: Float32 =  0.0751304179430008
comptime R02: Float32 = -0.4052018225193024
comptime R10: Float32 = -0.3360927104949951
comptime R11: Float32 = -0.7044632434844971
comptime R12: Float32 =  0.6251187324523926
comptime R20: Float32 = -0.2384843230247498
comptime R21: Float32 =  0.7057529091835022
comptime R22: Float32 =  0.6671117544174194
comptime T0: Float32 = 0.1
comptime T1: Float32 = -0.2
comptime T2: Float32 = 0.3
comptime D0: Float32 = 0.03
comptime D1: Float32 = -0.15
comptime D2: Float32 = 0.001
comptime D3: Float32 = 0.001
comptime D4: Float32 = 0.05
comptime FX: Float32 = 1200.0
comptime CX: Float32 = 400.0
comptime FY: Float32 = 1200.0
comptime CY: Float32 = 300.0
comptime POINT_STEP: Int = 8


def project_kernel(cp: UnsafePointer[Float32, MutAnyOrigin],
                   ox: UnsafePointer[Float32, MutAnyOrigin],
                   oy: UnsafePointer[Float32, MutAnyOrigin],
                   oz: UnsafePointer[Float32, MutAnyOrigin],
                   ov: UnsafePointer[Int32, MutAnyOrigin],
                   n: Int):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    var base = i * POINT_STEP
    var p0 = cp[base]
    var p1 = cp[base + 1]
    var p2 = cp[base + 2]
    var pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02
    var pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12
    var pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22
    var close = pt2 > Float32(2.5)
    var denom: Float32 = pt2 if close else Float32(1.0)
    var tmpx = pt0 / denom
    var tmpy = pt1 / denom
    var r2 = tmpx * tmpx + tmpy * tmpy
    var tmpdist = Float32(1.0) + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2
    var ix = tmpx * tmpdist + Float32(2.0) * D2 * tmpx * tmpy + D3 * (r2 + Float32(2.0) * tmpx * tmpx)
    var iy = tmpy * tmpdist + D2 * (r2 + Float32(2.0) * tmpy * tmpy) + Float32(2.0) * D3 * tmpx * tmpy
    var ux = FX * ix + CX
    var uy = FY * iy + CY
    if close:
        ox[i] = ux + Float32(0.5)
        oy[i] = uy + Float32(0.5)
        oz[i] = pt2 * Float32(100.0)
        ov[i] = Int32(1)
    else:
        ox[i] = Float32(0.0)
        oy[i] = Float32(0.0)
        oz[i] = Float32(0.0)
        ov[i] = Int32(0)


def main() raises:
    var args = argv()
    var n_batches = 1
    var i = 1
    while i + 1 < len(args):
        if args[i] == "-p":
            n_batches = Int(atol(args[i + 1]))
        i += 1

    var n_points = 100000
    print("[note] synthetic", n_points, "points x", n_batches, "batches")

    var ctx = DeviceContext()
    var d_cp = ctx.enqueue_create_buffer[DType.float32](n_points * POINT_STEP)
    var d_x  = ctx.enqueue_create_buffer[DType.float32](n_points)
    var d_y  = ctx.enqueue_create_buffer[DType.float32](n_points)
    var d_z  = ctx.enqueue_create_buffer[DType.float32](n_points)
    var d_v  = ctx.enqueue_create_buffer[DType.int32](n_points)

    var ok_all = True
    var total_us: Float64 = 0.0

    for b in range(n_batches):
        var s: UInt64 = UInt64(20260721 + (b + 1))
        with d_cp.map_to_host() as cph:
            for i in range(n_points * POINT_STEP):
                cph[i] = Float32(0.0)
            for i in range(n_points):
                s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
                var u0 = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
                s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
                var u1 = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
                s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
                var u2 = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
                s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
                var u3 = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
                var base = i * POINT_STEP
                cph[base]     = (u0 - Float32(0.5)) * Float32(2.0)
                cph[base + 1] = (u1 - Float32(0.5)) * Float32(2.0)
                cph[base + 2] = u2 * Float32(10.0) + Float32(15.0)
                cph[base + 4] = u3

        comptime BLOCK: Int = 256
        var blocks = (n_points + BLOCK - 1) // BLOCK

        ctx.synchronize()
        var t0 = perf_counter_ns()
        ctx.enqueue_function[func=project_kernel](
            d_cp.unsafe_ptr(), d_x.unsafe_ptr(), d_y.unsafe_ptr(),
            d_z.unsafe_ptr(), d_v.unsafe_ptr(), n_points,
            grid_dim=blocks, block_dim=BLOCK)
        ctx.synchronize()
        total_us += Float64(perf_counter_ns() - t0) / 1e3

        var kept: Int = 0
        var mismatch: Int = 0
        var max_err: Float32 = 0.0
        with d_cp.map_to_host() as cph, d_x.map_to_host() as xh, d_y.map_to_host() as yh, d_z.map_to_host() as zh, d_v.map_to_host() as vh:
            for j in range(n_points):
                var base = j * POINT_STEP
                var p0 = cph[base]
                var p1 = cph[base + 1]
                var p2 = cph[base + 2]
                var pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02
                var pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12
                var pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22
                var close = pt2 > Float32(2.5)
                var denom: Float32 = pt2 if close else Float32(1.0)
                var tmpx = pt0 / denom
                var tmpy = pt1 / denom
                var r2 = tmpx * tmpx + tmpy * tmpy
                var tmpdist = Float32(1.0) + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2
                var ix = tmpx * tmpdist + Float32(2.0) * D2 * tmpx * tmpy + D3 * (r2 + Float32(2.0) * tmpx * tmpx)
                var iy = tmpy * tmpdist + D2 * (r2 + Float32(2.0) * tmpy * tmpy) + Float32(2.0) * D3 * tmpx * tmpy
                var rx: Float32
                var ry: Float32
                var rz: Float32
                var rv: Int32
                if close:
                    rx = FX * ix + CX + Float32(0.5)
                    ry = FY * iy + CY + Float32(0.5)
                    rz = pt2 * Float32(100.0)
                    rv = Int32(1)
                    kept += 1
                else:
                    rx = Float32(0.0); ry = Float32(0.0); rz = Float32(0.0); rv = Int32(0)
                if vh[j] != rv:
                    mismatch += 1
                    continue
                if rv == Int32(1):
                    var e = xh[j] - rx
                    if e < Float32(0.0): e = -e
                    if e > max_err: max_err = e
                    e = yh[j] - ry
                    if e < Float32(0.0): e = -e
                    if e > max_err: max_err = e
                    e = zh[j] - rz
                    if e < Float32(0.0): e = -e
                    if e > max_err: max_err = e
        print("[batch]", b, "kept=", kept, "/", n_points, "max_err=", max_err, "valid_mismatch=", mismatch)
        if mismatch != 0 or max_err > Float32(1e-3):
            ok_all = False

    print("Average kernel execution time:", total_us / Float64(n_batches), "(us)")
    if ok_all:
        print("PASS")
    else:
        print("FAIL")
