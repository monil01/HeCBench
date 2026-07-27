# Mojo GPU port of the HeCBench `affine` benchmark.
#
# Applies an inverse-affine + bilinear resample to a 512x512 uint16 image.
# Verifies against an in-Mojo CPU reference and prints max absolute error.
#
# Usage: main.mojo <input.raw> <output.raw> <iterations>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.math import cos, sin, floor
from std.time import perf_counter_ns


alias X_SIZE: Int = 512
alias Y_SIZE: Int = 512
alias PI: Float32 = 3.14159265359
alias WHITE: Int = 1


def affine_kernel(
        src: UnsafePointer[UInt16, MutAnyOrigin],
        dst: UnsafePointer[UInt16, MutAnyOrigin]):
    var x = block_idx.x * block_dim.x + thread_idx.x
    var y = block_idx.y * block_dim.y + thread_idx.y
    if x >= X_SIZE or y >= Y_SIZE:
        return

    var lx_rot: Float32 = 30.0
    var ly_rot: Float32 = 0.0
    var lx_expan: Float32 = 0.5
    var ly_expan: Float32 = 0.5

    var a00 = lx_expan * cos(lx_rot * PI / 180.0)
    var a01 = ly_expan * sin(ly_rot * PI / 180.0)
    var a10 = lx_expan * sin(lx_rot * PI / 180.0)
    var a11 = ly_expan * cos(ly_rot * PI / 180.0)
    var beta0: Float32 = 0.0
    var beta1: Float32 = 0.0

    var det = a00 * a11 - a01 * a10
    var i00: Float32
    var i01: Float32
    var i10: Float32
    var i11: Float32
    var ib0: Float32
    var ib1: Float32
    if det == 0.0:
        i00 = 1.0; i01 = 0.0; i10 = 0.0; i11 = 1.0
        ib0 = -beta0; ib1 = -beta1
    else:
        i00 =  a11 / det
        i01 = -a01 / det
        i10 = -a10 / det
        i11 =  a00 / det
        ib0 = -i00 * beta0 - i01 * beta1
        ib1 = -i10 * beta0 - i11 * beta1

    var fx = Float32(x) - Float32(X_SIZE) / 2.0
    var fy = Float32(y) - Float32(Y_SIZE) / 2.0
    var x_new = ib0 + i00 * fx + i01 * fy + Float32(X_SIZE) / 2.0
    var y_new = ib1 + i10 * fx + i11 * fy + Float32(Y_SIZE) / 2.0
    var m = Int(floor(x_new))
    var n = Int(floor(y_new))
    var x_frac = x_new - Float32(m)
    var y_frac = y_new - Float32(n)

    var out: UInt16 = 0
    if m >= 0 and m + 1 < X_SIZE and n >= 0 and n + 1 < Y_SIZE:
        var g = (Float32(1.0) - y_frac) * ((Float32(1.0) - x_frac) * Float32(Int(src[n * X_SIZE + m])) + x_frac * Float32(Int(src[n * X_SIZE + m + 1]))) \
                + y_frac * ((Float32(1.0) - x_frac) * Float32(Int(src[(n + 1) * X_SIZE + m])) + x_frac * Float32(Int(src[(n + 1) * X_SIZE + m + 1])))
        out = UInt16(Int(g))
    elif (m + 1 == X_SIZE and n >= 0 and n < Y_SIZE) or (n + 1 == Y_SIZE and m >= 0 and m < X_SIZE):
        out = src[n * X_SIZE + m]
    else:
        out = UInt16(WHITE)
    dst[y * X_SIZE + x] = out


def affine_ref(src: UnsafePointer[UInt16, MutAnyOrigin],
               dst: UnsafePointer[UInt16, MutAnyOrigin]):
    for y in range(Y_SIZE):
        for x in range(X_SIZE):
            var lx_rot: Float32 = 30.0
            var ly_rot: Float32 = 0.0
            var lx_expan: Float32 = 0.5
            var ly_expan: Float32 = 0.5

            var a00 = lx_expan * cos(lx_rot * PI / 180.0)
            var a01 = ly_expan * sin(ly_rot * PI / 180.0)
            var a10 = lx_expan * sin(lx_rot * PI / 180.0)
            var a11 = ly_expan * cos(ly_rot * PI / 180.0)
            var beta0: Float32 = 0.0
            var beta1: Float32 = 0.0

            var det = a00 * a11 - a01 * a10
            var i00: Float32
            var i01: Float32
            var i10: Float32
            var i11: Float32
            var ib0: Float32
            var ib1: Float32
            if det == 0.0:
                i00 = 1.0; i01 = 0.0; i10 = 0.0; i11 = 1.0
                ib0 = -beta0; ib1 = -beta1
            else:
                i00 =  a11 / det
                i01 = -a01 / det
                i10 = -a10 / det
                i11 =  a00 / det
                ib0 = -i00 * beta0 - i01 * beta1
                ib1 = -i10 * beta0 - i11 * beta1

            var fx = Float32(x) - Float32(X_SIZE) / 2.0
            var fy = Float32(y) - Float32(Y_SIZE) / 2.0
            var x_new = ib0 + i00 * fx + i01 * fy + Float32(X_SIZE) / 2.0
            var y_new = ib1 + i10 * fx + i11 * fy + Float32(Y_SIZE) / 2.0
            var m = Int(floor(x_new))
            var n = Int(floor(y_new))
            var x_frac = x_new - Float32(m)
            var y_frac = y_new - Float32(n)

            var out: UInt16 = 0
            if m >= 0 and m + 1 < X_SIZE and n >= 0 and n + 1 < Y_SIZE:
                var g = (Float32(1.0) - y_frac) * ((Float32(1.0) - x_frac) * Float32(Int(src[n * X_SIZE + m])) + x_frac * Float32(Int(src[n * X_SIZE + m + 1]))) \
                        + y_frac * ((Float32(1.0) - x_frac) * Float32(Int(src[(n + 1) * X_SIZE + m])) + x_frac * Float32(Int(src[(n + 1) * X_SIZE + m + 1])))
                out = UInt16(Int(g))
            elif (m + 1 == X_SIZE and n >= 0 and n < Y_SIZE) or (n + 1 == Y_SIZE and m >= 0 and m < X_SIZE):
                out = src[n * X_SIZE + m]
            else:
                out = UInt16(WHITE)
            dst[y * X_SIZE + x] = out


def main() raises:
    var args = argv()
    if len(args) != 4:
        print("Usage:", args[0], "<input.raw> <output.raw> <iterations>")
        exit(1)
    var in_path = String(args[1])
    var out_path = String(args[2])
    var iters = Int(atol(args[3]))

    var pixels = X_SIZE * Y_SIZE

    var ctx = DeviceContext()
    var d_in  = ctx.enqueue_create_buffer[DType.uint16](pixels)
    var d_out = ctx.enqueue_create_buffer[DType.uint16](pixels)
    var d_ref = ctx.enqueue_create_buffer[DType.uint16](pixels)

    # Load the raw image
    print("Reading input image...")
    print("   Reading RAW Image")
    var content: List[UInt8]
    with open(in_path, "r") as f:
        content = f.read_bytes()
    print("   Bytes read =", len(content))
    if len(content) != pixels * 2:
        print("unexpected file size, expected", pixels * 2)
        exit(1)
    var src_bytes = content.unsafe_ptr()
    var src_words = src_bytes.bitcast[UInt16]()
    with d_in.map_to_host() as hin:
        for i in range(pixels):
            hin[i] = src_words[i]

    var grid = (X_SIZE // 16, Y_SIZE // 16)
    var block = (16, 16)

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[func=affine_kernel](
            d_in.unsafe_ptr(), d_out.unsafe_ptr(),
            grid_dim=grid, block_dim=block)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("   Average kernel execution time",
          Float64(t1 - t0) * 1e-9 / Float64(iters), "(s)")

    # CPU reference
    with d_in.map_to_host() as hin, d_ref.map_to_host() as href:
        affine_ref(hin.unsafe_ptr(), href.unsafe_ptr())
    var max_err: Int = 0
    with d_out.map_to_host() as hgpu, d_ref.map_to_host() as href:
        for i in range(pixels):
            var diff = Int(hgpu[i]) - Int(href[i])
            if diff < 0: diff = -diff
            if diff > max_err:
                max_err = diff
    print("   Max output error is", max_err)

    if max_err <= 1:
        print("PASS")
    else:
        print("FAIL")

    # Optional: write the output. Small file so cheap.
    print("   Writing RAW Image")
    with open(out_path, "w") as f, d_out.map_to_host() as hgpu:
        # Serialize by writing UInt16s as two bytes each.
        # `write_bytes` accepts a Span[Byte].
        var bytes = List[UInt8]()
        for i in range(pixels):
            var v = hgpu[i]
            bytes.append(UInt8(Int(v) & 0xFF))
            bytes.append(UInt8((Int(v) >> 8) & 0xFF))
        f.write_bytes(Span(bytes))
    print("   Bytes written =", pixels * 2)
