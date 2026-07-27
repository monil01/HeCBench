# Mojo GPU port of the HeCBench `inversek2j` benchmark.
#
# Cyclic-coordinate-descent (CCD) inverse kinematics for a 3-joint planar
# manipulator. Each input coordinate (x_target, y_target) is solved
# independently by MAX_LOOP CCD iterations.
#
# Usage: main.mojo <coord_in.txt> <iterations>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.math import acos, sqrt
from std.time import perf_counter_ns


alias NUM_JOINTS: Int = 3
alias NUM_JOINTS_P1: Int = 4
alias MAX_LOOP: Int = 25
alias BLOCK: Int = 128
alias PI: Float32 = 3.14159265358979


def solve(cx: Float32, cy: Float32, aout: UnsafePointer[Float32, MutAnyOrigin]):
    # Stack-local arrays, unrolled since NUM_JOINTS_P1 == 4.
    var a0: Float32 = 0.0
    var a1: Float32 = 0.0
    var a2: Float32 = 0.0
    var xd0: Float32 = 0.0
    var xd1: Float32 = 1.0
    var xd2: Float32 = 2.0
    var xd3: Float32 = 3.0
    var yd0: Float32 = 0.0
    var yd1: Float32 = 0.0
    var yd2: Float32 = 0.0
    var yd3: Float32 = 0.0
    # We don't actually need to update xdata/ydata between CCD iters because
    # the reference code doesn't update them either — it just resets a
    # cumulative sum each pass. Match the reference exactly.

    for _ in range(MAX_LOOP):
        for iter_r in range(NUM_JOINTS, 0, -1):
            # end-effector = joint 3
            var pe_x: Float32 = xd3
            var pe_y: Float32 = yd3
            var pc_x: Float32
            var pc_y: Float32
            if iter_r - 1 == 0:
                pc_x = xd0; pc_y = yd0
            elif iter_r - 1 == 1:
                pc_x = xd1; pc_y = yd1
            else:
                pc_x = xd2; pc_y = yd2
            var dpx = pe_x - pc_x
            var dpy = pe_y - pc_y
            var dtx = cx - pc_x
            var dty = cy - pc_y
            var lp = sqrt(dpx * dpx + dpy * dpy)
            var lt = sqrt(dtx * dtx + dty * dty)
            var ax = dpx / lp
            var ay = dpy / lp
            var bx = dtx / lt
            var by = dty / lt
            var adotb = ax * bx + ay * by
            if adotb > 1.0:
                adotb = 1.0
            elif adotb < -1.0:
                adotb = -1.0
            var angle: Float32 = acos(adotb) * (180.0 / PI)
            var direction = ax * by - ay * bx
            if direction < 0.0:
                angle = -angle
            if angle > 30.0:
                angle = 30.0
            elif angle < -30.0:
                angle = -30.0
            if iter_r - 1 == 0:
                a0 = angle
            elif iter_r - 1 == 1:
                a1 = angle
            else:
                a2 = angle
            # Reference: for i in range(NUM_JOINTS-1): angle_out[i+1] += angle_out[i]
            # i=0: a1 += a0; i=1: a2 += a1
            a1 = a1 + a0
            a2 = a2 + a1
    aout[0] = a0
    aout[1] = a1
    aout[2] = a2


def invkin_kernel(
        xt: UnsafePointer[Float32, MutAnyOrigin],
        yt: UnsafePointer[Float32, MutAnyOrigin],
        ang: UnsafePointer[Float32, MutAnyOrigin],
        size: Int):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= size:
        return
    solve(xt[idx], yt[idx], ang + idx * NUM_JOINTS)


def invkin_cpu(xt: UnsafePointer[Float32, MutAnyOrigin],
               yt: UnsafePointer[Float32, MutAnyOrigin],
               ang: UnsafePointer[Float32, MutAnyOrigin],
               size: Int):
    for idx in range(size):
        solve(xt[idx], yt[idx], ang + idx * NUM_JOINTS)


def main() raises:
    var args = argv()
    if len(args) != 3:
        print("Usage:", args[0], "<coord_in.txt> <iterations>")
        exit(1)
    var path = String(args[1])
    var iteration = Int(atol(args[2]))

    # Parse file: first token is the number of coords, then pairs of floats.
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var tokens = text.split()

    if len(tokens) < 1:
        print("empty input")
        exit(1)
    var data_size = Int(atol(String(tokens[0])))
    print("# Data Size =", data_size)
    if len(tokens) < 1 + 2 * data_size:
        print("input file has fewer coordinates than declared size")
        exit(1)

    var ctx = DeviceContext()
    var d_x = ctx.enqueue_create_buffer[DType.float32](data_size)
    var d_y = ctx.enqueue_create_buffer[DType.float32](data_size)
    var d_a = ctx.enqueue_create_buffer[DType.float32](data_size * NUM_JOINTS)

    with d_x.map_to_host() as hx, d_y.map_to_host() as hy:
        for i in range(data_size):
            hx[i] = Float32(atof(String(tokens[1 + 2 * i])))
            hy[i] = Float32(atof(String(tokens[2 + 2 * i])))
    print("# Coordinates are read from file...")
    print("# Memory allocation on GPU is done...")
    print("# Data are transfered to GPU...")

    var blocks = (data_size + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iteration):
        ctx.enqueue_function[func=invkin_kernel](
            d_x.unsafe_ptr(), d_y.unsafe_ptr(), d_a.unsafe_ptr(),
            data_size,
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("Average kernel execution time",
          Float64(t1 - t0) * 1e-3 / Float64(iteration), "(us)")

    # CPU reference stored in a host-mapped device buffer.
    var d_ref = ctx.enqueue_create_buffer[DType.float32](data_size * NUM_JOINTS)
    with d_ref.map_to_host() as href, d_x.map_to_host() as hx, d_y.map_to_host() as hy:
        for i in range(data_size * NUM_JOINTS):
            href[i] = 0.0
        invkin_cpu(hx.unsafe_ptr(), hy.unsafe_ptr(), href.unsafe_ptr(), data_size)

    var errors: Int = 0
    with d_a.map_to_host() as ha, d_ref.map_to_host() as href:
        for i in range(data_size):
            for j in range(NUM_JOINTS):
                var g = ha[i * NUM_JOINTS + j]
                var r = href[i * NUM_JOINTS + j]
                var diff = g - r
                if diff < 0.0:
                    diff = -diff
                if diff > 1e-3:
                    errors += 1

    if errors == 0:
        print("PASS")
    else:
        print("FAIL:", errors, "mismatches")
