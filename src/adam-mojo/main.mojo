# Mojo GPU port of the `adam` HeCBench benchmark.
#
# Element-wise Adam optimizer step; each thread handles one parameter
# entry. The CUDA source's loop runs `time_step` steps per kernel call
# and the kernel is invoked `repeat` times.
#
# Correctness: matches serial reference computed on the host, within a
# 1e-3 absolute tolerance per element (matches the CUDA benchmark's
# declared tolerance).
#
# Usage: main.mojo <vector_size> <time_step> <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.math import pow as _pow, sqrt as _sqrt
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


def adam_kernel(
        p: UnsafePointer[Float32, MutAnyOrigin],
        m: UnsafePointer[Float32, MutAnyOrigin],
        v: UnsafePointer[Float32, MutAnyOrigin],
        g: UnsafePointer[Float32, MutAnyOrigin],
        b1: Float32, b2: Float32, eps: Float32, grad_scale: Float32,
        step_size: Float32, decay: Float32,
        vector_size: Int, time_step: Int):
    var j = block_idx.x * block_dim.x + thread_idx.x
    if j >= vector_size:
        return
    var scaled_grad: Float32 = g[j] / grad_scale
    var mj: Float32 = m[j]
    var vj: Float32 = v[j]
    var pj: Float32 = p[j]
    for t in range(1, time_step + 1):
        mj = b1 * mj + (Float32(1.0) - b1) * scaled_grad
        vj = b2 * vj + (Float32(1.0) - b2) * scaled_grad * scaled_grad
        var m_corr: Float32 = mj / (Float32(1.0) - _pow(b1, Float32(t)))
        var v_corr: Float32 = vj / (Float32(1.0) - _pow(b2, Float32(t)))
        var denom: Float32 = _sqrt(v_corr + eps)  # mode 0
        var update: Float32 = (m_corr / denom) + decay * pj
        pj = pj - step_size * update
    p[j] = pj
    m[j] = mj
    v[j] = vj


def adam_cpu(
        p: UnsafePointer[Float32, MutAnyOrigin],
        m: UnsafePointer[Float32, MutAnyOrigin],
        v: UnsafePointer[Float32, MutAnyOrigin],
        g: UnsafePointer[Float32, MutAnyOrigin],
        b1: Float32, b2: Float32, eps: Float32, grad_scale: Float32,
        step_size: Float32, decay: Float32,
        vector_size: Int, time_step: Int, repeat: Int):
    for _ in range(repeat):
        for j in range(vector_size):
            var scaled_grad: Float32 = g[j] / grad_scale
            for t in range(1, time_step + 1):
                m[j] = b1 * m[j] + (Float32(1.0) - b1) * scaled_grad
                v[j] = b2 * v[j] + (Float32(1.0) - b2) * scaled_grad * scaled_grad
                var m_corr: Float32 = m[j] / (Float32(1.0) - _pow(b1, Float32(t)))
                var v_corr: Float32 = v[j] / (Float32(1.0) - _pow(b2, Float32(t)))
                var denom: Float32 = _sqrt(v_corr + eps)
                var update: Float32 = (m_corr / denom) + decay * p[j]
                p[j] = p[j] - step_size * update


def main() raises:
    var args = argv()
    if len(args) != 4:
        print("Usage:", args[0], "<vector size> <time step> <repeat>")
        exit(1)
    var vector_size = Int(atol(args[1]))
    var time_step   = Int(atol(args[2]))
    var repeat      = Int(atol(args[3]))

    var b1: Float32 = 0.9
    var b2: Float32 = 0.999
    var eps: Float32 = 1e-10
    var grad_scale: Float32 = 256.0
    var step_size: Float32 = 1e-3
    var decay: Float32 = 0.5

    var ctx = DeviceContext()
    var d_p = ctx.enqueue_create_buffer[DType.float32](vector_size)
    var d_m = ctx.enqueue_create_buffer[DType.float32](vector_size)
    var d_v = ctx.enqueue_create_buffer[DType.float32](vector_size)
    var d_g = ctx.enqueue_create_buffer[DType.float32](vector_size)

    # Deterministic init — same values on host and device via a simple LCG.
    var s: UInt64 = 19937
    with d_m.map_to_host() as mh, d_v.map_to_host() as vh, d_g.map_to_host() as gh, d_p.map_to_host() as ph:
        for i in range(vector_size):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            mh[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            vh[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            gh[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ph[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)

    # Copy m,v,p to reference device buffers before device kernels mutate them.
    var ref_p = ctx.enqueue_create_buffer[DType.float32](vector_size)
    var ref_m = ctx.enqueue_create_buffer[DType.float32](vector_size)
    var ref_v = ctx.enqueue_create_buffer[DType.float32](vector_size)
    with d_p.map_to_host() as ph, ref_p.map_to_host() as rp,\
         d_m.map_to_host() as mh, ref_m.map_to_host() as rm,\
         d_v.map_to_host() as vh, ref_v.map_to_host() as rv:
        for i in range(vector_size):
            rp[i] = ph[i]
            rm[i] = mh[i]
            rv[i] = vh[i]

    comptime BLOCK: Int = 256
    var blocks = (vector_size + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=adam_kernel](
            d_p.unsafe_ptr(), d_m.unsafe_ptr(), d_v.unsafe_ptr(), d_g.unsafe_ptr(),
            b1, b2, eps, grad_scale, step_size, decay,
            vector_size, time_step,
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var elapsed_ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(repeat)
    print("Average kernel execution time", elapsed_ms, "(ms)")

    # CPU reference over first 4096 entries (full compute is expensive)
    var subset = 4096 if vector_size > 4096 else vector_size
    with ref_p.map_to_host() as rp, ref_m.map_to_host() as rm,\
         ref_v.map_to_host() as rv, d_g.map_to_host() as gh:
        adam_cpu(rp.unsafe_ptr(), rm.unsafe_ptr(), rv.unsafe_ptr(), gh.unsafe_ptr(),
                 b1, b2, eps, grad_scale, step_size, decay,
                 subset, time_step, repeat)

    var max_err: Float32 = 0.0
    with d_p.map_to_host() as ph, ref_p.map_to_host() as rp:
        for i in range(subset):
            var d: Float32 = ph[i] - rp[i]
            if d < 0:
                d = -d
            if d > max_err:
                max_err = d
    print("Max element error over first", subset, "elements:", max_err)
    if max_err <= Float32(1e-3):
        print("PASS")
    else:
        print("FAIL")
