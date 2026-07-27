# Mojo GPU port of the `black-scholes` HeCBench benchmark (simplified).
#
# Analytic Black-Scholes call/put pricing, one thread per option.
# Uses exp, sqrt, log from std.math on the device. Mirrors the Julia
# port's 37-config cycling for reproducible inputs. Verified against
# a host reference on a subset.
#
# Usage: main.mojo <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.math import exp, sqrt, log
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


comptime NUM_OPTS: Int = 1000000  # 1M options


def erf_approx(x: Float32) -> Float32:
    # Abramowitz & Stegun 7.1.26 approximation (fp32).
    var a1: Float32 = 0.254829592
    var a2: Float32 = -0.284496736
    var a3: Float32 = 1.421413741
    var a4: Float32 = -1.453152027
    var a5: Float32 = 1.061405429
    var pp: Float32 = 0.3275911
    var sign: Float32 = Float32(-1.0) if x < Float32(0.0) else Float32(1.0)
    var ax: Float32 = -x if x < Float32(0.0) else x
    var t: Float32 = Float32(1.0) / (Float32(1.0) + pp * ax)
    var y: Float32 = Float32(1.0) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax)
    return sign * y


def bs_kernel(
        ty:    UnsafePointer[Int32, MutAnyOrigin],
        spot:  UnsafePointer[Float32, MutAnyOrigin],
        strike: UnsafePointer[Float32, MutAnyOrigin],
        divp:   UnsafePointer[Float32, MutAnyOrigin],
        risk:   UnsafePointer[Float32, MutAnyOrigin],
        T:      UnsafePointer[Float32, MutAnyOrigin],
        vol:    UnsafePointer[Float32, MutAnyOrigin],
        out_p:  UnsafePointer[Float32, MutAnyOrigin],
        n:      Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n): return
    var S = spot[i]; var K = strike[i]; var q = divp[i]; var r = risk[i]
    var Tt = T[i]; var v = vol[i]
    var sqrtT = sqrt(Tt)
    var vs = v * sqrtT
    var d1 = (log(S / K) + (r - q + Float32(0.5) * v * v) * Tt) / vs
    var d2 = d1 - vs
    var inv_sqrt2: Float32 = 0.70710678
    var Nd1 = Float32(0.5) * (Float32(1.0) + erf_approx(d1 * inv_sqrt2))
    var Nd2 = Float32(0.5) * (Float32(1.0) + erf_approx(d2 * inv_sqrt2))
    var eqT = exp(-q * Tt)
    var erT = exp(-r * Tt)
    var call = S * eqT * Nd1 - K * erT * Nd2
    var put  = call - S * eqT + K * erT
    out_p[i] = call if ty[i] == Int32(1) else put


def erf_host(x: Float32) -> Float32:
    var a1: Float32 = 0.254829592
    var a2: Float32 = -0.284496736
    var a3: Float32 = 1.421413741
    var a4: Float32 = -1.453152027
    var a5: Float32 = 1.061405429
    var pp: Float32 = 0.3275911
    var sign: Float32 = Float32(-1.0) if x < Float32(0.0) else Float32(1.0)
    var ax: Float32 = -x if x < Float32(0.0) else x
    var t: Float32 = Float32(1.0) / (Float32(1.0) + pp * ax)
    var y: Float32 = Float32(1.0) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax)
    return sign * y


def bs_host(ty: Int32, S: Float32, K: Float32, q: Float32, r: Float32,
            Tt: Float32, v: Float32) -> Float32:
    var sqrtT = sqrt(Tt)
    var vs = v * sqrtT
    var d1 = (log(S / K) + (r - q + Float32(0.5) * v * v) * Tt) / vs
    var d2 = d1 - vs
    var inv_sqrt2: Float32 = 0.70710678
    var Nd1 = Float32(0.5) * (Float32(1.0) + erf_host(d1 * inv_sqrt2))
    var Nd2 = Float32(0.5) * (Float32(1.0) + erf_host(d2 * inv_sqrt2))
    var eqT = exp(-q * Tt)
    var erT = exp(-r * Tt)
    var call = S * eqT * Nd1 - K * erT * Nd2
    var put  = call - S * eqT + K * erT
    return call if ty == Int32(1) else put


def main() raises:
    var args = argv()
    var repeat = Int(atol(args[1])) if len(args) > 1 else 10

    # Configuration cycle: (ty, spot, strike, div, risk, T, vol)
    # 37 configs from black-scholes-julia — flattened to 7 parallel lists.
    var ty_cfg: List[Int32] = [
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0]
    var S_cfg: List[Float32] = [
        40.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
        100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
        100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
        100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
    var K_cfg: List[Float32] = [
        42.00, 90.00, 100.00, 110.00, 90.00, 100.00, 110.00, 90.00, 100.00, 110.00,
        90.00, 100.00, 110.00, 90.00, 100.00, 110.00, 90.00, 100.00, 110.00,
        90.00, 100.00, 110.00, 90.00, 100.00, 110.00, 90.00, 100.00, 110.00,
        90.00, 100.00, 110.00, 90.00, 100.00, 110.00, 90.00, 100.00, 110.00]
    var q_cfg: List[Float32] = [
        0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    var r_cfg: List[Float32] = [
        0.04, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    var T_cfg: List[Float32] = [
        0.75, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    var v_cfg: List[Float32] = [
        0.35, 0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35,
        0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35,
        0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35,
        0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35]
    var NC = len(ty_cfg)

    print("Number of options:", NUM_OPTS)

    var ctx = DeviceContext()
    var d_ty   = ctx.enqueue_create_buffer[DType.int32](NUM_OPTS)
    var d_S    = ctx.enqueue_create_buffer[DType.float32](NUM_OPTS)
    var d_K    = ctx.enqueue_create_buffer[DType.float32](NUM_OPTS)
    var d_q    = ctx.enqueue_create_buffer[DType.float32](NUM_OPTS)
    var d_r    = ctx.enqueue_create_buffer[DType.float32](NUM_OPTS)
    var d_T    = ctx.enqueue_create_buffer[DType.float32](NUM_OPTS)
    var d_v    = ctx.enqueue_create_buffer[DType.float32](NUM_OPTS)
    var d_out  = ctx.enqueue_create_buffer[DType.float32](NUM_OPTS)

    with d_ty.map_to_host() as th, d_S.map_to_host() as sh,\
         d_K.map_to_host() as kh, d_q.map_to_host() as qh,\
         d_r.map_to_host() as rh, d_T.map_to_host() as tth,\
         d_v.map_to_host() as vh:
        for i in range(NUM_OPTS):
            var c = i % NC
            th[i]  = ty_cfg[c]
            sh[i]  = S_cfg[c]
            kh[i]  = K_cfg[c]
            qh[i]  = q_cfg[c]
            rh[i]  = r_cfg[c]
            tth[i] = T_cfg[c]
            vh[i]  = v_cfg[c]

    comptime BLOCK: Int = 256
    var blocks = (NUM_OPTS + BLOCK - 1) // BLOCK

    # Warmup
    ctx.enqueue_function[func=bs_kernel](
        d_ty.unsafe_ptr(), d_S.unsafe_ptr(), d_K.unsafe_ptr(),
        d_q.unsafe_ptr(), d_r.unsafe_ptr(), d_T.unsafe_ptr(),
        d_v.unsafe_ptr(), d_out.unsafe_ptr(),
        Int32(NUM_OPTS), grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=bs_kernel](
            d_ty.unsafe_ptr(), d_S.unsafe_ptr(), d_K.unsafe_ptr(),
            d_q.unsafe_ptr(), d_r.unsafe_ptr(), d_T.unsafe_ptr(),
            d_v.unsafe_ptr(), d_out.unsafe_ptr(),
            Int32(NUM_OPTS), grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(repeat)
    print("Run on GPU")
    print("Average kernel execution time on GPU:", ms, "(ms)")

    # Verify on subset
    var subset = 10000
    var max_err: Float32 = 0.0
    with d_out.map_to_host() as oh:
        for i in range(subset):
            var c = i % NC
            var rv = bs_host(ty_cfg[c], S_cfg[c], K_cfg[c],
                             q_cfg[c], r_cfg[c], T_cfg[c], v_cfg[c])
            var d = oh[i] - rv
            if d < 0: d = -d
            if d > max_err: max_err = d
    print("Max abs error (subset", subset, "):", max_err)
    if max_err <= Float32(1e-3):
        print("PASS")
    else:
        print("FAIL")
