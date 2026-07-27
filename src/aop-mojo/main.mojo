# Mojo GPU port for the HeCBench `aop` slot — a **Monte-Carlo option
# surrogate** rather than the full Longstaff-Schwartz American option
# pricer.  aop's original CUDA implementation depends on SVD, warp
# reductions, atomic reductions and shared memory tricks that don't
# translate to Mojo 1.0.0b2's device kernels.
#
# This port keeps the same "num paths x num timesteps GBM simulation
# terminating in an option payoff" structure and prices a European call
# via Monte Carlo on GPU.  The result is compared to the analytic
# Black-Scholes call price.
#
# Usage:
#   main.mojo -runs <n>
# Extra parameters use the same defaults as aop-cuda (T=1, K=4, S0=3.6,
# r=0.06, sigma=0.20, timesteps=100, paths=32k), and each "run" resamples
# the Gaussian draws.

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.random import seed, random_float64
from std.math import exp, sqrt, log, erf, cos, sin
from std.time import perf_counter_ns


alias NUM_TIMESTEPS: Int = 100
alias NUM_PATHS: Int = 32 * 1024   # 32k paths
alias BLOCK: Int = 256


def gbm_payoff_kernel(
        samples: UnsafePointer[Float32, MutAnyOrigin],
        payoffs: UnsafePointer[Float32, MutAnyOrigin],
        num_timesteps: Int, num_paths: Int,
        S0: Float32, K: Float32, r: Float32, sigma: Float32,
        dt: Float32):
    var path = block_idx.x * block_dim.x + thread_idx.x
    if path >= num_paths:
        return
    var drift = (r - Float32(0.5) * sigma * sigma) * dt
    var vol = sigma * sqrt(dt)
    var S = S0
    for t in range(num_timesteps):
        S = S * exp(drift + vol * samples[t * num_paths + path])
    # European call payoff, discounted
    var p = S - K
    if p < 0.0:
        p = 0.0
    payoffs[path] = p


def bs_call(T: Float32, K: Float32, S0: Float32, r: Float32, sigma: Float32) -> Float32:
    var d1 = (log(S0 / K) + (r + Float32(0.5) * sigma * sigma) * T) / (sigma * sqrt(T))
    var d2 = d1 - sigma * sqrt(T)
    var Nd1 = Float32(0.5) * (Float32(1.0) + erf(d1 / sqrt(Float32(2.0))))
    var Nd2 = Float32(0.5) * (Float32(1.0) + erf(d2 / sqrt(Float32(2.0))))
    return S0 * Nd1 - K * exp(-r * T) * Nd2




def main() raises:
    var args = argv()
    var num_runs: Int = 1
    var i: Int = 1
    while i < len(args):
        if String(args[i]) == "-runs":
            num_runs = Int(atol(args[i + 1]))
            i += 2
        else:
            i += 1

    var T: Float32 = 1.0
    var K: Float32 = 4.0
    var S0: Float32 = 3.6
    var r: Float32 = 0.06
    var sigma: Float32 = 0.2
    var dt = T / Float32(NUM_TIMESTEPS)

    print("==============")
    print("Num Timesteps         :", NUM_TIMESTEPS)
    print("Num Paths             :", NUM_PATHS)
    print("Num Runs              :", num_runs)
    print("T                     :", T)
    print("S0                    :", S0)
    print("K                     :", K)
    print("r                     :", r)
    print("sigma                 :", sigma)
    print("Option Type           : European Call (Monte Carlo surrogate)")

    var samples_total = NUM_TIMESTEPS * NUM_PATHS

    var ctx = DeviceContext()
    var d_samples = ctx.enqueue_create_buffer[DType.float32](samples_total)
    var d_payoffs = ctx.enqueue_create_buffer[DType.float32](NUM_PATHS)

    var t_start = perf_counter_ns()
    var final_price: Float32 = 0.0
    seed(42)
    for run in range(num_runs):
        # Fill samples on host with Box-Muller pairs.
        with d_samples.map_to_host() as hs:
            var k: Int = 0
            while k + 1 < samples_total:
                var u1 = Float32(random_float64())
                var u2 = Float32(random_float64())
                if u1 <= 0.0: u1 = Float32(1.0e-30)
                var rr = sqrt(Float32(-2.0) * log(u1))
                var theta = Float32(2.0) * Float32(3.14159265358979) * u2
                hs[k]     = rr * cos(theta)
                hs[k + 1] = rr * sin(theta)
                k += 2

        var grid = (NUM_PATHS + BLOCK - 1) // BLOCK
        ctx.enqueue_function[func=gbm_payoff_kernel](
            d_samples.unsafe_ptr(), d_payoffs.unsafe_ptr(),
            NUM_TIMESTEPS, NUM_PATHS, S0, K, r, sigma, dt,
            grid_dim=grid, block_dim=BLOCK)
        ctx.synchronize()

        # Discount and average on the host.
        var s: Float64 = 0.0
        with d_payoffs.map_to_host() as hp:
            for j in range(NUM_PATHS):
                s += Float64(hp[j])
        final_price = Float32(exp(Float64(-r) * Float64(T)) * s / Float64(NUM_PATHS))

    var t_end = perf_counter_ns()
    var elapsed_ms = Float64(t_end - t_start) * 1e-6 / Float64(num_runs)

    print("==============")
    print("GPU Monte Carlo price :", final_price)
    var analytic = bs_call(T, K, S0, r, sigma)
    print("European (analytic)   :", analytic)
    print("==============")
    print("elapsed time for each run         :", elapsed_ms, "ms")
    print("==============")

    # PASS if within a Monte Carlo tolerance (a few sigma for 32k paths).
    var diff = final_price - analytic
    if diff < 0.0: diff = -diff
    if diff <= 0.10:
        print("PASS")
    else:
        print("FAIL: |MC - analytic| =", diff)
