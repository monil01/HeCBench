using CUDA
using Printf

# Julia port of black-scholes-cuda benchmark.
# Prices 50M European options via analytic Black-Scholes; uses 37-config cycle.

const CONFIGS = [
    (1,  40.00f0,  42.00f0, 0.08f0, 0.04f0, 0.75f0, 0.35f0),
    (1, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.10f0, 0.15f0),
    (1, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.10f0, 0.15f0),
    (1, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.10f0, 0.15f0),
    (1, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.10f0, 0.25f0),
    (1, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.10f0, 0.25f0),
    (1, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.10f0, 0.25f0),
    (1, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.10f0, 0.35f0),
    (1, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.10f0, 0.35f0),
    (1, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.10f0, 0.35f0),
    (1, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.50f0, 0.15f0),
    (1, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.50f0, 0.15f0),
    (1, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.50f0, 0.15f0),
    (1, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.50f0, 0.25f0),
    (1, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.50f0, 0.25f0),
    (1, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.50f0, 0.25f0),
    (1, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.50f0, 0.35f0),
    (1, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.50f0, 0.35f0),
    (1, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.50f0, 0.35f0),
    (0, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.10f0, 0.15f0),
    (0, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.10f0, 0.15f0),
    (0, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.10f0, 0.15f0),
    (0, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.10f0, 0.25f0),
    (0, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.10f0, 0.25f0),
    (0, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.10f0, 0.25f0),
    (0, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.10f0, 0.35f0),
    (0, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.10f0, 0.35f0),
    (0, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.10f0, 0.35f0),
    (0, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.50f0, 0.15f0),
    (0, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.50f0, 0.15f0),
    (0, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.50f0, 0.15f0),
    (0, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.50f0, 0.25f0),
    (0, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.50f0, 0.25f0),
    (0, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.50f0, 0.25f0),
    (0, 100.00f0,  90.00f0, 0.10f0, 0.10f0, 0.50f0, 0.35f0),
    (0, 100.00f0, 100.00f0, 0.10f0, 0.10f0, 0.50f0, 0.35f0),
    (0, 100.00f0, 110.00f0, 0.10f0, 0.10f0, 0.50f0, 0.35f0),
]

# Abramowitz & Stegun 7.1.26 erf approximation (single precision).
@inline function erf_approx(x::Float32)
    a1 =  0.254829592f0
    a2 = -0.284496736f0
    a3 =  1.421413741f0
    a4 = -1.453152027f0
    a5 =  1.061405429f0
    p  =  0.3275911f0
    sign = x < 0.0f0 ? -1.0f0 : 1.0f0
    ax = abs(x)
    t = 1.0f0 / (1.0f0 + p * ax)
    y = 1.0f0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax*ax)
    return sign * y
end

function bs_kernel!(ty, spot, strike, div, risk, T, vol, out, N::Int32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    @inbounds if i <= N
        S = spot[i]; K = strike[i]; q = div[i]; r = risk[i]
        Tt = T[i]; v = vol[i]
        sqrtT = sqrt(Tt)
        vs = v * sqrtT
        d1 = (log(S / K) + (r - q + 0.5f0 * v * v) * Tt) / vs
        d2 = d1 - vs
        inv_sqrt2 = 0.70710678f0
        Nd1 = 0.5f0 * (1.0f0 + erf_approx(d1 * inv_sqrt2))
        Nd2 = 0.5f0 * (1.0f0 + erf_approx(d2 * inv_sqrt2))
        eqT = exp(-q * Tt)
        erT = exp(-r * Tt)
        call = S * eqT * Nd1 - K * erT * Nd2
        put  = call - S * eqT + K * erT
        out[i] = ty[i] == Int32(1) ? call : put
    end
    return
end

function bs_cpu_ref(cfg_idx, cfgs, N)
    inv_sqrt2 = Float32(1.0 / sqrt(2.0))
    ref = Vector{Float32}(undef, N)
    for i in 1:N
        c = cfgs[cfg_idx[i]]
        ty, S, K, q, r, Tt, v = c
        sqrtT = sqrt(Tt)
        vs = v * sqrtT
        d1 = (log(S / K) + (r - q + 0.5f0 * v * v) * Tt) / vs
        d2 = d1 - vs
        Nd1 = 0.5f0 * (1.0f0 + erf_approx(d1 * inv_sqrt2))
        Nd2 = 0.5f0 * (1.0f0 + erf_approx(d2 * inv_sqrt2))
        eqT = exp(-q * Tt)
        erT = exp(-r * Tt)
        call = S * eqT * Nd1 - K * erT * Nd2
        put  = call - S * eqT + K * erT
        ref[i] = ty == 1 ? call : put
    end
    return ref
end

function main()
    if length(ARGS) < 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat_n = parse(Int, ARGS[1])
    N = 50_000_000

    println("Number of options: $N\n")

    NC = length(CONFIGS)
    ty_h     = Vector{Int32}(undef, N)
    spot_h   = Vector{Float32}(undef, N)
    strike_h = Vector{Float32}(undef, N)
    div_h    = Vector{Float32}(undef, N)
    risk_h   = Vector{Float32}(undef, N)
    T_h      = Vector{Float32}(undef, N)
    vol_h    = Vector{Float32}(undef, N)
    for i in 1:N
        c = CONFIGS[((i-1) % NC) + 1]
        ty_h[i]     = Int32(c[1])
        spot_h[i]   = c[2]
        strike_h[i] = c[3]
        div_h[i]    = c[4]
        risk_h[i]   = c[5]
        T_h[i]      = c[6]
        vol_h[i]    = c[7]
    end

    d_ty    = CuArray(ty_h)
    d_spot  = CuArray(spot_h)
    d_strk  = CuArray(strike_h)
    d_div   = CuArray(div_h)
    d_risk  = CuArray(risk_h)
    d_T     = CuArray(T_h)
    d_vol   = CuArray(vol_h)
    d_out   = CUDA.zeros(Float32, N)

    threads = 256
    blocks  = cld(N, threads)

    # Warmup
    @cuda threads=threads blocks=blocks bs_kernel!(d_ty, d_spot, d_strk, d_div,
                                                    d_risk, d_T, d_vol, d_out, Int32(N))
    CUDA.synchronize()

    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks bs_kernel!(d_ty, d_spot, d_strk, d_div,
                                                        d_risk, d_T, d_vol, d_out, Int32(N))
    end
    CUDA.synchronize()
    ktime_ms = (time_ns() - t0) * 1e-6 / repeat_n
    println("Run on GPU")
    @printf("Average kernel execution time on GPU: %f (ms)\n", ktime_ms)
    @printf("Processing time using GPU %f (ms)\n", ktime_ms)

    out_h = Array(d_out)
    tot = sum(out_h)
    mid = out_h[N ÷ 2 + 1]
    @printf("Summation of output prices on GPU: %f\n", tot)
    @printf("Output price at index %d on GPU: %f\n\n", N ÷ 2, mid)

    subset = 100_000
    cfg_idx = [((i-1) % NC) + 1 for i in 1:subset]
    ref = bs_cpu_ref(cfg_idx, CONFIGS, subset)
    maxerr = 0.0f0
    for i in 1:subset
        e = abs(out_h[i] - ref[i])
        if e > maxerr
            maxerr = e
        end
    end
    @printf("Max abs error (subset %d): %g\n", subset, maxerr)
    println(maxerr < 1f-3 ? "PASS" : "FAIL")
    return 0
end

main()
