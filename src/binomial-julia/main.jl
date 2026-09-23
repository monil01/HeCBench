using CUDA
using Printf
using StaticArrays

const NUM_STEPS = Int32(2048)
const MAX_OPTIONS = 1024
const NUM_ITERATIONS = 1000
const THREADBLOCK_SIZE = 128
const ELEMS_PER_THREAD = Int32(NUM_STEPS ÷ THREADBLOCK_SIZE)
const RAND_MAX_C = Float32(2147483647)

function rand_data(low::Float32, high::Float32)
    r = ccall(:rand, Cint, ())
    t = Float32(r) / RAND_MAX_C
    return (Float32(1) - t) * low + t * high
end

function cnd(d::Float32)
    a1 = Float32(0.31938153)
    a2 = Float32(-0.356563782)
    a3 = Float32(1.781477937)
    a4 = Float32(-1.821255978)
    a5 = Float32(1.330274429)
    rsqrt2pi = Float32(0.39894228040143267794)

    k = Float32(1) / (Float32(1) + Float32(0.2316419) * abs(d))
    value = rsqrt2pi * exp(Float32(-0.5) * d * d) *
        (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))
    return d > 0 ? Float32(1) - value : value
end

function black_scholes_call(S::Float32, X::Float32, T::Float32, R::Float32, V::Float32)
    sqrtT = sqrt(T)
    d1 = (log(S / X) + (R + Float32(0.5) * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    return S * cnd(d1) - X * exp(-R * T) * cnd(d2)
end

function expiry_call_value_cpu(S::Float32, X::Float32, vDt::Float32, i::Int)
    d = S * exp(vDt * Float32(2 * i - Int(NUM_STEPS))) - X
    return d > 0 ? d : Float32(0)
end

function binomial_options_cpu(S::Float32, X::Float32, T::Float32, R::Float32, V::Float32)
    call = Vector{Float32}(undef, Int(NUM_STEPS) + 1)
    dt = T / Float32(NUM_STEPS)
    vDt = V * sqrt(dt)
    rDt = R * dt
    if_ = exp(rDt)
    df = exp(-rDt)
    u = exp(vDt)
    d = exp(-vDt)
    pu = (if_ - d) / (u - d)
    pd = Float32(1) - pu
    puByDf = pu * df
    pdByDf = pd * df

    @inbounds for i in 0:Int(NUM_STEPS)
        call[i + 1] = expiry_call_value_cpu(S, X, vDt, i)
    end

    @inbounds for i in Int(NUM_STEPS):-1:1
        for j in 1:i
            call[j] = puByDf * call[j + 1] + pdByDf * call[j]
        end
    end
    return call[1]
end

function binomial_kernel!(S, X, vDt, puByDf, pdByDf, out)
    tid = threadIdx().x
    tid0 = tid - Int32(1)
    bid = blockIdx().x

    exchange = CUDA.@cuStaticSharedMem(Float32, 129)
    local_call = MVector{17, Float32}(undef)

    s = S[bid]
    x = X[bid]
    vdt = vDt[bid]
    pu = puByDf[bid]
    pd = pdByDf[bid]

    @inbounds for j in Int32(0):(ELEMS_PER_THREAD - Int32(1))
        step = tid0 * ELEMS_PER_THREAD + j
        value = s * exp(vdt * (Float32(2) * Float32(step) - Float32(NUM_STEPS))) - x
        local_call[Int(j) + 1] = value > 0 ? value : Float32(0)
    end

    if tid == Int32(1)
        value = s * exp(vdt * Float32(NUM_STEPS)) - x
        exchange[129] = value > 0 ? value : Float32(0)
    end

    final_it = max(Int32(0), tid0 * ELEMS_PER_THREAD - Int32(1))

    @inbounds for i in NUM_STEPS:-Int32(1):Int32(1)
        exchange[tid] = local_call[1]
        sync_threads()
        local_call[17] = exchange[tid + Int32(1)]
        sync_threads()

        if i > final_it
            for j in 1:Int(ELEMS_PER_THREAD)
                local_call[j] = pu * local_call[j + 1] + pd * local_call[j]
            end
        end
    end

    if tid == Int32(1)
        out[bid] = local_call[1]
    end
    return
end

function binomial_options_gpu(S, X, T, R, V)
    host_vDt = Vector{Float32}(undef, MAX_OPTIONS)
    host_puByDf = Vector{Float32}(undef, MAX_OPTIONS)
    host_pdByDf = Vector{Float32}(undef, MAX_OPTIONS)

    @inbounds for i in 1:MAX_OPTIONS
        dt = T[i] / Float32(NUM_STEPS)
        vDt = V[i] * sqrt(dt)
        rDt = R[i] * dt
        if_ = exp(rDt)
        df = exp(-rDt)
        u = exp(vDt)
        d = exp(-vDt)
        pu = (if_ - d) / (u - d)
        pd = Float32(1) - pu
        host_vDt[i] = vDt
        host_puByDf[i] = pu * df
        host_pdByDf[i] = pd * df
    end

    d_S = CuArray(S)
    d_X = CuArray(X)
    d_vDt = CuArray(host_vDt)
    d_puByDf = CuArray(host_puByDf)
    d_pdByDf = CuArray(host_pdByDf)
    d_call = CUDA.zeros(Float32, MAX_OPTIONS)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:NUM_ITERATIONS
        @cuda threads=THREADBLOCK_SIZE blocks=MAX_OPTIONS binomial_kernel!(
            d_S, d_X, d_vDt, d_puByDf, d_pdByDf, d_call)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1e-3 / NUM_ITERATIONS
    @printf("Average kernel execution time : %f (us)\n", elapsed_us)

    return Array(d_call)
end

function main()
    println("[/home/imo/HeCBench/src/binomial-cuda/main] - Starting...")
    println("Generating input data...")

    ccall(:srand, Cvoid, (Cuint,), UInt32(123))
    S = Vector{Float32}(undef, MAX_OPTIONS)
    X = Vector{Float32}(undef, MAX_OPTIONS)
    T = Vector{Float32}(undef, MAX_OPTIONS)
    R = fill(Float32(0.06), MAX_OPTIONS)
    V = fill(Float32(0.10), MAX_OPTIONS)
    call_bs = Vector{Float32}(undef, MAX_OPTIONS)
    call_cpu = Vector{Float32}(undef, MAX_OPTIONS)

    @inbounds for i in 1:MAX_OPTIONS
        S[i] = rand_data(Float32(5), Float32(30))
        X[i] = rand_data(Float32(1), Float32(100))
        T[i] = rand_data(Float32(0.25), Float32(10))
        call_bs[i] = black_scholes_call(S[i], X[i], T[i], R[i], V[i])
    end

    println("Running GPU binomial tree...")
    start = time()
    call_gpu = binomial_options_gpu(S, X, T, R, V)
    gpu_time = Float32(time() - start)

    @printf("Options count            : %i     \n", MAX_OPTIONS)
    @printf("Time steps               : %i     \n", NUM_STEPS)
    @printf("Total binomialOptionsGPU() time: %f msec\n", gpu_time * 1000)
    @printf("Options per second       : %f     \n", MAX_OPTIONS / gpu_time)

    println("Running CPU binomial tree...")
    @inbounds for i in 1:MAX_OPTIONS
        call_cpu[i] = binomial_options_cpu(S[i], X[i], T[i], R[i], V[i])
    end

    println("Comparing the results...")
    println("GPU binomial vs. Black-Scholes")
    sum_delta = sum(abs.(call_bs .- call_gpu))
    sum_ref = sum(abs.(call_bs))
    if sum_ref > Float32(1f-5)
        @printf("L1 norm: %E\n", Float64(sum_delta / sum_ref))
    else
        @printf("Avg. diff: %E\n", Float64(sum_delta / Float32(MAX_OPTIONS)))
    end

    println("CPU binomial vs. Black-Scholes")
    sum_delta = sum(abs.(call_bs .- call_cpu))
    sum_ref = sum(abs.(call_bs))
    if sum_ref > Float32(1f-5)
        @printf("L1 norm: %E\n", Float64(sum_delta / sum_ref))
    else
        @printf("Avg. diff: %E\n", Float64(sum_delta / Float32(MAX_OPTIONS)))
    end

    println("CPU binomial vs. GPU binomial")
    sum_delta = sum(abs.(call_gpu .- call_cpu))
    sum_ref = sum(call_cpu)
    @printf("Avg. diff: %E\n", Float64(sum_delta / Float32(MAX_OPTIONS)))
    error_val = sum_delta / sum_ref
    @printf("L1 norm: %E\n", Float64(error_val))

    if error_val > Float32(5e-4)
        println("Test failed!")
        exit(1)
    end

    println("Test passed")
end

main()
