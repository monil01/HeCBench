using CUDA
using Printf
using Random

const EPS = Float32(1.0f-6)

function layernorm_kernel!(out, input, bias, gamma, beta, eps::Float32, n::Int32)
    tid = threadIdx().x
    row0 = blockIdx().x - Int32(1)
    block_threads = blockDim().x
    base = row0 * n
    shared_sum = CUDA.CuStaticSharedArray(Float32, 256)
    shared_var = CUDA.CuStaticSharedArray(Float32, 256)

    local_sum = Float32(0)
    j = tid
    while j <= n
        idx = base + j
        v = Float32(out[idx]) + Float32(input[idx]) + Float32(bias[j])
        out[idx] = v
        local_sum += v
        j += block_threads
    end
    shared_sum[tid] = local_sum
    sync_threads()

    stride = block_threads >>> Int32(1)
    while stride > 0
        if tid <= stride
            shared_sum[tid] += shared_sum[tid + stride]
        end
        sync_threads()
        stride >>>= Int32(1)
    end

    mean = shared_sum[1] / Float32(n)
    local_var = Float32(0)
    j = tid
    while j <= n
        diff = Float32(out[base + j]) - mean
        local_var += diff * diff
        j += block_threads
    end
    shared_var[tid] = local_var
    sync_threads()

    stride = block_threads >>> Int32(1)
    while stride > 0
        if tid <= stride
            shared_var[tid] += shared_var[tid + stride]
        end
        sync_threads()
        stride >>>= Int32(1)
    end

    inv_std = inv(sqrt(shared_var[1] / Float32(n) + eps))
    j = tid
    while j <= n
        idx = base + j
        v = (Float32(out[idx]) - mean) * inv_std * Float32(gamma[j]) + Float32(beta[j])
        out[idx] = v
        j += block_threads
    end
    return
end

function reference!(out::Vector{T}, input::Vector{T}, bias::Vector{T},
                    gamma::Vector{T}, beta::Vector{T}, eps::Float32,
                    m::Int, n::Int) where {T}
    for row in 0:(m - 1)
        base = row * n
        mean = Float32(0)
        for i in 1:n
            idx = base + i
            v = Float32(out[idx]) + Float32(input[idx]) + Float32(bias[i])
            out[idx] = T(v)
            mean += Float32(out[idx])
        end
        mean /= Float32(n)

        variance = Float32(0)
        for i in 1:n
            diff = Float32(out[base + i]) - mean
            variance += diff * diff
        end
        inv_std = inv(sqrt(variance / Float32(n) + eps))

        for i in 1:n
            idx = base + i
            v = (Float32(out[idx]) - mean) * inv_std * Float32(gamma[i]) + Float32(beta[i])
            out[idx] = T(v)
        end
    end
    return out
end

function run_layer(::Type{T}, label::String, version::Int, m::Int, n::Int, repeat::Int) where {T}
    rng = MersenneTwister(19937 + version + sizeof(T))
    input = T.(rand(rng, Float32, m * n))
    bias = T.(rand(rng, Float32, n))
    gamma = T.(rand(rng, Float32, n))
    beta = T.(rand(rng, Float32, n))
    output = zeros(T, m * n)
    ref = zeros(T, m * n)

    d_input = CuArray(input)
    d_bias = CuArray(bias)
    d_gamma = CuArray(gamma)
    d_beta = CuArray(beta)
    d_output = CuArray(output)

    threads = min(n, 256)
    blocks = m
    for _ in 1:100
        @cuda threads=threads blocks=blocks layernorm_kernel!(
            d_output, d_input, d_bias, d_gamma, d_beta, EPS, Int32(n))
        reference!(ref, input, bias, gamma, beta, EPS, m, n)
    end
    CUDA.synchronize()

    got = Array(d_output)
    error_bound = sizeof(T) >= 4 ? Float32(1.0f-4) : Float32(0.5)
    ok = true
    bad_i = 0
    bad_got = Float32(0)
    bad_ref = Float32(0)
    for i in eachindex(got)
        diff = abs(Float32(got[i]) - Float32(ref[i]))
        if diff > error_bound
            ok = false
            bad_i = i - 1
            bad_got = Float32(got[i])
            bad_ref = Float32(ref[i])
            break
        end
    end
    if !ok
        @printf("i=%d %f != %f (ref)\n", bad_i, bad_got, bad_ref)
    end
    println(ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks layernorm_kernel!(
            d_output, d_input, d_bias, d_gamma, d_beta, EPS, Int32(n))
    end
    CUDA.synchronize()
    dt_us = (time_ns() - t0) * 1.0e-3 / repeat
    @printf("Average execution time of AddBiasResidualLayerNorm (%d x %d): %f (us)\n",
            m, n, dt_us)
    return ok
end

function main()
    if length(ARGS) < 1 || length(ARGS) > 3
        println("Usage: julia main.jl <repeat> [m] [max_n]")
        return 1
    end
    repeat = parse(Int, ARGS[1])
    m = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4096
    max_n = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8192

    all_ok = true
    n = 256
    while n <= max_n
        println("---------------- float32 (version 1) -------------")
        all_ok &= run_layer(Float32, "float32", 1, m, n, repeat)
        println("---------------- float32 (version 2) -------------")
        all_ok &= run_layer(Float32, "float32", 2, m, n, repeat)

        println("---------------- float16 (version 1) -------------")
        all_ok &= run_layer(Float16, "float16", 1, m, n, repeat)
        println("---------------- float16 (version 2) -------------")
        all_ok &= run_layer(Float16, "float16", 2, m, n, repeat)

        println("---------------- bfloat16 (version 1) -------------")
        all_ok &= run_layer(Float32, "bfloat16", 1, m, n, repeat)
        println("---------------- bfloat16 (version 2) -------------")
        all_ok &= run_layer(Float32, "bfloat16", 2, m, n, repeat)
        println()
        n *= 2
    end
    return all_ok ? 0 : 2
end

exit(main())
