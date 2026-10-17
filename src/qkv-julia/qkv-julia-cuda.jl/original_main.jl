using CUDA
using LinearAlgebra
using Printf
using Random

function make_random_float(rng, n::Int)
    return rand(rng, Float32, n) .* 2.0f0 .- 1.0f0
end

function matmul_forward_cpu(inp, weight, bias, bt::Int, c::Int, oc::Int)
    out = Matrix{Float32}(undef, bt, oc)
    @inbounds for row in 1:bt
        for col in 1:oc
            val = bias[col]
            for k in 1:c
                val += inp[row, k] * weight[col, k]
            end
            out[row, col] = val
        end
    end
    return out
end

function matmul_kernel1!(out, inp, weight, bias, bt::Int32, c::Int32, ocn::Int32)
    row0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    col0 = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if row0 < bt && col0 < ocn
        val = @inbounds bias[col0 + Int32(1)]
        k = Int32(0)
        while k < c
            val += @inbounds inp[row0 + k * bt + Int32(1)] * weight[col0 + k * ocn + Int32(1)]
            k += Int32(1)
        end
        @inbounds out[row0 + col0 * bt + Int32(1)] = val
    end
    return
end

function add_bias_kernel!(out, bias, total::Int32, ocn::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    step = blockDim().x * gridDim().x
    i = idx0
    while i < total
        col0 = i ÷ (total ÷ ocn)
        @inbounds out[i + Int32(1)] += bias[col0 + Int32(1)]
        i += step
    end
    return
end

function matmul_forward_gpu!(kernel_num::Int, d_out, d_inp, d_weight, d_bias, bt::Int, c::Int, oc::Int)
    if kernel_num == 1 || kernel_num == 4
        threads = (16, 16)
        blocks = (cld(bt, 16), cld(oc, 16))
        @cuda threads=threads blocks=blocks matmul_kernel1!(d_out, d_inp, d_weight, d_bias, Int32(bt), Int32(c), Int32(oc))
    elseif kernel_num == 2 || kernel_num == 3
        mul!(d_out, d_inp, transpose(d_weight))
        d_out .+= reshape(d_bias, 1, :)
    else
        println("Invalid kernel number")
        exit(1)
    end
    CUDA.synchronize()
end

function validate_result(d_out, cpu_reference; tolerance=1.0f-1)
    out_gpu = Array(d_out)
    nfaults = 0
    epsilon = eps(Float32)
    @inbounds for i in eachindex(cpu_reference)
        t_eff = tolerance + abs(cpu_reference[i]) * epsilon
        if abs(cpu_reference[i] - out_gpu[i]) > t_eff
            idx = Tuple(CartesianIndices(cpu_reference)[i])
            @printf("Mismatch of out at %zu: CPU_ref: %f vs GPU: %f\n",
                    i - 1, cpu_reference[idx...], out_gpu[idx...])
            nfaults += 1
            nfaults >= 10 && exit(1)
        end
    end
end

function benchmark_kernel(f, repeats)
    elapsed_ms = 0.0
    for _ in 1:repeats
        CUDA.synchronize()
        start = time_ns()
        f()
        CUDA.synchronize()
        elapsed_ms += (time_ns() - start) * 1.0e-6
    end
    return elapsed_ms / repeats
end

function parse_args(args)
    kernel_num = length(args) >= 1 ? parse(Int, args[1]) : 1
    b = length(args) >= 2 ? parse(Int, args[2]) : 4
    t = length(args) >= 3 ? parse(Int, args[3]) : 1024
    c = length(args) >= 4 ? parse(Int, args[4]) : 768
    oc = length(args) >= 5 ? parse(Int, args[5]) : 768 * 3
    repeats = length(args) >= 6 ? parse(Int, args[6]) : 100
    return kernel_num, b, t, c, oc, repeats
end

function main()
    rng = MersenneTwister(0)
    kernel_num, b, t, c, oc = parse_args(ARGS)[1:5]
    repeats = parse_args(ARGS)[6]
    CUDA.allowscalar(false)
    dev = CUDA.device()
    println("Device 0: ", CUDA.name(dev))
    println("enable_tf32: 0")
    bt = b * t
    inp = reshape(make_random_float(rng, bt * c), bt, c)
    weight = reshape(make_random_float(rng, oc * c), oc, c)
    bias = make_random_float(rng, oc)
    d_inp = CuArray(inp)
    d_weight = CuArray(weight)
    d_bias = CuArray(bias)
    d_out = CUDA.zeros(Float32, bt, oc)
    println("Using kernel $kernel_num")
    out = matmul_forward_cpu(inp, weight, bias, bt, c, oc)
    matmul_forward_gpu!(kernel_num, d_out, d_inp, d_weight, d_bias, bt, c, oc)
    validate_result(d_out, out)
    println("All results match. Starting benchmarks.")
    println()
    elapsed_time = benchmark_kernel(repeats) do
        matmul_forward_gpu!(kernel_num, d_out, d_inp, d_weight, d_bias, bt, c, oc)
    end
    tflops = Float32(b) * Float32(t) * Float32(c) * Float32(oc) * 2.0f0 / Float32(elapsed_time) * 1.0f3 / 1.0f12
    @printf("time %.4f ms | tflops %.2f\n", elapsed_time, tflops)
end

main()
