using CUDA
using Printf
using Random

function moe_sum_kernel!(out, input, hidden_size::Int32, topk::Int32)
    token0 = blockIdx().x - Int32(1)
    idx0 = threadIdx().x - Int32(1)
    output_base = token0 * hidden_size
    input_base = output_base * topk

    while idx0 < hidden_size
        x = 0.0f0
        for k in Int32(0):topk-Int32(1)
            @inbounds x += input[input_base + k * hidden_size + idx0 + Int32(1)]
        end
        @inbounds out[output_base + idx0 + Int32(1)] = x
        idx0 += blockDim().x
    end
    return
end

function moe_sum_vec4_kernel!(out, input, hidden_size::Int32, topk::Int32)
    token0 = blockIdx().x - Int32(1)
    lane0 = (threadIdx().x - Int32(1)) * Int32(4)
    stride = blockDim().x * Int32(4)
    output_base = token0 * hidden_size
    input_base = output_base * topk

    while lane0 < hidden_size
        for c in Int32(0):Int32(3)
            idx0 = lane0 + c
            if idx0 < hidden_size
                x = 0.0f0
                for k in Int32(0):topk-Int32(1)
                    @inbounds x += input[input_base + k * hidden_size + idx0 + Int32(1)]
                end
                @inbounds out[output_base + idx0 + Int32(1)] = x
            end
        end
        lane0 += stride
    end
    return
end

function moe_sum_ref!(topk::Int, out, input, num_tokens::Int, hidden_size::Int)
    for token in 0:num_tokens-1
        output_base = token * hidden_size
        input_base = output_base * topk
        for idx in 0:hidden_size-1
            x = 0.0f0
            for k in 0:topk-1
                x += input[input_base + k * hidden_size + idx + 1]
            end
            out[output_base + idx + 1] = x
        end
    end
end

function time_kernel!(kernel, out, input, hidden_size::Int, num_tokens::Int, topk::Int, repeat::Int)
    threads = min(hidden_size, 1024)
    for _ in 1:100
        @cuda threads=threads blocks=num_tokens kernel(out, input, Int32(hidden_size), Int32(topk))
    end
    CUDA.synchronize()

    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=num_tokens kernel(out, input, Int32(hidden_size), Int32(topk))
    end
    CUDA.synchronize()
    return time_ns() - t0
end

function run_case(num_tokens::Int, hidden_size::Int, repeat::Int)
    if hidden_size % 4 != 0
        println("Hidden size is a multiple of four")
        return 1
    end

    output_len = num_tokens * hidden_size
    d_output = CUDA.zeros(Float32, output_len)
    output = Vector{Float32}(undef, output_len)
    output_vec4 = Vector{Float32}(undef, output_len)
    r_output = Vector{Float32}(undef, output_len)

    for topk in 2:4
        input_len = output_len * topk
        rng = MersenneTwister(topk)
        input = rand(rng, Float32, input_len) .* 2.0f0 .- 1.0f0
        moe_sum_ref!(topk, r_output, input, num_tokens, hidden_size)
        d_input = CuArray(input)

        ns = time_kernel!(moe_sum_kernel!, d_output, d_input, hidden_size, num_tokens, topk, repeat)
        @printf("Average execution time of kernel (TopK = %d): %f (us)\n",
                topk, ns * 1.0e-3 / repeat)
        copyto!(output, d_output)
        ok = all(abs.(r_output .- output) .<= 1.0f-4)
        println(ok ? "PASS" : "FAIL")

        io_bytes = Float32(repeat) * Float32((input_len + output_len) * sizeof(Float32))
        bw = Float64(io_bytes) / Float64(ns)
        @printf("Kernel bandwidth: %f GB/s \n", bw)

        ns_vec4 = time_kernel!(moe_sum_vec4_kernel!, d_output, d_input, hidden_size, num_tokens, topk, repeat)
        @printf("Average execution time of vec4 kernel (TopK = %d): %f (us)\n",
                topk, ns_vec4 * 1.0e-3 / repeat)
        copyto!(output_vec4, d_output)
        println(output == output_vec4 ? "PASS" : "FAIL")

        bw_vec4 = Float64(io_bytes) / Float64(ns_vec4)
        pct = bw == 0.0 ? 0.0 : 100.0 * (bw_vec4 - bw) / bw
        @printf("Kernel(vec4) bandwidth: %f GB/s (%f%%)\n", bw_vec4, pct)
    end
    return 0
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of tokens> <hidden size> <repeat>")
        return 1
    end
    return run_case(parse(Int, args[1]), parse(Int, args[2]), parse(Int, args[3]))
end

exit(main(ARGS))
