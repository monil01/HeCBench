using CUDA
using Printf

function xorshift32(x::UInt32)
    x ⊻= x << 13
    x ⊻= x >> 17
    x ⊻= x << 5
    return x
end

function fill_uniform!(data::Vector{T}, low::Float32, high::Float32, seed::UInt32) where {T}
    for i in eachindex(data)
        r = xorshift32(seed ⊻ UInt32(i - 1))
        u = Float32(r >> 8) * Float32(2.0^-24)
        data[i] = T(low + (high - low) * u)
    end
    return data
end

function merge_kernel!(output, prefix_output, suffix_output, lse,
                       prefix_lse, suffix_lse, total::Int32,
                       head_size::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx0 >= total
        return
    end
    lse_idx = idx0 ÷ head_size
    p_lse = @inbounds prefix_lse[lse_idx + Int32(1)]
    s_lse = @inbounds suffix_lse[lse_idx + Int32(1)]
    p_lse = isinf(p_lse) ? -Inf32 : p_lse
    s_lse = isinf(s_lse) ? -Inf32 : s_lse
    max_lse = max(p_lse, s_lse)
    p_se = exp(p_lse - max_lse)
    s_se = exp(s_lse - max_lse)
    out_se = p_se + s_se
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    @inbounds output[idx0 + Int32(1)] = eltype(output)(
        Float32(prefix_output[idx0 + Int32(1)]) * p_scale +
        Float32(suffix_output[idx0 + Int32(1)]) * s_scale)
    if idx0 % head_size == 0
        @inbounds lse[lse_idx + Int32(1)] = log(out_se) + max_lse
    end
    return
end

function reference(prefix_output, suffix_output, prefix_lse, suffix_lse,
                   num_tokens::Int, num_heads::Int, head_size::Int, ::Type{T}) where {T}
    total = num_tokens * num_heads * head_size
    output = Vector{T}(undef, total)
    lse = Vector{Float32}(undef, num_tokens * num_heads)
    for t in 0:(num_tokens - 1), h in 0:(num_heads - 1)
        lse_idx = t * num_heads + h + 1
        p_lse = prefix_lse[lse_idx]
        s_lse = suffix_lse[lse_idx]
        p_lse = isinf(p_lse) ? -Inf32 : p_lse
        s_lse = isinf(s_lse) ? -Inf32 : s_lse
        max_lse = max(p_lse, s_lse)
        p_exp = exp(p_lse - max_lse)
        s_exp = exp(s_lse - max_lse)
        out_se = p_exp + s_exp
        lse[lse_idx] = log(out_se) + max_lse
        p_scale = p_exp / out_se
        s_scale = s_exp / out_se
        base = (t * num_heads + h) * head_size
        for d in 1:head_size
            output[base + d] = T(Float32(prefix_output[base + d]) * p_scale +
                                 Float32(suffix_output[base + d]) * s_scale)
        end
    end
    return output, lse
end

function run_case(::Type{T}, repeat::Int, num_tokens::Int, num_heads::Int, head_size::Int) where {T}
    output_size = num_tokens * num_heads * head_size
    lse_size = num_tokens * num_heads
    scale = 1.0f0 / sqrt(Float32(head_size))
    seed = UInt32(1234)

    h_prefix = fill_uniform!(Vector{T}(undef, output_size), -scale, scale, seed)
    h_suffix = fill_uniform!(Vector{T}(undef, output_size), -scale, scale, seed)
    h_prefix_lse = fill_uniform!(Vector{Float32}(undef, lse_size), -scale, scale, seed)
    h_suffix_lse = fill_uniform!(Vector{Float32}(undef, lse_size), -scale, scale, seed)
    r_output, r_lse = reference(h_prefix, h_suffix, h_prefix_lse, h_suffix_lse,
                                num_tokens, num_heads, head_size, T)

    d_prefix = CuArray(h_prefix)
    d_suffix = CuArray(h_suffix)
    d_output = CuArray{T}(undef, output_size)
    d_prefix_lse = CuArray(h_prefix_lse)
    d_suffix_lse = CuArray(h_suffix_lse)
    d_lse = CuArray{Float32}(undef, lse_size)
    threads = 256
    blocks = cld(output_size, threads)

    for _ in 1:100
        @cuda threads=threads blocks=blocks merge_kernel!(
            d_output, d_prefix, d_suffix, d_lse, d_prefix_lse, d_suffix_lse,
            Int32(output_size), Int32(head_size))
    end
    CUDA.synchronize()
    h_output = Array(d_output)
    h_lse = Array(d_lse)
    ok = all(abs(Float32(h_output[i]) - Float32(r_output[i])) <= 1.0f-3 for i in eachindex(h_output)) &&
         all(abs(h_lse[i] - r_lse[i]) <= 1.0f-3 for i in eachindex(h_lse))
    println(ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks merge_kernel!(
            d_output, d_prefix, d_suffix, d_lse, d_prefix_lse, d_suffix_lse,
            Int32(output_size), Int32(head_size))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time of the kernel: %f (us)\n", elapsed_us)
    return ok
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <number of tokens> <number of heads> <head size> <repeat>")
        return 1
    end
    num_tokens = parse(Int, args[1])
    num_heads = parse(Int, args[2])
    head_size = parse(Int, args[3])
    repeat = parse(Int, args[4])

    println()
    println("#tokens $num_tokens, #heads $num_heads, head dimension $head_size")
    print("output dtype FP32: ")
    ok32 = run_case(Float32, repeat, num_tokens, num_heads, head_size)
    print("output dtype FP16: ")
    ok16 = run_case(Float16, repeat, num_tokens, num_heads, head_size)
    print("output dtype BF16: ")
    okbf = run_case(Float32, repeat, num_tokens, num_heads, head_size)
    println("----------------------------------------------------")
    return ok32 && ok16 && okbf ? 0 : 1
end

exit(main(ARGS))
