using CUDA
using Printf
using Random

const REDUCE_THREADS = 1024

@inline function clamp_i8_i32(x::Int32)
    x < Int32(-128) && return Int8(-128)
    x > Int32(127) && return Int8(127)
    return Int8(x)
end

@inline function float_to_i32_rn(x::Float32)
    x >= Float32(2147483647) && return typemax(Int32)
    x <= Float32(-2147483648) && return typemin(Int32)
    return Int32(round(x))
end

@inline function float_to_i8_rn(x::Float32)
    y = round(x)
    y < Float32(-128) && return Int8(-128)
    y > Float32(127) && return Int8(127)
    return Int8(y)
end

function static_scaled_int8_quant_kernel!(input, out, scale::Float32, hidden_size::Int32)
    tid0 = threadIdx().x - Int32(1)
    token0 = blockIdx().x - Int32(1)
    base = token0 * hidden_size
    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        out[idx] = float_to_i8_rn(Float32(input[idx]) / scale)
        i0 += blockDim().x
    end
    return
end

function static_scaled_int8_azp_quant_kernel!(input, out, scale::Float32, azp::Int32, hidden_size::Int32)
    tid0 = threadIdx().x - Int32(1)
    token0 = blockIdx().x - Int32(1)
    base = token0 * hidden_size
    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        quant = float_to_i32_rn(Float32(input[idx]) / scale) + azp
        out[idx] = clamp_i8_i32(quant)
        i0 += blockDim().x
    end
    return
end

function dynamic_scaled_int8_quant_kernel!(input, out, scale, hidden_size::Int32)
    tid = threadIdx().x
    tid0 = tid - Int32(1)
    token0 = blockIdx().x - Int32(1)
    base = token0 * hidden_size
    scratch = CUDA.@cuStaticSharedMem(Float32, REDUCE_THREADS)

    absmax = Float32(0)
    i0 = tid0
    @inbounds while i0 < hidden_size
        v = abs(Float32(input[base + i0 + Int32(1)]))
        absmax = ifelse(v > absmax, v, absmax)
        i0 += blockDim().x
    end
    @inbounds scratch[tid] = absmax
    sync_threads()

    stride = blockDim().x ÷ Int32(2)
    while stride > 0
        if tid <= stride
            other = @inbounds scratch[tid + stride]
            mine = @inbounds scratch[tid]
            @inbounds scratch[tid] = ifelse(other > mine, other, mine)
        end
        sync_threads()
        stride ÷= Int32(2)
    end

    block_absmax = @inbounds scratch[1]
    if tid == Int32(1)
        @inbounds scale[token0 + Int32(1)] = block_absmax / Float32(127)
    end
    sync_threads()

    tmp_scale = Float32(127) / block_absmax
    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        out[idx] = float_to_i8_rn(Float32(input[idx]) * tmp_scale)
        i0 += blockDim().x
    end
    return
end

function dynamic_scaled_int8_azp_quant_kernel!(input, out, scale, azp, hidden_size::Int32)
    tid = threadIdx().x
    tid0 = tid - Int32(1)
    token0 = blockIdx().x - Int32(1)
    base = token0 * hidden_size
    scratch = CUDA.@cuStaticSharedMem(Float32, REDUCE_THREADS)

    max_val = Float32(floatmin(Float32))
    min_val = Float32(floatmax(Float32))
    i0 = tid0
    @inbounds while i0 < hidden_size
        v = Float32(input[base + i0 + Int32(1)])
        max_val = ifelse(v > max_val, v, max_val)
        min_val = ifelse(v < min_val, v, min_val)
        i0 += blockDim().x
    end

    @inbounds scratch[tid] = max_val
    sync_threads()
    stride = blockDim().x ÷ Int32(2)
    while stride > 0
        if tid <= stride
            other = @inbounds scratch[tid + stride]
            mine = @inbounds scratch[tid]
            @inbounds scratch[tid] = ifelse(other > mine, other, mine)
        end
        sync_threads()
        stride ÷= Int32(2)
    end
    block_max = @inbounds scratch[1]
    sync_threads()

    @inbounds scratch[tid] = min_val
    sync_threads()
    stride = blockDim().x ÷ Int32(2)
    while stride > 0
        if tid <= stride
            other = @inbounds scratch[tid + stride]
            mine = @inbounds scratch[tid]
            @inbounds scratch[tid] = ifelse(other < mine, other, mine)
        end
        sync_threads()
        stride ÷= Int32(2)
    end
    block_min = @inbounds scratch[1]

    scale_sh = (block_max - block_min) / Float32(255)
    azp_sh = float_to_i32_rn(Float32(-128) - block_min / scale_sh)
    if tid == Int32(1)
        @inbounds scale[token0 + Int32(1)] = scale_sh
        @inbounds azp[token0 + Int32(1)] = azp_sh
    end
    sync_threads()

    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        quant = float_to_i32_rn(Float32(input[idx]) / scale_sh) + azp_sh
        out[idx] = clamp_i8_i32(quant)
        i0 += blockDim().x
    end
    return
end

function static_scaled_ref!(input, out, scale::Float32)
    @inbounds for i in eachindex(input)
        out[i] = float_to_i8_rn(Float32(input[i]) / scale)
    end
    return out
end

function static_scaled_azp_ref!(input, out, scale::Float32, azp::Int32)
    @inbounds for i in eachindex(input)
        out[i] = clamp_i8_i32(float_to_i32_rn(Float32(input[i]) / scale) + azp)
    end
    return out
end

function dynamic_scaled_ref!(input, out, scales, num_tokens::Int, hidden_size::Int)
    @inbounds for token in 0:(num_tokens - 1)
        base = token * hidden_size
        absmax = Float32(0)
        for i0 in 0:(hidden_size - 1)
            v = abs(Float32(input[base + i0 + 1]))
            absmax = max(absmax, v)
        end
        scales[token + 1] = absmax / Float32(127)
        tmp_scale = Float32(127) / absmax
        for i0 in 0:(hidden_size - 1)
            idx = base + i0 + 1
            out[idx] = float_to_i8_rn(Float32(input[idx]) * tmp_scale)
        end
    end
    return out
end

function dynamic_scaled_azp_ref!(input, out, scales, azp, num_tokens::Int, hidden_size::Int)
    @inbounds for token in 0:(num_tokens - 1)
        base = token * hidden_size
        max_val = floatmin(Float32)
        min_val = floatmax(Float32)
        for i0 in 0:(hidden_size - 1)
            v = Float32(input[base + i0 + 1])
            max_val = max(max_val, v)
            min_val = min(min_val, v)
        end
        scale_val = (max_val - min_val) / Float32(255)
        azp_val = float_to_i32_rn(Float32(-128) - min_val / scale_val)
        scales[token + 1] = scale_val
        azp[token + 1] = azp_val
        for i0 in 0:(hidden_size - 1)
            idx = base + i0 + 1
            out[idx] = clamp_i8_i32(float_to_i32_rn(Float32(input[idx]) / scale_val) + azp_val)
        end
    end
    return out
end

function bf16_round(x::Float32)
    bits = reinterpret(UInt32, x)
    lsb = (bits >> 16) & UInt32(1)
    rounded = bits + UInt32(0x7fff) + lsb
    return reinterpret(Float32, rounded & UInt32(0xffff0000))
end

function make_input(::Type{Float16}, values)
    return Float16.(values)
end

function make_input(::Type{Float32}, values)
    return Float32.(values)
end

function make_bf16_input(values)
    out = Vector{Float32}(undef, length(values))
    @inbounds for i in eachindex(values)
        out[i] = bf16_round(Float32(values[i]))
    end
    return out
end

function run_quant(label::String, input, num_tokens::Int, hidden_size::Int, repeat::Int)
    println("Input type is $label")
    total = num_tokens * hidden_size
    d_input = CuArray(input)
    d_output = CUDA.zeros(Int8, total)
    d_scale = CUDA.zeros(Float32, num_tokens)
    d_azp = CUDA.zeros(Int32, num_tokens)
    h_output = Vector{Int8}(undef, total)
    h_ref = Vector{Int8}(undef, total)
    h_scale = Vector{Float32}(undef, num_tokens)
    h_azp = Vector{Int32}(undef, num_tokens)
    threads = min(hidden_size, REDUCE_THREADS)
    blocks = num_tokens
    scale = Float32(0.1)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks static_scaled_int8_quant_kernel!(d_input, d_output, scale, Int32(hidden_size))
    end
    CUDA.synchronize()
    @printf("Average execution time of static_scaled_int8_quant kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    copyto!(h_output, d_output)
    static_scaled_ref!(input, h_ref, scale)
    error = h_output == h_ref ? 0 : 1

    azp = Int32(54)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks static_scaled_int8_azp_quant_kernel!(d_input, d_output, scale, azp, Int32(hidden_size))
    end
    CUDA.synchronize()
    @printf("Average execution time of static_scaled_int8_quant_azp kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    copyto!(h_output, d_output)
    static_scaled_azp_ref!(input, h_ref, scale, azp)
    error += h_output == h_ref ? 0 : 1

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks dynamic_scaled_int8_quant_kernel!(d_input, d_output, d_scale, Int32(hidden_size))
    end
    CUDA.synchronize()
    @printf("Average execution time of dynamic_scaled_int8_quant kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    copyto!(h_output, d_output)
    dynamic_scaled_ref!(input, h_ref, h_scale, num_tokens, hidden_size)
    error += h_output == h_ref ? 0 : 1

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks dynamic_scaled_int8_azp_quant_kernel!(d_input, d_output, d_scale, d_azp, Int32(hidden_size))
    end
    CUDA.synchronize()
    @printf("Average execution time of dynamic_scaled_int8_quant_azp kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    copyto!(h_output, d_output)
    dynamic_scaled_azp_ref!(input, h_ref, h_scale, h_azp, num_tokens, hidden_size)
    error += h_output == h_ref ? 0 : 1

    println(error == 0 ? "PASS" : "FAIL")
    return error == 0
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <number of tokens> <hidden size> <repeat>")
        return 1
    end
    num_tokens = parse(Int, ARGS[1])
    hidden_size = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])

    Random.seed!(123)
    values = rand(-300:699, num_tokens * hidden_size)

    ok = true
    ok &= run_quant("FP16", make_input(Float16, values), num_tokens, hidden_size, repeat)
    ok &= run_quant("BF16", make_bf16_input(values), num_tokens, hidden_size, repeat)
    ok &= run_quant("FP32", make_input(Float32, values), num_tokens, hidden_size, repeat)
    return ok ? 0 : 1
end

exit(main())
