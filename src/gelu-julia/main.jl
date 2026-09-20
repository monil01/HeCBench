using CUDA
using Printf

function gelu_f32(x::Float32)
    return 0.5f0 * x * (1.0f0 + tanh(0.79788456f0 * (x + 0.044715f0 * x * x * x)))
end

function input_value(idx0::Int64)
    return Float16(Float32(idx0 % Int64(1024)) / 1024.0f0)
end

function bias_value(y::Int32)
    return Float16(Float32((y % Int32(12)) - Int32(6)))
end

function init_input_kernel!(src, total::Int64)
    idx0 = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx0
    while i < total
        src[i + 1] = input_value(i)
        i += stride
    end
    return
end

function init_bias_kernel!(bias, width::Int32)
    y = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if y < width
        bias[y + Int32(1)] = bias_value(y)
    end
    return
end

function gelu_bias_loop_base!(src, bias, width::Int32, height::Int32)
    x = blockIdx().x - Int32(1)
    batch = blockIdx().y - Int32(1)
    index = batch * width * height + x * width
    y = threadIdx().x - Int32(1)
    while y < width
        v = Float32(src[index + y + Int32(1)]) + Float32(bias[y + Int32(1)])
        src[index + y + Int32(1)] = Float16(gelu_f32(v))
        y += blockDim().x
    end
    return
end

function gelu_bias_loop_vec2!(src, bias, width::Int32, height::Int32)
    x = blockIdx().x - Int32(1)
    batch = blockIdx().y - Int32(1)
    index = batch * width * height + x * width
    y = (threadIdx().x - Int32(1)) * Int32(2)
    step = blockDim().x * Int32(2)
    while y < width
        v0 = Float32(src[index + y + Int32(1)]) + Float32(bias[y + Int32(1)])
        src[index + y + Int32(1)] = Float16(gelu_f32(v0))
        y1 = y + Int32(1)
        if y1 < width
            v1 = Float32(src[index + y1 + Int32(1)]) + Float32(bias[y1 + Int32(1)])
            src[index + y1 + Int32(1)] = Float16(gelu_f32(v1))
        end
        y += step
    end
    return
end

function verify_kernel!(src, flag, total::Int64, width::Int32)
    idx0 = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx0
    while i < total
        y = Int32(i % Int64(width))
        v = Float32(input_value(i)) + Float32(bias_value(y))
        ref = Float16(gelu_f32(v))
        if abs(Float32(src[i + 1]) - Float32(ref)) > 1.0f-3
            flag[1] = Int32(1)
        end
        i += stride
    end
    return
end

function reset_input!(src, total::Int64)
    blocks = min(cld(total, Int64(256)), Int64(65535))
    @cuda threads=256 blocks=Int(blocks) init_input_kernel!(src, total)
    return
end

function verify_output(src, total::Int64, width::Int32)
    flag = CuArray([Int32(0)])
    blocks = min(cld(total, Int64(256)), Int64(65535))
    @cuda threads=256 blocks=Int(blocks) verify_kernel!(src, flag, total, width)
    CUDA.synchronize()
    return CUDA.@allowscalar flag[1] == Int32(0)
end

function run_case(batch_size::Int, seq_len::Int, hidden_dim::Int, repeat::Int)
    src_size = Int64(batch_size) * Int64(seq_len) * Int64(hidden_dim)

    block_size = hidden_dim >= 4096 ? 512 : (hidden_dim >= 2048 ? 256 : 128)
    grid = (seq_len, batch_size)

    d_bias = CuArray{Float16}(undef, hidden_dim)
    @cuda threads=256 blocks=cld(hidden_dim, 256) init_bias_kernel!(d_bias, Int32(hidden_dim))

    d_output = CuArray{Float16}(undef, src_size)
    reset_input!(d_output, src_size)
    @cuda threads=block_size blocks=grid gelu_bias_loop_base!(d_output, d_bias, Int32(hidden_dim), Int32(seq_len))
    CUDA.synchronize()
    ok_base = verify_output(d_output, src_size, Int32(hidden_dim))
    println(ok_base ? "PASS" : "FAIL")

    reset_input!(d_output, src_size)
    @cuda threads=block_size blocks=grid gelu_bias_loop_vec2!(d_output, d_bias, Int32(hidden_dim), Int32(seq_len))
    CUDA.synchronize()
    ok_vec = verify_output(d_output, src_size, Int32(hidden_dim))
    println(ok_vec ? "PASS" : "FAIL")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block_size blocks=grid gelu_bias_loop_vec2!(d_output, d_bias, Int32(hidden_dim), Int32(seq_len))
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6
    @printf("Average execution time of vectorized kernel %f (ms)\n", elapsed_ms / repeat)

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block_size blocks=grid gelu_bias_loop_base!(d_output, d_bias, Int32(hidden_dim), Int32(seq_len))
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6
    @printf("Average execution time of baseline kernel %f (ms)\n", elapsed_ms / repeat)

    return ok_base && ok_vec
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <batch> <sequence length> <hidden dimension> <repeat>")
        println("The hidden dimension is a multiple of two")
        return 1
    end

    batch_size = parse(Int, args[1])
    seq_len = parse(Int, args[2])
    hidden_dim = parse(Int, args[3])
    repeat = parse(Int, args[4])

    ok = run_case(batch_size, seq_len, hidden_dim, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
