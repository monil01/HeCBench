using CUDA
using Printf

const BLOCK_SIZE = 512
const ILP = Int32(4)

function init_tensors!(x, y, lengths, offsets, ntensors::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    t = Int32(0)
    while t < ntensors
        len = lengths[t + Int32(1)]
        off = offsets[t + Int32(1)]
        i = tid
        while i < len
            idx = off + i + Int32(1)
            x[idx] = Float32((Int64(t) * 17 + Int64(i) * 3) % 1009)
            y[idx] = Float32((Int64(t) * 29 + Int64(i) * 5) % 1013)
            i += stride
        end
        t += Int32(1)
    end
    return
end

function axpby_kernel!(out, x, y, lengths, offsets, ntensors::Int32,
                       chunk_size::Int32, a::Float32, b::Float32)
    block0 = blockIdx().x - Int32(1)
    t = Int32(0)
    remaining = block0
    while t < ntensors
        chunks = cld(lengths[t + Int32(1)], chunk_size)
        if remaining < chunks
            break
        end
        remaining -= chunks
        t += Int32(1)
    end
    if t >= ntensors
        return
    end

    len = lengths[t + Int32(1)]
    off = offsets[t + Int32(1)]
    base = remaining * chunk_size
    i0 = (threadIdx().x - Int32(1)) * ILP
    stride = blockDim().x * ILP
    i = i0
    while i < chunk_size
        lane = Int32(0)
        while lane < ILP
            idx_local = base + i + lane
            if idx_local < len
                idx = off + idx_local + Int32(1)
                out[idx] = a * x[idx] + b * y[idx]
            end
            lane += Int32(1)
        end
        i += stride
    end
    return
end

function verify_kernel!(out, x, y, flag, total::Int32, a::Float32, b::Float32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= total
        ref = a * x[i] + b * y[i]
        if abs(out[i] - ref) > 1.0f-3
            flag[1] = Int32(1)
        end
        i += stride
    end
    return
end

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function build_lengths(max_tensors::Int)
    lengths = Vector{Int32}(undef, max_tensors)
    for t in 1:max_tensors
        lengths[t] = Int32(1024 + ((t * 7919) % 16384))
    end
    offsets = Vector{Int32}(undef, max_tensors)
    off = Int32(0)
    for t in 1:max_tensors
        offsets[t] = off
        off += lengths[t]
    end
    return lengths, offsets, off
end

function total_blocks(lengths::Vector{Int32}, chunk_size::Int)
    total = 0
    for len in lengths
        total += ceil_div(Int(len), chunk_size)
    end
    return total
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <number of tensors")
        return 1
    end
    max_tensors = parse(Int, args[1])

    h_lengths, h_offsets, total_i32 = build_lengths(max_tensors)
    total = Int(total_i32)

    lengths = CuArray(h_lengths)
    offsets = CuArray(h_offsets)
    x = CuArray{Float32}(undef, total)
    y = CuArray{Float32}(undef, total)
    out = CuArray{Float32}(undef, total)

    init_blocks = min(ceil_div(total, BLOCK_SIZE), 65535)
    @cuda threads=BLOCK_SIZE blocks=init_blocks init_tensors!(x, y, lengths, offsets, Int32(max_tensors))
    CUDA.synchronize()

    a = 1.0f0
    b = 1.0f0
    all_ok = true

    chunk_size = 256
    while chunk_size <= 1024 * 1024
        blocks = total_blocks(h_lengths, chunk_size)
        @cuda threads=BLOCK_SIZE blocks=blocks axpby_kernel!(out, x, y, lengths, offsets,
                                                             Int32(max_tensors), Int32(chunk_size), a, b)
        CUDA.synchronize()

        CUDA.synchronize()
        start = time_ns()
        @cuda threads=BLOCK_SIZE blocks=blocks axpby_kernel!(out, x, y, lengths, offsets,
                                                             Int32(max_tensors), Int32(chunk_size), a, b)
        CUDA.synchronize()
        elapsed_us = (time_ns() - start) * 1.0e-3
        @printf("Chunk size %8d | Total execution time of multi_tensor_axpby: %f (us)\n",
                chunk_size, elapsed_us)

        flag = CuArray([Int32(0)])
        verify_blocks = min(ceil_div(total, BLOCK_SIZE), 65535)
        @cuda threads=BLOCK_SIZE blocks=verify_blocks verify_kernel!(out, x, y, flag, Int32(total), a, b)
        CUDA.synchronize()
        ok = CUDA.@allowscalar flag[1] == Int32(0)
        println(ok ? "PASS" : "FAIL")
        all_ok &= ok
        chunk_size *= 2
    end
    return all_ok ? 0 : 1
end

exit(main(ARGS))
