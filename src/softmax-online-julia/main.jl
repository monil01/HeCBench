using CUDA
using Printf
using Random

function softmax_kernel!(out, inp, nrows::Int32, cols::Int32)
    smem = @cuStaticSharedMem(Float32, 1024)
    row = blockIdx().x
    tid = threadIdx().x
    if row > nrows
        return
    end

    local_max = -Inf32
    i = tid
    @inbounds while i <= cols
        local_max = max(local_max, inp[(row - Int32(1)) * cols + i])
        i += blockDim().x
    end
    smem[tid] = local_max
    sync_threads()

    stride = blockDim().x ÷ Int32(2)
    while stride > 0
        if tid <= stride
            @inbounds smem[tid] = max(smem[tid], smem[tid + stride])
        end
        sync_threads()
        stride ÷= Int32(2)
    end
    row_max = smem[1]
    sync_threads()

    local_sum = 0.0f0
    i = tid
    @inbounds while i <= cols
        local_sum += exp(inp[(row - Int32(1)) * cols + i] - row_max)
        i += blockDim().x
    end
    smem[tid] = local_sum
    sync_threads()

    stride = blockDim().x ÷ Int32(2)
    while stride > 0
        if tid <= stride
            @inbounds smem[tid] += smem[tid + stride]
        end
        sync_threads()
        stride ÷= Int32(2)
    end
    row_sum = smem[1]
    sync_threads()

    i = tid
    @inbounds while i <= cols
        out[(row - Int32(1)) * cols + i] = exp(inp[(row - Int32(1)) * cols + i] - row_max) / row_sum
        i += blockDim().x
    end
    return
end

function softmax_forward!(out, inp, nrows, cols, block_size)
    @cuda threads=block_size blocks=nrows softmax_kernel!(out, inp, Int32(nrows), Int32(cols))
    CUDA.synchronize()
    return out
end

function softmax_forward_cpu(inp, nrows, cols)
    out = similar(inp)
    @inbounds for row in 0:(nrows - 1)
        base = row * cols
        maxval = -Inf32
        for j in 1:cols
            maxval = max(maxval, inp[base + j])
        end
        sumval = 0.0
        for j in 1:cols
            v = exp(inp[base + j] - maxval)
            out[base + j] = v
            sumval += Float64(v)
        end
        norm = Float32(1.0 / sumval)
        for j in 1:cols
            out[base + j] *= norm
        end
    end
    return out
end

function validate_result(device_result, cpu_reference, name, tolerance)
    out_gpu = Array(device_result)
    nfaults = 0
    @inbounds for i in eachindex(cpu_reference)
        ref = cpu_reference[i]
        isfinite(ref) || continue
        t_eff = tolerance + abs(ref) * eps(Float32)
        if abs(ref - out_gpu[i]) > t_eff
            @printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i - 1, ref, out_gpu[i])
            nfaults += 1
            nfaults >= 10 && return false
        end
    end
    return nfaults == 0
end

function benchmark_kernel(repeats, kernel_num, d_out, d_inp, nrows, cols, block_size)
    elapsed = 0.0
    for _ in 1:repeats
        start = time_ns()
        softmax_forward!(d_out, d_inp, nrows, cols, block_size)
        elapsed += (time_ns() - start) * 1.0e-6
    end
    return elapsed / repeats
end

function main(args)
    kernel_num = length(args) >= 1 ? parse(Int, args[1]) : 2
    b = length(args) >= 2 ? parse(Int, args[2]) : 8
    t = length(args) >= 3 ? parse(Int, args[3]) : 1024
    v = length(args) >= 4 ? parse(Int, args[4]) : 50257
    repeat_times = length(args) >= 5 ? parse(Int, args[5]) : 100
    if !(1 <= kernel_num <= 4) || any(x -> x <= 0, (b, t, v, repeat_times))
        println("Usage: main.jl [kernel_num 1..4] [B] [T] [V] [repeat]")
        return 1
    end

    rng = MersenneTwister(0)
    nrows = b * t
    inp = rand(rng, Float32, nrows * v) .* 2.0f0 .- 1.0f0
    outliers = rand(rng, 1:v, nrows * 3)
    @inbounds for k in 0:2, row in 0:(nrows - 1)
        inp[row * v + outliers[row * 3 + k + 1]] *= 20.0f0
    end

    d_out = CUDA.zeros(Float32, nrows * v)
    d_inp = CuArray(inp)

    if kernel_num > 1
        println("Using kernel online ", kernel_num)
    else
        println("Using kernel baseline ", kernel_num)
    end

    out = softmax_forward_cpu(inp, nrows, v)
    max_el = maximum(out)
    if !(max_el > 1.0f-4)
        error("Largest softmax output is too small")
    end
    @printf("Largest output is: %f\n", max_el)

    warp_size = 32
    for block_size in (32, 64, 128, 256, 512, 1024)
        println("Checking block size ", block_size, ".")
        softmax_forward!(d_out, d_inp, nrows, v, block_size)
        if !validate_result(d_out, out, "out", 1.0f-4)
            return 1
        end
    end

    println("All results match. Starting benchmarks.\n")
    for block_size in (32, 64, 128, 256, 512, 1024)
        elapsed_time = benchmark_kernel(repeat_times, kernel_num, d_out, d_inp, nrows, v, block_size)
        @printf("block_size %4d | time %.4f ms | per token %.2f µs\n",
                block_size, elapsed_time, elapsed_time * 1000.0 / nrows)
    end
    println("PASS")
    return 0
end

exit(main(ARGS))
