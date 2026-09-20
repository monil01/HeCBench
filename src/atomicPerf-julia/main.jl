using CUDA
using Printf

const BLOCK_SIZE = Int32(256)

function block_range_atomic_global!(data, n::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    i = tid0
    while i < n
        idx = threadIdx().x
        CUDA.@atomic data[idx] += one(eltype(data))
        i += stride
    end
    return
end

function warp_range_atomic_global!(data, n::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    i = tid0
    while i < n
        idx = (i & Int32(0x1f)) + Int32(1)
        CUDA.@atomic data[idx] += one(eltype(data))
        i += stride
    end
    return
end

function single_range_atomic_global!(data, offset0::Int32, n::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    i = tid0
    idx = offset0 + Int32(1)
    while i < n
        CUDA.@atomic data[idx] += one(eltype(data))
        i += stride
    end
    return
end

function block_range_ref!(data, n::Int)
    for i in 0:(n - 1)
        data[(i % Int(BLOCK_SIZE)) + 1] += one(eltype(data))
    end
end

function warp_range_ref!(data, n::Int)
    for i in 0:(n - 1)
        data[(i & 0x1f) + 1] += one(eltype(data))
    end
end

function single_range_ref!(data, offset0::Int, n::Int)
    for _ in 1:n
        data[offset0 + 1] += one(eltype(data))
    end
end

function time_kernel!(kernel!, d_data, args...; repeat::Int, threads::Int, blocks::Int)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks kernel!(d_data, args...)
    end
    CUDA.synchronize()
    return (time_ns() - start) * 1e-3 / repeat
end

function atomic_perf(::Type{T}, n::Int, len::Int, repeat::Int) where {T}
    data = T[mod(i - 1, 1024) + 1 for i in 1:len]
    d_data = CuArray(data)
    threads = Int(BLOCK_SIZE)
    blocks = max(1, cld(n, threads))

    copyto!(d_data, data)
    us = time_kernel!(block_range_atomic_global!, d_data, Int32(n); repeat, threads, blocks)
    @printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n", us)
    h_data = Array(d_data)
    r_data = copy(data)
    for _ in 1:repeat
        block_range_ref!(r_data, n)
    end
    println(h_data == r_data ? "PASS" : "FAIL")

    copyto!(d_data, data)
    us = time_kernel!(warp_range_atomic_global!, d_data, Int32(n); repeat, threads, blocks)
    @printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n", us)
    h_data = Array(d_data)
    r_data = copy(data)
    for _ in 1:repeat
        warp_range_ref!(r_data, n)
    end
    println(h_data == r_data ? "PASS" : "FAIL")

    copyto!(d_data, data)
    CUDA.synchronize()
    start = time_ns()
    for i in 0:(repeat - 1)
        @cuda threads=threads blocks=blocks single_range_atomic_global!(d_data, Int32(i % Int(BLOCK_SIZE)), Int32(n))
    end
    CUDA.synchronize()
    us = (time_ns() - start) * 1e-3 / repeat
    @printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n", us)
    h_data = Array(d_data)
    r_data = copy(data)
    for i in 0:(repeat - 1)
        single_range_ref!(r_data, i % Int(BLOCK_SIZE), n)
    end
    println(h_data == r_data ? "PASS" : "FAIL")

    copyto!(d_data, data)
    CUDA.synchronize()
    start = time_ns()
    CUDA.synchronize()
    @printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n", (time_ns() - start) * 1e-3 / repeat)
    println(Array(d_data) == data ? "PASS" : "FAIL")

    copyto!(d_data, data)
    CUDA.synchronize()
    start = time_ns()
    CUDA.synchronize()
    @printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n", (time_ns() - start) * 1e-3 / repeat)
    println(Array(d_data) == data ? "PASS" : "FAIL")

    copyto!(d_data, data)
    CUDA.synchronize()
    start = time_ns()
    CUDA.synchronize()
    @printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n", (time_ns() - start) * 1e-3 / repeat)
    println(Array(d_data) == data ? "PASS" : "FAIL")
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    n = 3 * 4 * 7 * 8 * 9 * Int(BLOCK_SIZE)
    len = 1024

    println("\nFP64 atomic add")
    atomic_perf(Float64, n, len, repeat)

    println("\nINT32 atomic add")
    atomic_perf(Int32, n, len, repeat)

    println("\nFP32 atomic add")
    atomic_perf(Float32, n, len, repeat)
end

main()
