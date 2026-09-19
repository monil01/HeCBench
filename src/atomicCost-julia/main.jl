using CUDA
using Printf

const BLOCK_SIZE = 256
const DEFAULT_LENGTH = 922_521_600

function without_atomic_kernel!(result, size::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    acc = zero(eltype(result))
    first = tid0 * size
    last = (tid0 + Int32(1)) * size - Int32(1)

    for i in first:last
        acc += convert(eltype(result), i % Int32(2))
    end

    @inbounds result[tid0 + Int32(1)] += acc
    return
end

function with_atomic_kernel!(result, size::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    idx = tid0 + Int32(1)
    first = tid0 * size
    last = (tid0 + Int32(1)) * size - Int32(1)

    for i in first:last
        CUDA.atomic_add!(pointer(result, idx), convert(eltype(result), i % Int32(2)))
    end

    return
end

function atomic_cost(::Type{T}, label::String, total_length::Int, size::Int, repeat::Int) where {T}
    println()
    println()
    @printf("Each thread sums up %d elements\n", size)

    if total_length % size != 0
        error("length must be divisible by size")
    end

    num_threads = div(total_length, size)
    if num_threads % BLOCK_SIZE != 0
        error("num_threads must be divisible by BLOCK_SIZE")
    end

    d_result_wi = CUDA.zeros(T, num_threads)
    d_result_wo = CUDA.zeros(T, num_threads)
    blocks = div(num_threads, BLOCK_SIZE)

    CUDA.synchronize()
    start_ns = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=blocks with_atomic_kernel!(d_result_wi, Int32(size))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start_ns) * 1.0e-3 / repeat
    @printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n", elapsed_us)

    start_ns = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=blocks without_atomic_kernel!(d_result_wo, Int32(size))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start_ns) * 1.0e-3 / repeat
    @printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n", elapsed_us)

    result_wi = Array(d_result_wi)
    result_wo = Array(d_result_wo)
    println(result_wi == result_wo ? "PASS" : "FAIL")
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <N> <repeat>")
        println("N: the number of elements to sum per thread (1 - 16)")
        return 1
    end

    nelems = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])
    total_length = parse(Int, get(ENV, "HECBENCH_ATOMICCOST_LENGTH", string(DEFAULT_LENGTH)))

    if total_length % BLOCK_SIZE != 0
        error("length must be divisible by BLOCK_SIZE")
    end

    println()
    println("FP64 atomic add")
    atomic_cost(Float64, "FP64", total_length, nelems, repeat)

    println()
    println("INT32 atomic add")
    atomic_cost(Int32, "INT32", total_length, nelems, repeat)

    println()
    println("FP32 atomic add")
    atomic_cost(Float32, "FP32", total_length, nelems, repeat)

    return 0
end

exit(main())
