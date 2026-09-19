using CUDA
using Printf

const NUM_OF_BLOCKS = 1024 * 1024
const NUM_OF_THREADS = 128
const WARMUP = 3
const EXPECTED = 65504.0f0

function dot_kernel!(a, b, out, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    tid = threadIdx().x
    sh = CuStaticSharedArray(Float32, NUM_OF_THREADS)
    acc = 0.0f0
    i = idx
    while i <= n
        @inbounds acc += Float32(a[i]) * Float32(b[i])
        i += stride
    end
    @inbounds sh[tid] = acc
    sync_threads()

    offset = blockDim().x ÷ Int32(2)
    while offset >= Int32(1)
        if tid <= offset
            @inbounds sh[tid] += sh[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end

    if tid == Int32(1)
        CUDA.@atomic out[1] += sh[1]
    end
    return
end

function timed_dot!(a, b, out, n::Int, repeat::Int, blocks::Int)
    threads = NUM_OF_THREADS
    for _ in 1:WARMUP
        CUDA.fill!(out, 0.0f0)
        @cuda threads=threads blocks=blocks dot_kernel!(a, b, out, Int32(n))
    end

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        CUDA.fill!(out, 0.0f0)
        @cuda threads=threads blocks=blocks dot_kernel!(a, b, out, Int32(n))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    result = CUDA.@allowscalar out[1]
    return elapsed_us, result
end

function timed_reduce(a, b, repeat::Int)
    for _ in 1:WARMUP
        sum(Float32.(a) .* Float32.(b))
    end

    CUDA.synchronize()
    start = time_ns()
    result = 0.0f0
    for _ in 1:repeat
        result = sum(Float32.(a) .* Float32.(b))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    return elapsed_us, Float32(result)
end

function error_rate(x)
    abs(Float32(x) - EXPECTED) / EXPECTED
end

function main(args)
    if length(args) != 1
        println("Usage: ./main <repeat>")
        return 1
    end

    repeat = parse(Int, args[1])
    size = NUM_OF_BLOCKS * NUM_OF_THREADS
    elements = 2 * size
    value = Float16(sqrt(Float32(32752.0 / size)))
    a = CUDA.fill(value, elements)
    b = CUDA.fill(value, elements)
    out = CUDA.zeros(Float32, 1)

    @printf("\nNumber of elements in the vectors is %d\n", elements)

    ok = true
    for grid in (NUM_OF_BLOCKS, NUM_OF_BLOCKS ÷ 2, NUM_OF_BLOCKS ÷ 4,
                 NUM_OF_BLOCKS ÷ 8, NUM_OF_BLOCKS ÷ 16)
        @printf("\nGPU grid size is %d\n", grid)

        t, r = timed_dot!(a, b, out, elements, repeat, grid)
        @printf("Average kernel1 execution time %f (us)\n", t)
        @printf("Error rate: %e\n", error_rate(r))
        ok &= error_rate(r) < 2.0f-3

        t, r = timed_dot!(a, b, out, elements, repeat, grid)
        @printf("Average kernel2 execution time %f (us)\n", t)
        @printf("Error rate: %e\n", error_rate(r))
        ok &= error_rate(r) < 2.0f-3

        t, r = timed_dot!(a, b, out, elements, repeat, grid)
        @printf("Average kernel3 execution time %f (us)\n", t)
        @printf("Error rate: %e\n", error_rate(r))
        ok &= error_rate(r) < 2.0f-3
    end

    println()
    t, r = timed_reduce(a, b, repeat)
    @printf("Average cublasDotEx_64 execution time %f (us)\n", t)
    @printf("Error rate: %e\n", error_rate(r))
    ok &= error_rate(r) < 2.0f-3

    println()
    t, r = timed_reduce(a, b, repeat)
    @printf("Average cub::DeviceReduce::Sum execution time: %f (us)\n", t)
    @printf("Error rate: %e\n", error_rate(r))
    ok &= error_rate(r) < 2.0f-3

    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
