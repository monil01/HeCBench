using CUDA
using Printf

function clock_block_kernel!(out, slot::Int32, clock_count::Int64)
    clock_offset = Int64(0)
    for i in Int64(0):(clock_count - Int64(1))
        clock_offset += i % Int64(3)
    end
    @inbounds out[slot] = clock_offset
    return
end

function sum_kernel!(clocks, n::Int32)
    shared = CuStaticSharedArray(Int64, 32)
    tid = threadIdx().x
    lane = tid - Int32(1)
    acc = Int64(0)
    i = lane
    while i < n
        @inbounds acc += clocks[i + Int32(1)]
        i += blockDim().x
    end
    @inbounds shared[tid] = acc
    sync_threads()

    offset = Int32(16)
    while offset > 0
        if lane < offset
            @inbounds shared[tid] += shared[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end

    if lane == 0
        @inbounds clocks[1] = shared[1]
    end
    return
end

function host_clock_sum(clock_count::Int64)
    groups = clock_count ÷ Int64(3)
    rem = clock_count % Int64(3)
    total = groups * Int64(3)
    if rem == Int64(2)
        total += Int64(1)
    end
    return total
end

function run_concurrent_kernels(nkernels::Int)
    if nkernels < 1
        println("Usage: main.jl <number of concurrent kernels>")
        return 1
    end

    nstreams = nkernels + 1
    nbytes = nkernels * sizeof(Int64)
    kernel_time = 20

    println("[main.jl] - Starting...")
    @printf("time clocks = %d\n", Int64(kernel_time) * Int64(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE)))

    clock_count = Int64(kernel_time) * Int64(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE))
    d_clocks = CUDA.zeros(Int64, nkernels)

    CUDA.synchronize()
    start = time_ns()
    for i in 1:nkernels
        @cuda threads=1 blocks=1 clock_block_kernel!(
            d_clocks, Int32(i), clock_count)
    end
    @cuda threads=32 blocks=1 sum_kernel!(d_clocks, Int32(nkernels))
    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1e-9

    result = Array(d_clocks)[1]
    expected = Int64(nkernels) * host_clock_sum(clock_count)
    @printf("Measured time for sample = %.3fs\n", elapsed_s)
    println(result == expected ? "PASS" : "FAIL")
    return result == expected ? 0 : 1
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <number of concurrent kernels>")
        return 1
    end
    return run_concurrent_kernels(parse(Int, ARGS[1]))
end

exit(main())
