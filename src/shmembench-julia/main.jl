using CUDA
using Printf

const VECTOR_SIZE = 1024 * 1024
const BLOCK_SIZE = 256
const TOTAL_ITERATIONS = 1024

function shmem_kernel!(out)
    shm = @cuStaticSharedMem(Float32, 6 * BLOCK_SIZE * 4)
    tid0 = threadIdx().x - Int32(1)
    global0 = (blockIdx().x - Int32(1)) * blockDim().x + tid0

    for lane in Int32(0):Int32(3)
        base = tid0 * Int32(4) + lane + Int32(1)
        shm[base + Int32(0) * blockDim().x * Int32(4)] = Float32(tid0) + lane
        shm[base + Int32(1) * blockDim().x * Int32(4)] = Float32(tid0 + Int32(1)) + lane
        shm[base + Int32(2) * blockDim().x * Int32(4)] = Float32(tid0 + Int32(3)) + lane
        shm[base + Int32(3) * blockDim().x * Int32(4)] = Float32(tid0 + Int32(7)) + lane
        shm[base + Int32(4) * blockDim().x * Int32(4)] = Float32(tid0 + Int32(13)) + lane
        shm[base + Int32(5) * blockDim().x * Int32(4)] = Float32(tid0 + Int32(17)) + lane
    end
    sync_threads()

    stride = blockDim().x * Int32(4)
    for _ in Int32(1):Int32(TOTAL_ITERATIONS)
        for lane in Int32(0):Int32(3)
            base = tid0 * Int32(4) + lane + Int32(1)
            tmp = shm[base + stride]
            shm[base + stride] = shm[base]
            shm[base] = tmp
            tmp = shm[base + Int32(3) * stride]
            shm[base + Int32(3) * stride] = shm[base + Int32(2) * stride]
            shm[base + Int32(2) * stride] = tmp
            tmp = shm[base + Int32(5) * stride]
            shm[base + Int32(5) * stride] = shm[base + Int32(4) * stride]
            shm[base + Int32(4) * stride] = tmp
        end
        sync_threads()

        for lane in Int32(0):Int32(3)
            base = tid0 * Int32(4) + lane + Int32(1)
            tmp = shm[base + Int32(2) * stride]
            shm[base + Int32(2) * stride] = shm[base + stride]
            shm[base + stride] = tmp
            tmp = shm[base + Int32(4) * stride]
            shm[base + Int32(4) * stride] = shm[base + Int32(3) * stride]
            shm[base + Int32(3) * stride] = tmp
        end
        sync_threads()
    end

    for lane in Int32(0):Int32(3)
        base = tid0 * Int32(4) + lane + Int32(1)
        s = shm[base] + shm[base + stride] + shm[base + Int32(2) * stride] +
            shm[base + Int32(3) * stride] + shm[base + Int32(4) * stride] +
            shm[base + Int32(5) * stride]
        @inbounds out[global0 * Int32(4) + lane + Int32(1)] = s
    end
    return
end

function run_bench(repeat::Int)
    size = VECTOR_SIZE
    total_blocks = size ÷ BLOCK_SIZE
    blocks = total_blocks ÷ 4
    out = CUDA.zeros(Float32, size)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=blocks shmem_kernel!(out)
    end
    CUDA.synchronize()
    time_shmem_128b = Float64(time_ns() - t0) / repeat

    @printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6)
    println("Memory throughput")
    operations_bytes = (6 + 4 * 5 * TOTAL_ITERATIONS + 6) * size * sizeof(Float32)
    operations_128bit = (6 + 4 * 5 * TOTAL_ITERATIONS + 6) * size ÷ 4
    @printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n",
            Float64(operations_bytes) / time_shmem_128b,
            Float64(operations_128bit) / time_shmem_128b)
end

function main()
    println("Shared memory bandwidth microbenchmark")
    if length(ARGS) != 1
        @printf("Usage: %s <repeat>\n", PROGRAM_FILE)
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    datasize = VECTOR_SIZE * sizeof(Float64)
    @printf("Buffer sizes: %dMB\n", datasize ÷ (1024 * 1024))
    run_bench(repeat)
end

main()
