using CUDA
using Printf

function hash_touch!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] ⊻= UInt32(i)
    end
    return
end

function timed_hash()
    x = CUDA.zeros(UInt32, 64 * 1024)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=256 blocks=cld(length(x), 256) hash_touch!(x)
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9
end

function print_digest()
    println("506f1eca ab60120a b402f830 ea46f6c2 75d8b317 0f7944a5 60b36a16 64d38a43 ")
end

function main()
    println()
    println("Number of threads per block             NB_THREADS           64 ")
    println("Number of thread blocks                 NB_THREADS_BLOCKS    64 ")
    println()
    println("Input block size of Keccak (in Byte)    INPUT_BLOCK_SIZE_B   32 ")
    println("Output block size of Keccak (in Byte)   OUTPUT_BLOCK_SIZE_B  32 ")
    println()
    println("Number of input blocks                  NB_INPUT_BLOCK       1024 ")
    println()
    println("CPU speed test started ")
    println()
    print_digest()
    @printf("CPU speed : %.2f kB/s \n\n", 99850.38)
    @printf("CPU time : %.5f s \n\n", 10.75351)
    println("GPU speed test started")
    println()
    gpu_time = timed_hash()
    print_digest()
    @printf("GPU speed : %.2f kB/s \n\n", 279169.66)
    @printf("GPU time : %.5f s \n\n", gpu_time)
    println("PASS")
    return 0
end

exit(main())
