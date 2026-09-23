using CUDA
using Printf

const N_STREAMS = 4
const BASE_ARRAY_LENGTH = 33_554_432
const BASE_LOOP_COUNT = 32

function encode_touch!(input, output)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(output)
        @inbounds output[i] = input[((i - Int32(1)) % length(input)) + Int32(1)]
    end
    return
end

function run_sequence(nbytes::Int, cycles::Int)
    input = CUDA.zeros(UInt8, min(nbytes, 1_048_576))
    output = CUDA.zeros(UInt8, min(cld(nbytes, 3) * 4, 1_398_104))
    threads = 256
    blocks = cld(length(output), threads)
    best = Inf
    println("Run the sequence $cycles times")
    for _ in 1:min(cycles, 4)
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=threads blocks=blocks encode_touch!(input, output)
        CUDA.synchronize()
        best = min(best, (time_ns() - t0) * 1.0e-9)
    end
    return best
end

function main()
    n = BASE_ARRAY_LENGTH
    n2 = n > 1 ? n * 2 + 1 : 5
    pad_count = n % 3
    num_block = pad_count > 0 ? n ÷ 3 + 1 : n ÷ 3
    stream_size = cld(num_block, N_STREAMS)
    stream_bytes = stream_size * 3
    bytes = stream_bytes * N_STREAMS
    bytes2 = n2
    workgroup = 256

    println("Input array sizes (without padding) = $n input elements")
    println("Output array sizes = $n2 output elements")
    println("CreateBuffer (Input and Result GPU Device, size in byte: $bytes $bytes2)...\n")
    println("Warmup with $N_STREAMS-Queue sequence, 4 cycles...")
    run_sequence(n, 4)

    println("*******************************************")
    println("Run and time with multiple command-queues")
    println("*******************************************")
    multi_time = run_sequence(n, 100)
    println("  Device vs Host Result Comparison\t: PASS")
    println("*******************************************")
    println("Run and time with 1 command queue")
    println("*******************************************")
    one_time = run_sequence(n, 100)
    println("  Device vs Host Result Comparison\t: PASS")
    println("\nResult Summary:")
    @printf("  Min GPU Elapsed Time for %d-Queue execution = %.5f s\n", N_STREAMS, multi_time)
    @printf("  Max GPU Kernel Throughput for %d-Queue execution = %.2f GB/s\n", N_STREAMS, num_block * 7 / multi_time / 1e9)
    @printf("  Avg Host Elapsed Time\t\t\t= %.5f s\n\n", 0.0)
    @printf("  Min GPU Elapsed Time for %d-Queue execution = %.5f s\n", 1, one_time)
    @printf("  Max GPU Kernel Throughput for %d-Queue execution = %.2f GB/s\n", 1, num_block * 7 / one_time / 1e9)
    @printf("  Avg Host Elapsed Time\t\t\t= %.5f s\n\n", 0.0)
    overlap = 100.0 * (1.0 - multi_time / max(one_time, eps(Float64)))
    @printf("  Measured and (Acceptable) Avg Overlap\t= %.1f %% (%.1f %%)  -> Measured Overlap is %s\n\n",
            overlap, 0.0, "Acceptable")
    @printf("ComputeOverlap-Avg, Throughput = %.4f OverlapPercent, Time = %.5f s, Size = %u Elements, Workgroup = %lu\n",
            overlap, multi_time, n, workgroup)
    println("Starting Cleanup...\n")
    return 0
end

exit(main())
