using CUDA
using Printf

const STREAM_COUNT = 4

function inc_kernel!(out, input, n::Int32, inner_reps::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx < n
        value = Int32(0)
        for i in Int32(0):(inner_reps - Int32(1))
            value = (i == Int32(0) ? input[idx + Int32(1)] : out[idx + Int32(1)]) + Int32(1)
            @inbounds out[idx + Int32(1)] = value
        end
    end
    return
end

function process_with_streams!(d_in, d_out, n::Int, nreps::Int, inner_reps::Int, streams_used::Int)
    current = 1
    threads = 256
    blocks = n ÷ threads

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:nreps
        @cuda threads=threads blocks=blocks inc_kernel!(
            d_out[current], d_in[current], Int32(n), Int32(inner_reps))
        current = current % streams_used + 1
    end
    CUDA.synchronize()
    return (time_ns() - start) * 1e-6
end

function run_overlap()
    n = 1 << 22
    nreps = 10
    inner_reps = 5
    memsize = n * sizeof(Int32)

    @printf("Length of the array = %d\n", n)

    d_in = [CUDA.zeros(Int32, n) for _ in 1:STREAM_COUNT]
    d_out = [CUDA.zeros(Int32, n) for _ in 1:STREAM_COUNT]

    serial_time = process_with_streams!(d_in, d_out, n, nreps, inner_reps, 1)
    overlap_time = process_with_streams!(d_in, d_out, n, nreps, inner_reps, STREAM_COUNT)

    @printf("\nAverage measured timings over %d repetitions:\n", nreps)
    @printf(" Avg. time when execution fully serialized\t: %f ms\n", serial_time / nreps)
    @printf(" Avg. time when overlapped using %d streams\t: %f ms\n", STREAM_COUNT, overlap_time / nreps)
    @printf(" Avg. speedup gained (serialized - overlapped)\t: %f\n", (serial_time - overlap_time) / nreps)

    @printf("\nMeasured throughput:\n")
    @printf(" Fully serialized execution\t\t: %f GB/s\n", (nreps * (memsize * 2e-6)) / serial_time)
    @printf(" Overlapped using %d streams\t\t: %f GB/s\n", STREAM_COUNT, (nreps * (memsize * 2e-6)) / overlap_time)

    passed = true
    for arr in d_out
        host = Array(arr)
        if any(x -> x != inner_reps, host)
            passed = false
            break
        end
    end

    println()
    println(passed ? "PASS" : "FAIL")
    return passed ? 0 : 1
end

exit(run_overlap())
