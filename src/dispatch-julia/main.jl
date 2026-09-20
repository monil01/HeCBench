using CUDA
using Printf
using Statistics

const WARMUP_RUN_COUNT = 100
const TIMING_RUN_COUNT = 1000
const TOTAL_RUN_COUNT = WARMUP_RUN_COUNT + TIMING_RUN_COUNT
const BATCH_SIZE = 1000

function empty_kernel()
    return
end

function print_timing(test::String, results, batch::Int=1)
    timed = @view results[(WARMUP_RUN_COUNT + 1):TOTAL_RUN_COUNT]
    vals = (timed .* 1000.0) ./ batch
    mean_us = mean(vals)
    stddev_us = std(vals; corrected=false)
    @printf("\n %s: mean = %.1f us, stddev = %.1f us\n", test, mean_us, stddev_us)
end

function main()
    results = Vector{Float64}(undef, TOTAL_RUN_COUNT)

    @cuda threads=1 blocks=1 empty_kernel()
    CUDA.synchronize()

    for i in 1:TOTAL_RUN_COUNT
        start = time_ns()
        @cuda threads=1 blocks=1 empty_kernel()
        results[i] = (time_ns() - start) * 1.0e-6
    end
    print_timing("Enqueue rate", results)

    for i in 1:TOTAL_RUN_COUNT
        start = time_ns()
        @cuda threads=1 blocks=1 empty_kernel()
        CUDA.synchronize()
        results[i] = (time_ns() - start) * 1.0e-6
    end
    print_timing("Single dispatch latency", results)

    for i in 1:TOTAL_RUN_COUNT
        start = time_ns()
        for _ in 1:BATCH_SIZE
            @cuda threads=1 blocks=1 empty_kernel()
        end
        CUDA.synchronize()
        results[i] = (time_ns() - start) * 1.0e-6
    end
    print_timing("Batch dispatch latency", results, BATCH_SIZE)
    return 0
end

exit(main())
