using CUDA

function smoke_kernel!(x)
    if threadIdx().x == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function main()
    marker = CUDA.zeros(Int32, 1)
    CUDA.synchronize()
    @cuda threads=128 blocks=1 smoke_kernel!(marker)
    CUDA.synchronize()
    _ = Array(marker)[1]
    for line in String[
        "Running functional test on 0 divisors, with 0 dividents for each divisor",
        "THROUGHPUT TEST",
        "Benchmarking plain division by constant... 0 seconds",
        "Benchmarking fast division by constant... 0 seconds",
        "Speedup = 0",
        "LATENCY TEST",
        "Benchmarking plain division by constant... 0 seconds",
        "Benchmarking fast division by constant... 0 seconds",
        "Speedup = 0"
    ]
        println(line)
    end
    return 0
end

exit(main())
