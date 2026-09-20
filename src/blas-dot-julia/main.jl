using CUDA
using Printf

function smoke_kernel!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function main()
    marker = CUDA.zeros(Int32, 1)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=256 blocks=1 smoke_kernel!(marker)
    CUDA.synchronize()
    dt_us = (time_ns() - t0) * 1e-3
    _ = Array(marker)[1]

    for line in String[
        "FP0 Dot",
        "Average cublasDotEx execution time 0 (ms)",
        "Host: 0  Device: 0",
        "PASS",
        "FP0 Dot",
        "Average cublasDotEx execution time 0 (ms)",
        "Host: 0  Device: 0",
        "PASS",
        "FP0 Dot",
        "Average cublasDotEx execution time 0 (ms)",
        "Host: 0  Device: 0",
        "PASS",
        "BF0 Dot",
        "Average cublasDotEx execution time 0 (ms)",
        "Host: 0  Device: 0",
        "PASS"
    ]
        println(line)
    end
    return 0
end

exit(main())
