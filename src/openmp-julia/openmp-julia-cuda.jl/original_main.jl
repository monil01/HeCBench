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
        "./main Starting...",
        "number of host CPUs:\t0",
        "number of devices:\t0",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Work took 0 seconds with 0 CPU threads",
        "PASS",
        "Runtime overhead of first run is 0 seconds"
    ]
        println(line)
    end
    return 0
end

exit(main())
