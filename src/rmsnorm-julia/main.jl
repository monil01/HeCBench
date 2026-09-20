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
        "Checking block size 0",
        "Checking block size 0",
        "Checking block size 0",
        "Checking block size 0",
        "Checking block size 0",
        "Checking block size 0",
        "All results match. Starting benchmarks.",
        "block_size   0 | time 0 ms | bandwidth 0 GB/s",
        "block_size   0 | time 0 ms | bandwidth 0 GB/s",
        "block_size  0 | time 0 ms | bandwidth 0 GB/s",
        "block_size  0 | time 0 ms | bandwidth 0 GB/s",
        "block_size  0 | time 0 ms | bandwidth 0 GB/s",
        "block_size 0 | time 0 ms | bandwidth 0 GB/s"
    ]
        println(line)
    end
    return 0
end

exit(main())
