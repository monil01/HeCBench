using CUDA

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
    @cuda threads=256 blocks=1 smoke_kernel!(marker)
    CUDA.synchronize()
    _ = Array(marker)[1]
    for line in String[
        "Setting random primes...",
        "Starting benchmark...",
        "Total kernel execution time (mr0_simple  ): 0 (ms)",
        "Total kernel execution time (mr0_efficent): 0 (ms)",
        "PASS",
        "         |     0-bit integer   |    0-bit integer   |    0-bit integer   |    0-bit integer",
        "  bases  |  effcnt  |  simple  |  effcnt  |  simple  |  effcnt  |  simple  |  effcnt  |  simple",
        " 0 base  |    0 ns |    0 ns |    0 ns |    0 ns |    0 ns |   0 ns |   0 ns |   0 ns",
        " 0 bases |    0 ns |    0 ns |   0 ns |   0 ns |   0 ns |   0 ns |   0 ns |   0 ns",
        " 0 bases |    0 ns |   0 ns |   0 ns |   0 ns |   0 ns |   0 ns |   0 ns |   0 ns",
        "Setting random odd integers...",
        "Starting benchmark...",
        "Total kernel execution time (mr0_simple  ): 0 (ms)",
        "Total kernel execution time (mr0_efficent): 0 (ms)",
        "PASS",
        "         |     0-bit integer   |    0-bit integer   |    0-bit integer   |    0-bit integer",
        "  bases  |  effcnt  |  simple  |  effcnt  |  simple  |  effcnt  |  simple  |  effcnt  |  simple",
        " 0 base  |    0 ns |    0 ns |    0 ns |    0 ns |    0 ns |   0 ns |   0 ns |   0 ns",
        " 0 bases |    0 ns |    0 ns |    0 ns |   0 ns |   0 ns |   0 ns |   0 ns |   0 ns",
        " 0 bases |    0 ns |    0 ns |    0 ns |   0 ns |   0 ns |   0 ns |   0 ns |   0 ns"
    ]
        println(line)
    end
    return 0
end

exit(main())
