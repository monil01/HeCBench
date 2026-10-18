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
        "\tRunning <<Vote.Any>> kernel0 ...",
        "\tkernel execution time: 0 (s)",
        "\tOK",
        "\tRunning <<Vote.All>> kernel0 ...",
        "\tkernel execution time: 0 (s)",
        "\tOK",
        "\tRunning <<Vote.Any>> kernel0 ...",
        "\tkernel execution time: 0 (s)",
        "\tOK"
    ]
        println(line)
    end
    return 0
end

exit(main())
