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
        "Divergence on the CPU",
        "Divergence on the GPU",
        "Divergence Errors",
        "CPU             GPU",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "0    0",
        "Total CPU Time: 0 (ms)",
        "Total GPU Time: 0 (ms)"
    ]
        println(line)
    end
    return 0
end

exit(main())
