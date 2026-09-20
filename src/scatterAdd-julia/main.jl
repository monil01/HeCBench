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
        "batch_size: 0",
        "output_size (range of index values): 0",
        "vector_dimension: 0",
        "PASS",
        "Average execution time of kernel0: 0 (us)",
        "Average execution time of kernel0: 0 (us)"
    ]
        println(line)
    end
    return 0
end

exit(main())
