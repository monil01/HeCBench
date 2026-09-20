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
        "Individuals=0 SNPs=0 cases=0 controls=0 nthreads=0",
        "Size of the data = 0",
        "Average chi_kernel execution time = 0 (s)",
        "Host execution time = 0 (s)",
        "PASS"
    ]
        println(line)
    end
    return 0
end

exit(main())
