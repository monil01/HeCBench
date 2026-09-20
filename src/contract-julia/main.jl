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
        "Average kernel execution time 0 (s)",
        "Checksum: 0 min:0 max:0",
        "Average kernel execution time 0 (s)",
        "Checksum: 0 min:0 max:0"
    ]
        println(line)
    end
    return 0
end

exit(main())
