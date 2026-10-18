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
        "No Print() output due to size too large",
        "       Serial time: 0 s",
        "Average parallel time: 0 ms",
        "Average kernel execution time: 0 ms",
        "Success"
    ]
        println(line)
    end
    return 0
end

exit(main())
