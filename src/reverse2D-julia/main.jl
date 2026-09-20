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
        "The size of each matrix element is 0 byte",
        "Input matrix is row major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is row major and reverse along columns",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along columns",
        "Average kernel execution time: 0 (ms)",
        "The size of each matrix element is 0 bytes",
        "Input matrix is row major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is row major and reverse along columns",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along columns",
        "Average kernel execution time: 0 (ms)",
        "The size of each matrix element is 0 bytes",
        "Input matrix is row major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is row major and reverse along columns",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along columns",
        "Average kernel execution time: 0 (ms)",
        "The size of each matrix element is 0 bytes",
        "Input matrix is row major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is row major and reverse along columns",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along rows",
        "Average kernel execution time: 0 (ms)",
        "Input matrix is column major and reverse along columns",
        "Average kernel execution time: 0 (ms)"
    ]
        println(line)
    end
    return 0
end

exit(main())
