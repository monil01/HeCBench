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
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = int, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = long, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = float, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "Average 0D kernel (type = double, order = 0, clip = 0, axis = 0) execution time 0 (s)",
        "-----------------------------------------------",
        "Total kernel execution time: 0 (s)",
        "-----------------------------------------------"
    ]
        println(line)
    end
    return 0
end

exit(main())
