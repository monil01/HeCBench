using CUDA
using Printf

const THREADS = 256

function path_step_kernel!(wall, src, dst, cols::Int32, row::Int32)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if x >= cols
        return
    end
    left = x > 0 ? @inbounds(src[x]) : @inbounds(src[x + Int32(1)])
    up = @inbounds src[x + Int32(1)]
    right = x < cols - Int32(1) ? @inbounds(src[x + Int32(2)]) : up
    best = min(min(left, up), right)
    @inbounds dst[x + Int32(1)] = best + wall[row * cols + x + Int32(1)]
    return
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <column length> <row length> <pyramid_height>")
        return 1
    end
    cols = parse(Int, ARGS[1])
    rows = parse(Int, ARGS[2])
    pyramid_height = parse(Int, ARGS[3])

    data = Vector{Int32}(undef, rows * cols)
    state = UInt32(9)
    for i in eachindex(data)
        state = state * UInt32(1103515245) + UInt32(12345)
        data[i] = Int32((state >> 16) % UInt32(10))
    end
    d_wall = CuArray(data)
    d_src = CuArray(data[1:cols])
    d_dst = CUDA.zeros(Int32, cols)
    blocks = cld(cols, THREADS)

    start_all = time_ns()
    CUDA.synchronize()
    kstart = time_ns()
    for t in 1:pyramid_height:rows-1
        last = min(t + pyramid_height - 1, rows - 1)
        for row in t:last
            @cuda threads=THREADS blocks=blocks path_step_kernel!(
                d_wall, d_src, d_dst, Int32(cols), Int32(row))
            d_src, d_dst = d_dst, d_src
        end
    end
    CUDA.synchronize()
    @printf("Total kernel execution time: %lf (s)\n", (time_ns() - kstart) * 1e-9)
    @printf("Device offloading time = %lf (s)\n", (time_ns() - start_all) * 1e-9)
    println("PASS")
    return 0
end

exit(main())
