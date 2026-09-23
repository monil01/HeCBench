using CUDA
using Printf

function stage0!(x, p1, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds p1[i] = x[i] + Float64(i & Int32(7))
        i += stride
    end
    return
end

function stage1!(p1, p2, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds p2[i] = p1[i] * 0.5
        i += stride
    end
    return
end

function stage2!(p2, y, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds y[i] = p2[i] - Float64(i & Int32(3))
        i += stride
    end
    return
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])

    n = 1_048_576
    x = CuArray(Float64.(mod.(0:(n - 1), 1024)) ./ 1024.0)
    p1 = CUDA.zeros(Float64, n)
    p2 = CUDA.zeros(Float64, n)
    y = CUDA.zeros(Float64, n)
    threads = 256
    blocks = cld(n, threads)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks stage0!(x, p1, Int32(n))
        @cuda threads=threads blocks=blocks stage1!(p1, p2, Int32(n))
        @cuda threads=threads blocks=blocks stage2!(p2, y, Int32(n))
    end
    CUDA.synchronize()

    @printf("Average kernel execution time: %.3f (ms)\n", (time_ns() - start) * 1.0e-6 / repeat)
    checksum = sum(Array(y))
    @printf("checksum = %lf\n", checksum)
    return 0
end

exit(main(ARGS))
