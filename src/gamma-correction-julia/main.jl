using CUDA
using Printf

function gamma_value(v::UInt8)
    x = Float32(v) / 255.0f0
    y = unsafe_trunc(Int32, 255.0f0 * x * x)
    y = ifelse(y > 255, 255, y)
    y = ifelse(y < 0, 0, y)
    return UInt8(y)
end

function gamma_kernel!(pixel, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds pixel[i] = gamma_value(pixel[i])
    end
    return
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <image width> <image height> <block size> <repeat>")
        return 1
    end
    width = parse(Int, args[1])
    height = parse(Int, args[2])
    block_size = parse(Int, args[3])
    repeat = parse(Int, args[4])
    n = width * height

    input = Vector{UInt8}(undef, n)
    for i in 1:n
        input[i] = UInt8((i - 1) % 256)
    end
    reference = gamma_value.(input)
    d_input = CuArray(input)

    grids = max(1, cld(n, block_size))
    @cuda threads=block_size blocks=grids gamma_kernel!(d_input, Int32(n))
    CUDA.synchronize()

    total_ns = 0
    for _ in 1:repeat
        copyto!(d_input, input)
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=block_size blocks=grids gamma_kernel!(d_input, Int32(n))
        CUDA.synchronize()
        total_ns += time_ns() - start
    end
    @printf("Average kernel execution time %f (s)\n", total_ns * 1.0e-9 / repeat)

    println(Array(d_input) == reference ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
