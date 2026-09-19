using CUDA
using Printf
using Random

function float_to_uchar(x::Float32)
    v = unsafe_trunc(Int32, x)
    return UInt8(v & Int32(0xff))
end

function reference_kernel!(out, input, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while idx <= n
        @inbounds out[idx] = float_to_uchar(input[idx])
        idx += stride
    end
    return
end

function blockaccess_kernel!(out, input, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while idx <= n
        @inbounds out[idx] = float_to_uchar(input[idx])
        idx += stride
    end
    return
end

function time_kernel(label::String, kernel, out, input, n::Int, grid::Int, block::Int, repeat::Int)
    @cuda threads=block blocks=grid kernel(out, input, Int32(n))
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block blocks=grid kernel(out, input, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average execution time of the %s kernel: %f (us)\n",
            label, (time_ns() - start) * 1.0e-3 / repeat)
end

function main(args)
    if length(args) != 3
        println("Block access N elements where N is represented as rows x columns")
        println("Usage: main.jl <number of rows> <number of columns> <repeat>")
        return 1
    end
    nrows = parse(Int, args[1])
    ncols = parse(Int, args[2])
    repeat = parse(Int, args[3])
    n = cld(nrows * ncols, 4) * 4

    rng = MersenneTwister(19937)
    input = Float32.(randn(rng, n) .* 127.0f0 .- 128.0f0)
    d_input = CuArray(input)
    d_out = CUDA.zeros(UInt8, n)
    d_ref = CUDA.zeros(UInt8, n)

    block = 256
    grid = min(cld(n, block), 4096)

    @cuda threads=block blocks=grid reference_kernel!(d_ref, d_input, Int32(n))
    @cuda threads=block blocks=grid blockaccess_kernel!(d_out, d_input, Int32(n))
    CUDA.synchronize()
    println(Array(d_out) == Array(d_ref) ? "PASS" : "FAIL")

    time_kernel("reference", reference_kernel!, d_ref, d_input, n, grid, block, repeat)
    time_kernel("blockAccess", blockaccess_kernel!, d_out, d_input, n, grid, block, repeat)
    return 0
end

exit(main(ARGS))
