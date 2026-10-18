using CUDA
using Printf

function kernel_small(args::NTuple{16, UInt8})
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i == Int32(0)
        return
    end
    return
end

function kernel_medium(args::NTuple{256, UInt8})
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i == Int32(0)
        return
    end
    return
end

function kernel_large(args::NTuple{4096, UInt8})
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i == Int32(0)
        return
    end
    return
end

function time_kernel(label::String, kernel, args, repeat::Int)
    for _ in 1:repeat
        @cuda threads=1 blocks=1 kernel(args)
    end
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=1 blocks=1 kernel(args)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time of %s: %f (us)\n", label, elapsed_us)
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    small_args = ntuple(_ -> UInt8(0), 16)
    medium_args = ntuple(_ -> UInt8(0), 256)
    large_args = ntuple(_ -> UInt8(0), 4096)

    time_kernel("kernelWithSmallArgs", kernel_small, small_args, repeat)
    time_kernel("kernelWithMediumArgs", kernel_medium, medium_args, repeat)
    time_kernel("kernelWithLargeArgs", kernel_large, large_args, repeat)
    return 0
end

exit(main(ARGS))
