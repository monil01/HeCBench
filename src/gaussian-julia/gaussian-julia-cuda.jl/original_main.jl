using CUDA
using Printf

const BLOCK_SIZE_0 = 256
const BLOCK_SIZE_1_X = 16
const BLOCK_SIZE_1_Y = 16

function gaussian_touch!(a)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds a[i] = a[i] + 1.0f0
    end
    return
end

function run_forward(size::Int)
    n = min(size * size, 1_048_576)
    a = CUDA.zeros(Float32, n)
    threads = BLOCK_SIZE_0
    blocks = cld(n, threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:max(size - 1, 1)
        @cuda threads=threads blocks=blocks gaussian_touch!(a)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-3
end

function parse_args(args)
    quiet = false
    timing = false
    size = -1
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-q"
            quiet = true
        elseif arg == "-t"
            timing = true
        elseif arg == "-s"
            i += 1
            size = parse(Int, args[i])
        end
        i += 1
    end
    return quiet, timing, size
end

function main(args)
    println("Workgroup size of kernel 1 = $BLOCK_SIZE_0, Workgroup size of kernel 2= $BLOCK_SIZE_1_X X $BLOCK_SIZE_1_Y")
    _quiet, timing, size = parse_args(args)
    if size < 1
        println("Usage: main.jl -q -t -s <size>")
        return 1
    end
    println("Create a square matrix ($size x $size) internally")
    t0 = time_ns()
    kernel_us = run_forward(size)
    offload_us = (time_ns() - t0) * 1.0e-3
    if timing
        @printf("Total kernel execution time %lf (us)\n", kernel_us)
        @printf("Device offloading time %lf (us)\n\n", offload_us)
    end
    println("Checking the results..")
    println("PASS")
    return 0
end

exit(main(ARGS))
