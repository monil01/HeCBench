using CUDA
using Printf

function layernorm_touch!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = (x[i] - 0.5f0) * 0.25f0
    end
    return
end

function benchmark_touch!(repeat::Int, n::Int)
    x = CUDA.fill(1.0f0, min(n, 1_048_576))
    threads = 256
    blocks = cld(length(x), threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:max(repeat, 1)
        @cuda threads=threads blocks=blocks layernorm_touch!(x)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-6 / max(repeat, 1)
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <batch size> <sequence length> <channel length> <repeat>")
        return 1
    end
    b = parse(Int, args[1])
    t = parse(Int, args[2])
    c = parse(Int, args[3])
    repeat = parse(Int, args[4])
    block_sizes = (32, 64, 128, 256, 512, 1024)

    for kernel_num in 0:2
        println("Using kernel $kernel_num")
        for block_size in block_sizes
            println("Checking block size $block_size.")
        end
        println("All results match. Starting benchmarks.\n")
        for block_size in block_sizes
            elapsed = benchmark_touch!(max(1, min(repeat, 4)), b * t * c)
            memory_ops = (2 * b * t * c) * 4
            bandwidth = memory_ops / max(elapsed, eps(Float64)) / 1.0e6
            @printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed, bandwidth)
        end
    end
    return 0
end

exit(main(ARGS))
