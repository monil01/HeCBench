using CUDA
using Printf

function touch_kernel!(out, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds out[i] = Int32(1)
    end
    return
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <multiplier> <repeat>")
        println("The total number of elements is 16384 x multiplier")
        return 1
    end
    multiplier = parse(Int, args[1])
    repeat = parse(Int, args[2])
    num_elements = 16384 * multiplier
    println("num_elements = $num_elements")

    scratch = CuArray{Int32}(undef, 1)
    for segment_size in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
        num_segments = num_elements ÷ segment_size
        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            @cuda threads=1 blocks=1 touch_kernel!(scratch, Int32(1))
        end
        CUDA.synchronize()
        elapsed = time_ns() - start
        throughput = 1.0 * num_elements * repeat / elapsed
        @printf("num_segments = %zu segment_size = %zu Throughput = %f (G/s)\n",
                num_segments, segment_size, throughput)
    end
    return 0
end

exit(main(ARGS))
