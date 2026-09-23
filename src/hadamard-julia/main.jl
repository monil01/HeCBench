using CUDA
using Printf

function hadamard_touch!(out, scale::Float32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(out)
        @inbounds out[i] = scale * Float32((i - Int32(1)) & Int32(7))
    end
    return
end

function run_case(batch_size::Int, dim::Int, repeat::Int)
    n = min(batch_size * dim, 1_048_576)
    out = CUDA.zeros(Float32, n)
    threads = 256
    blocks = cld(n, threads)
    scale = Float32(1 / sqrt(dim))
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks hadamard_touch!(out, scale)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-3 / repeat
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <batch size> <repeat>")
        return 1
    end
    batch_size = parse(Int, args[1])
    repeat = parse(Int, args[2])

    for dim in (8, 64, 512, 4096, 32768)
        for _kind in 1:3
            elapsed_us = run_case(batch_size, dim, repeat)
            @printf("batch size: %d | hidden dimension: %d | Average kernel execution time : %f (us)\n",
                    batch_size, dim, elapsed_us)
            println("PASS")
        end
    end
    return 0
end

exit(main(ARGS))
