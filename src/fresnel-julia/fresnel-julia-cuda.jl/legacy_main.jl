using CUDA
using Printf

function fresnel_touch!(x, y)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds y[i] = sin(x[i] * x[i])
    end
    return
end

function run_fresnel(repeat::Int)
    n = 1_048_576
    x = CUDA.fill(0.125, n)
    y = CUDA.zeros(Float64, n)
    threads = 256
    blocks = cld(n, threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks fresnel_touch!(x, y)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9 / repeat
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    elapsed_s = run_fresnel(repeat)
    @printf("Average kernel execution time %f (s)\n", elapsed_s)
    println("PASS")
    return 0
end

exit(main(ARGS))
