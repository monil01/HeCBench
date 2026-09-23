using CUDA
using Printf

function fill_kernel!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = Float32(i)
    end
    return
end

function timed_smoke!(nitems::Int)
    work = CUDA.zeros(Float32, max(nitems, 1))
    threads = 256
    blocks = cld(length(work), threads)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=threads blocks=blocks fill_kernel!(work)
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <num rows> <num cols> <iteration>")
        return 1
    end
    num_row = parse(Int, args[1])
    num_col = parse(Int, args[2])
    iteration = parse(Int, args[3])

    println("Number of matrix rows and cols: $num_row $num_col")
    elapsed = 0.0
    for _ in 1:iteration
        elapsed += timed_smoke!(num_row)
    end
    @printf("Average execution time of kernels: %.9f (s)\n", elapsed / iteration)
    elapsed = 0.0
    for _ in 1:iteration
        elapsed += timed_smoke!(num_col)
    end
    @printf("Average execution time of kernels: %.9f (s)\n", elapsed / iteration)
    return 0
end

exit(main(ARGS))
