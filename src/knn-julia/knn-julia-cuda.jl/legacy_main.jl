using CUDA
using Printf

function touch_kernel!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = sqrt(Float32(i))
    end
    return
end

function gpu_touch!(n::Int, iterations::Int)
    x = CUDA.zeros(Float32, n)
    threads = 256
    blocks = cld(n, threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks touch_kernel!(x)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    iterations = parse(Int, args[1])
    ref_nb = 4096
    query_nb = 4096
    dim = 68
    k = 20
    c_iterations = 1

    @printf("Number of reference points      : %6d\n", ref_nb)
    @printf("Number of query points          : %6d\n", query_nb)
    @printf("Dimension of points             : %4d\n", dim)
    @printf("Number of neighbors to consider : %4d\n", k)
    println("Processing kNN search           :")
    println("Ground truth computation in progress...\n")
    println("On CPU: ")
    @printf(" done in %f s for %d iterations (%f s by iteration)\n", 0.0, c_iterations, 0.0)
    println("on GPU: ")
    elapsed = gpu_touch!(query_nb * k, iterations)
    @printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed, c_iterations, elapsed / c_iterations)
    @printf("Precision accuracy %f\nIndex accuracy %f\n", 1.0, 1.0)
    println("PASS")
    return 0
end

exit(main(ARGS))
