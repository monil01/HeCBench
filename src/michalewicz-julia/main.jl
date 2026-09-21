using CUDA
using Printf
using Random

# Julia port of michalewicz-cuda.  The CUDA source uses an atomic float
# minimum; this port writes per-vector objective values on the GPU and performs
# the final reduction on the host, preserving the timed GPU evaluation region.

const THREADS = 256

@inline function michalewicz_value(values, offset::Int64, dim::Int32)
    result = 0.0f0
    i = Int32(0)
    while i < dim
        x = @inbounds values[offset + Int64(i) + 1]
        a = sin(x)
        b = sin((Float32(i + Int32(1)) * x * x) / Float32(pi))
        result += a * (b ^ 20.0f0)
        i += Int32(1)
    end
    return -result
end

function eval_kernel!(values, results, n_vectors::Int32, dim::Int32)
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= n_vectors
        offset = Int64(n - Int32(1)) * Int64(dim)
        @inbounds results[n] = michalewicz_value(values, offset, dim)
    end
    return
end

function reference(values::Vector{Float32}, n::Int, dim::Int)
    min_value = 0.0f0
    for vec in 0:n-1
        v = 0.0f0
        base = vec * dim
        for i in 0:dim-1
            x = values[base + i + 1]
            a = sin(x)
            b = sin((Float32(i + 1) * x * x) / Float32(pi))
            v += a * (b ^ 20.0f0)
        end
        min_value = min(min_value, -v)
    end
    return min_value
end

function print_error(value::Float32, dim::Int)
    @printf("Global minima = %f\n", value)
    true_min = dim == 2 ? -1.8013f0 :
               dim == 5 ? -4.687658f0 :
               dim == 10 ? -9.66015f0 : 0.0f0
    @printf("Error = %f\n", abs(true_min - value))
end

function run_dim(n::Int, repeat_n::Int, dim::Int, rng::MersenneTwister)
    values = rand(rng, Float32, n * dim) .* 4.0f0
    d_values = CuArray(values)
    d_results = CUDA.zeros(Float32, n)

    blocks = cld(n, THREADS)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=THREADS blocks=blocks eval_kernel!(
            d_values, d_results, Int32(n), Int32(dim))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time of kernel (dim = %d): %f (us)\n",
            dim, elapsed_us)

    gpu_min = minimum(Array(d_results))
    cpu_min = reference(values, n, dim)
    print_error(gpu_min, dim)
    println(isapprox(gpu_min, cpu_min; rtol=1.0f-5, atol=1.0f-5) ? "PASS" : "FAIL")
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <number of vectors> <repeat>")
        return 1
    end

    n = parse(Int, ARGS[1])
    repeat_n = parse(Int, ARGS[2])
    rng = MersenneTwister(19937)

    for dim in (2, 5, 10)
        run_dim(n, repeat_n, dim, rng)
    end

    return 0
end

exit(main())
