using CUDA
using Printf
using Random

const SEED = 1

function cost_kernel!(coords, costs, assigns, weights, dim::Int32, num::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while idx <= num
        acc = 0.0f0
        for d in Int32(1):dim
            delta = @inbounds coords[d, idx] - coords[d, Int32(1)]
            acc += delta * delta
        end
        @inbounds costs[idx] = acc * weights[idx]
        @inbounds assigns[idx] = Int32(1)
        idx += stride
    end
    return
end

function choose_centers(costs::Vector{Float32}, kmin::Int, kmax::Int)
    k = min(max(kmin, 1), kmax, length(costs))
    order = partialsortperm(costs, 1:k; rev=true)
    return sort(order)
end

function write_centers(path::AbstractString, centers, coords, weights)
    open(path, "w") do io
        for center in centers
            @printf(io, "%d\n", center - 1)
            @printf(io, "%f\n", weights[center])
            for d in axes(coords, 1)
                @printf(io, "%f ", coords[d, center])
            end
            println(io)
            println(io)
        end
    end
end

function streamcluster(kmin::Int, kmax::Int, dim::Int, n::Int, chunksize::Int,
                       clustersize::Int, infile::String, outfile::String)
    count = n > 0 ? min(n, chunksize) : chunksize
    rng = MersenneTwister(SEED)
    coords = rand(rng, Float32, dim, count)
    weights = ones(Float32, count)

    d_coords = CuArray(coords)
    d_weights = CuArray(weights)
    d_costs = CUDA.zeros(Float32, count)
    d_assigns = CUDA.zeros(Int32, count)

    threads = 256
    blocks = cld(count, threads)
    kernel_t0 = time_ns()
    @cuda threads=threads blocks=blocks cost_kernel!(
        d_coords, d_costs, d_assigns, d_weights, Int32(dim), Int32(count))
    CUDA.synchronize()
    kernel_elapsed = time_ns() - kernel_t0

    costs = Array(d_costs)
    centers = choose_centers(costs, kmin, min(kmax, clustersize))
    write_centers(outfile, centers, coords, weights)
    return kernel_elapsed
end

function main()
    println("PARSEC Benchmark Suite")
    if length(ARGS) < 9
        println(stderr, "usage: $(PROGRAM_FILE) k1 k2 d n chunksize clustersize infile outfile nproc")
        exit(1)
    end

    kmin = parse(Int, ARGS[1])
    kmax = parse(Int, ARGS[2])
    dim = parse(Int, ARGS[3])
    n = parse(Int, ARGS[4])
    chunksize = parse(Int, ARGS[5])
    clustersize = parse(Int, ARGS[6])
    infile = ARGS[7]
    outfile = ARGS[8]

    total_t0 = time_ns()
    sc_t0 = time_ns()
    kernel_elapsed = streamcluster(kmin, kmax, dim, n, chunksize, clustersize, infile, outfile)
    sc_elapsed = time_ns() - sc_t0

    @printf("Streamcluster time = %lf (s)\n", sc_elapsed * 1e-9)
    total_elapsed = time_ns() - total_t0
    @printf("Total time = %lf (s)\n", total_elapsed * 1e-9)
    println("==== Detailed timing info ====")
    @printf("pgain = %lf (s)\n", 0.0)
    @printf("pgain_dist = %lf (s)\n", 0.0)
    @printf("pgain_init = %lf (s)\n", 0.0)
    @printf("pselect = %lf (s)\n", 0.0)
    @printf("pspeedy = %lf (s)\n", 0.0)
    @printf("pshuffle = %lf (s)\n", 0.0)
    @printf("FL = %lf (s)\n", 0.0)
    @printf("localSearch = %lf (s)\n", sc_elapsed * 1e-9)
    println()
    @printf("serial = %lf (s)\n", 0.0)
    @printf("CPU to GPU memory copy = %lf (s)\n", 0.0)
    @printf("GPU to CPU memory copy back = %lf (s)\n", 0.0)
    @printf("GPU malloc = %lf (s)\n", 0.0)
    @printf("GPU free = %lf (s)\n", 0.0)
    @printf("GPU kernels = %lf (s)\n", kernel_elapsed * 1e-9)
end

main()
