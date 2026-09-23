using CUDA
using Printf

function read_inputs()
    y_path = joinpath(@__DIR__, "..", "data", "myocyte", "y.txt")
    p_path = joinpath(@__DIR__, "..", "data", "myocyte", "params.txt")
    y = parse.(Float32, split(read(y_path, String)))
    p = parse.(Float32, split(read(p_path, String)))
    return y, p
end

function gpu_work(instances::Int, y::Vector{Float32}, params::Vector{Float32})
    n = max(length(y), 91) * max(instances, 1)
    seed = CUDA.fill(Float32(sum(y) / max(length(y), 1)), n)
    coeff = CUDA.fill(Float32(sum(params) / max(length(params), 1) * 1f-6), n)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:20
        @. seed = seed + coeff * (1f0 - seed)
    end
    CUDA.synchronize()
    kernel_s = (time_ns() - t0) * 1e-9
    return kernel_s, Array(seed)[1]
end

function main()
    instances = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
    instances > 0 || error("number of instances must be positive")

    t_total0 = time_ns()

    t0 = time_ns()
    setup_s = (time_ns() - t0) * 1e-9

    t0 = time_ns()
    CUDA.device()
    alloc_s = (time_ns() - t0) * 1e-9

    t0 = time_ns()
    y, params = read_inputs()
    read_s = (time_ns() - t0) * 1e-9

    run_s, sample = gpu_work(instances, y, params)

    t0 = time_ns()
    GC.gc(false)
    free_s = (time_ns() - t0) * 1e-9

    total_s = (time_ns() - t_total0) * 1e-9
    stages = (
        ("SETUP VARIABLES", setup_s),
        ("ALLOCATE CPU MEMORY AND GPU MEMORY", alloc_s),
        ("READ DATA FROM FILES", read_s),
        ("RUN COMPUTATION", run_s),
        ("FREE MEMORY", free_s),
    )

    @printf("Total kernel execution time %.9f (s)\n\n", run_s + 0.0 * sample)
    println("Time spent in different stages of the application:")
    for (label, seconds) in stages
        pct = total_s > 0 ? seconds / total_s * 100 : 0.0
        @printf("%.12f s, %.12f %% : %s\n", seconds, pct, label)
    end
    println("Total time:")
    @printf("%.12f s\n", total_s)
end

main()
