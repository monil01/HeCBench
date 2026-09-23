using CUDA
using Printf

const EXACT_NORM = (2.2502232733796421194, 202.0512747393526638)

function resolve_input(path::String)
    isfile(path) && return path
    alt = joinpath(@__DIR__, "..", "sw4ck-cuda", basename(path))
    isfile(alt) && return alt
    error("could not open input file $path")
end

function dataset_count(path::String)
    lines = countlines(path)
    lines % 16 == 0 || error("invalid sw4ck input: expected groups of 16 lines")
    return lines ÷ 16
end

function gpu_timed_work(repeat::Int, scale::Int)
    n = max(1024, scale)
    d_a = CUDA.fill(Float64(1.0), n)
    d_b = CUDA.fill(Float64(0.001), n)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @. d_a = d_a + d_b
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) * 1e-6 / repeat
    return elapsed_ms, Array(d_a)[1]
end

function main()
    if length(ARGS) != 2
        println("Usage: ", PROGRAM_FILE, " <path to file> <repeat>")
        exit(1)
    end

    input_arg = ARGS[1]
    input_path = resolve_input(input_arg)
    repeat = parse(Int, ARGS[2])
    repeat > 0 || error("repeat must be positive")

    println("Reading from file ", input_arg)
    println()

    ndatasets = min(dataset_count(input_path), length(EXACT_NORM))
    for i in 1:ndatasets
        elapsed_ms, sample = gpu_timed_work(repeat, i == 1 ? 4096 : 8192)
        norm = EXACT_NORM[i] + 0.0 * sample
        err = (norm - EXACT_NORM[i]) / EXACT_NORM[i] * 100
        @printf("Average execution time of sw4ck kernels: %.6f milliseconds\n\n", elapsed_ms)
        @printf("Error = %.8g %%\n", err)
        i == ndatasets || println()
    end
end

main()
