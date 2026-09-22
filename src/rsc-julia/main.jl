using CUDA
using Printf

struct Params
    n_gpu_threads::Int
    n_gpu_blocks::Int
    n_threads::Int
    n_warmup::Int
    n_reps::Int
    file_name::String
    max_iter::Int
    error_threshold::Int
    convergence_threshold::Float32
end

function parse_args(argv)
    p = Dict{String,Any}(
        "i" => 256,
        "g" => 64,
        "t" => 1,
        "w" => 5,
        "r" => 1000,
        "f" => "../rsc-cuda/input/vectors.csv",
        "m" => 2000,
        "e" => 3,
        "c" => Float32(0.75),
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "-h"
            println("Usage: julia main.jl [-i threads] [-g blocks] [-t host_threads] [-w warmup] [-r reps] [-f file] [-m max_iter] [-e error] [-c convergence]")
            exit(0)
        elseif startswith(arg, "-") && length(arg) == 2
            key = arg[2:2]
            i += 1
            i <= length(argv) || error("missing value for $arg")
            if key in ("i", "g", "t", "w", "r", "m", "e")
                p[key] = parse(Int, argv[i])
            elseif key == "c"
                p[key] = parse(Float32, argv[i])
            elseif key == "f"
                p[key] = argv[i]
            else
                error("unrecognized option $arg")
            end
        else
            error("unrecognized option $arg")
        end
        i += 1
    end
    return Params(p["i"], p["g"], p["t"], p["w"], p["r"], p["f"],
                  p["m"], p["e"], Float32(p["c"]))
end

function resolve_input(path::String)
    isfile(path) && return path
    alt = joinpath(@__DIR__, "..", "rsc-cuda", path)
    isfile(alt) && return alt
    alt2 = joinpath(@__DIR__, "..", "rsc-cuda", "input", basename(path))
    isfile(alt2) && return alt2
    error("Error opening file!")
end

function read_input(path::String, max_iter::Int)
    file = resolve_input(path)
    lines = readlines(file)
    n = parse(Int, strip(lines[1]))
    xs = Vector{Int32}(undef, n)
    ys = Vector{Int32}(undef, n)
    vxs = Vector{Int32}(undef, n)
    vys = Vector{Int32}(undef, n)
    for i in 1:n
        parts = split(strip(lines[i + 1]), ',')
        length(parts) == 4 || error("Error: inconsistent file data!")
        xs[i] = parse(Int32, parts[1])
        ys[i] = parse(Int32, parts[2])
        vxs[i] = parse(Int32, parts[3])
        vys[i] = parse(Int32, parts[4])
    end
    ccall(:srand, Cvoid, (Cuint,), Cuint(123))
    random_numbers = Vector{Int32}(undef, 2 * max_iter)
    for i in eachindex(random_numbers)
        random_numbers[i] = Int32(mod(ccall(:rand, Cint, ()), n))
    end
    return xs, ys, vxs, vys, random_numbers
end

function gen_model_param(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
    temp = Float32((vx1 * (vx1 - (2 * vx2))) + (vx2 * vx2) + (vy1 * vy1) - (vy2 * ((2 * vy1) - vy2)))
    temp == 0.0f0 && return false, ntuple(_ -> -2011.0f0, 4)
    p0 = Float32(((vx1 * ((-vx2 * x1) + (vx1 * x2) - (vx2 * x2) + (vy2 * y1) - (vy2 * y2))) +
                  (vy1 * ((-vy2 * x1) + (vy1 * x2) - (vy2 * x2) - (vx2 * y1) + (vx2 * y2))) +
                  (x1 * ((vy2 * vy2) + (vx2 * vx2)))) / temp)
    p1 = Float32(((vx2 * ((vy1 * x1) - (vy1 * x2) - (vx1 * y1) + (vx2 * y1) - (vx1 * y2))) +
                  (vy2 * ((-vx1 * x1) + (vx1 * x2) - (vy1 * y1) + (vy2 * y1) - (vy1 * y2))) +
                  (y2 * ((vx1 * vx1) + (vy1 * vy1)))) / temp)

    temp = Float32((x1 * (x1 - (2 * x2))) + (x2 * x2) + (y1 * (y1 - (2 * y2))) + (y2 * y2))
    temp == 0.0f0 && return false, ntuple(_ -> -2011.0f0, 4)
    p2 = Float32((((x1 - x2) * (vx1 - vx2)) + ((y1 - y2) * (vy1 - vy2))) / temp)
    p3 = Float32((((x1 - x2) * (vy1 - vy2)) + ((y2 - y1) * (vx1 - vx2))) / temp)
    return true, (p0, p1, p2, p3)
end

function fit_models!(model_params, xs, ys, vxs, vys, random_numbers, max_iter)
    n = length(xs)
    fill!(model_params, -2011.0f0)
    @inbounds for iter0 in 0:max_iter-1
        r1 = Int(random_numbers[2 * iter0 + 1]) + 1
        r2 = Int(random_numbers[2 * iter0 + 2]) + 1
        vx1 = vxs[r1] - xs[r1]
        vy1 = vys[r1] - ys[r1]
        vx2 = vxs[r2] - xs[r2]
        vy2 = vys[r2] - ys[r2]
        ok, params = gen_model_param(xs[r1], ys[r1], vx1, vy1, xs[r2], ys[r2], vx2, vy2)
        if ok
            base = 4 * iter0
            model_params[base + 1] = params[1]
            model_params[base + 2] = params[2]
            model_params[base + 3] = params[3]
            model_params[base + 4] = params[4]
        end
    end
    return model_params
end

function trunc_i32(x)
    return Int32(trunc(x))
end

function ransac_kernel!(model_params, xs, ys, vxs, vys, outlier_counts,
                        n::Int32, max_iter::Int32, error_threshold::Int32)
    tx = threadIdx().x
    iter = blockIdx().x
    counts = @cuDynamicSharedMem(Int32, blockDim().x)
    while iter <= max_iter
        base = (iter - Int32(1)) * Int32(4)
        p0 = @inbounds model_params[base + Int32(1)]
        p1 = @inbounds model_params[base + Int32(2)]
        p2 = @inbounds model_params[base + Int32(3)]
        p3 = @inbounds model_params[base + Int32(4)]
        local_count = ifelse(p0 == -2011.0f0 && tx == 1, n, Int32(0))
        if p0 != -2011.0f0
            i = tx
            while i <= n
                x = @inbounds xs[i]
                y = @inbounds ys[i]
                vx = @inbounds vxs[i]
                vy = @inbounds vys[i]
                vx_error = Float32(x) + Float32(trunc_i32((Float32(x) - p0) * p2) -
                           trunc_i32((Float32(y) - p1) * p3)) - Float32(vx)
                vy_error = Float32(y) + Float32(trunc_i32((Float32(y) - p1) * p2) +
                           trunc_i32((Float32(x) - p0) * p3)) - Float32(vy)
                if abs(vx_error) >= Float32(error_threshold) || abs(vy_error) >= Float32(error_threshold)
                    local_count += Int32(1)
                end
                i += blockDim().x
            end
        end
        @inbounds counts[tx] = local_count
        sync_threads()
        stride = blockDim().x >>> 1
        while stride > 0
            if tx <= stride
                @inbounds counts[tx] += counts[tx + stride]
            end
            sync_threads()
            stride >>>= 1
        end
        if tx == 1
            @inbounds outlier_counts[iter] = counts[1]
        end
        iter += gridDim().x
    end
    return
end

function collect_candidates(outlier_counts, n::Int, convergence_threshold::Float32)
    model_candidate = Int32[]
    outliers_candidate = Int32[]
    limit = Float32(n) * convergence_threshold
    @inbounds for iter0 in 0:length(outlier_counts)-1
        c = outlier_counts[iter0 + 1]
        if Float32(c) < limit
            push!(model_candidate, Int32(iter0))
            push!(outliers_candidate, c)
        end
    end
    return model_candidate, outliers_candidate
end

function best_model(model_candidate, outliers_candidate, n)
    best_model = Int32(-1)
    best_outliers = Int32(n)
    @inbounds for i in eachindex(model_candidate)
        if outliers_candidate[i] < best_outliers
            best_outliers = outliers_candidate[i]
            best_model = model_candidate[i]
        end
    end
    return best_model, best_outliers
end

function verify_reference(xs, ys, vxs, vys, random_numbers, max_iter, error_threshold, convergence_threshold,
                          candidates, best_outliers)
    model_params = Vector{Float32}(undef, 4 * max_iter)
    fit_models!(model_params, xs, ys, vxs, vys, random_numbers, max_iter)
    counts = Vector{Int32}(undef, max_iter)
    n = length(xs)
    @inbounds for iter0 in 0:max_iter-1
        base = 4 * iter0
        if model_params[base + 1] == -2011.0f0
            counts[iter0 + 1] = n
            continue
        end
        outliers = Int32(0)
        p0 = model_params[base + 1]
        p1 = model_params[base + 2]
        p2 = model_params[base + 3]
        p3 = model_params[base + 4]
        for i in 1:n
            vx_error = Float32(xs[i]) + Float32(trunc_i32((Float32(xs[i]) - p0) * p2) -
                       trunc_i32((Float32(ys[i]) - p1) * p3)) - Float32(vxs[i])
            vy_error = Float32(ys[i]) + Float32(trunc_i32((Float32(ys[i]) - p1) * p2) +
                       trunc_i32((Float32(xs[i]) - p0) * p3)) - Float32(vys[i])
            if abs(vx_error) >= Float32(error_threshold) || abs(vy_error) >= Float32(error_threshold)
                outliers += Int32(1)
            end
        end
        counts[iter0 + 1] = outliers
    end
    ref_candidates, ref_outliers = collect_candidates(counts, n, convergence_threshold)
    ref_best_model, ref_best_outliers = best_model(ref_candidates, ref_outliers, n)
    println("Best model (reference) ", ref_best_model)
    if candidates != length(ref_candidates)
        println("Test failed (counts mismatch)")
    elseif best_outliers != ref_best_outliers
        println("Test failed (outliers mismatch)")
    else
        println("Test Passed")
    end
end

function main(argv)
    p = parse_args(argv)
    xs, ys, vxs, vys, random_numbers = read_input(p.file_name, p.max_iter)
    n = length(xs)
    model_params = Vector{Float32}(undef, 4 * p.max_iter)
    outlier_counts = Vector{Int32}(undef, p.max_iter)

    d_xs = CuArray(xs)
    d_ys = CuArray(ys)
    d_vxs = CuArray(vxs)
    d_vys = CuArray(vys)
    d_model_params = CuArray(model_params)
    d_outlier_counts = CuArray(outlier_counts)

    best_model_id = Int32(-1)
    best_outliers = Int32(n)
    candidates = 0

    CUDA.synchronize()
    start_ns = time_ns()
    for _ in 1:(p.n_warmup + p.n_reps)
        fit_models!(model_params, xs, ys, vxs, vys, random_numbers, p.max_iter)
        copyto!(d_model_params, model_params)
        fill!(d_outlier_counts, Int32(n))
        @cuda threads=p.n_gpu_threads blocks=p.n_gpu_blocks shmem=p.n_gpu_threads*sizeof(Int32) ransac_kernel!(
            d_model_params, d_xs, d_ys, d_vxs, d_vys, d_outlier_counts,
            Int32(n), Int32(p.max_iter), Int32(p.error_threshold))
        CUDA.synchronize()
        copyto!(outlier_counts, d_outlier_counts)
        model_candidate, outliers_candidate = collect_candidates(outlier_counts, n, p.convergence_threshold)
        candidates = length(model_candidate)
        best_model_id, best_outliers = best_model(model_candidate, outliers_candidate, n)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start_ns) * 1.0e-6
    @printf("Total task execution time for %d iterations: %f (ms)\n", p.n_reps + p.n_warmup, elapsed_ms)
    println("Best model (test) ", best_model_id)
    verify_reference(xs, ys, vxs, vys, random_numbers, p.max_iter, p.error_threshold,
                     p.convergence_threshold, candidates, best_outliers)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
