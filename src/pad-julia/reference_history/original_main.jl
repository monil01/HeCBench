using CUDA
using Printf
using Random

const REGS = 32

struct Params
    n_gpu_threads::Int
    n_gpu_blocks::Int
    n_threads::Int
    n_warmup::Int
    n_reps::Int
    alpha::Float64
    m::Int
    n::Int
    pad::Int
end

function usage()
    println("Usage: ", PROGRAM_FILE, " [options]")
    println("  -i <I> device threads per block")
    println("  -g <G> device blocks")
    println("  -t <T> host threads")
    println("  -w <W> warmup iterations")
    println("  -r <R> timed repetitions")
    println("  -a <A> fraction processed on host")
    println("  -m <M> rows")
    println("  -n <N> columns")
    println("  -e <B> extra padded columns")
    exit(1)
end

function parse_args(args)
    p = Dict{String,Any}(
        "i" => 256, "g" => 8, "t" => 4, "w" => 5, "r" => 1000,
        "a" => 0.1, "m" => 1000, "n" => 999, "e" => 1)
    i = 1
    while i <= length(args)
        arg = args[i]
        arg == "-h" && usage()
        startswith(arg, "-") || usage()
        key = arg[2:end]
        haskey(p, key) || usage()
        i < length(args) || usage()
        if key == "a"
            p[key] = parse(Float64, args[i + 1])
        else
            p[key] = parse(Int, args[i + 1])
        end
        i += 2
    end
    0.0 <= p["a"] <= 1.0 || error("Illegal value for -a")
    p["i"] > 0 && p["g"] > 0 && p["t"] > 0 || error("worker counts must be positive")
    Params(p["i"], p["g"], p["t"], p["w"], p["r"], p["a"], p["m"], p["n"], p["e"])
end

function padding_kernel!(out, matrix, n::Int32, m::Int32, pad::Int32, total::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    step = blockDim().x * gridDim().x
    idx0 = tid0
    width = n + pad
    while idx0 < total
        row = idx0 ÷ width
        col = idx0 - row * width
        val = if row < m && col < n
            @inbounds matrix[row * n + col + Int32(1)]
        else
            0.0
        end
        @inbounds out[idx0 + Int32(1)] = val
        idx0 += step
    end
    return
end

function cpu_padding(input::Vector{Float64}, n::Int, m::Int, n_pad::Int)
    out = Vector{Float64}(undef, m * n_pad)
    @inbounds for row in 0:(m - 1), col in 0:(n_pad - 1)
        out[row * n_pad + col + 1] = col < n ? input[row * n + col + 1] : 0.0
    end
    return out
end

function compare_output(outp, ref)
    sum_delta = 0.0
    sum_ref = 0.0
    @inbounds for i in eachindex(ref)
        sum_delta += abs(outp[i] - ref[i])
        sum_ref += abs(ref[i])
    end
    l1norm = sum_delta / sum_ref
    ok = l1norm < 1.0e-6
    println(ok ? "Test Passed" : "Test failed")
    return ok
end

function main()
    p = parse_args(ARGS)
    CUDA.allowscalar(false)
    rng = MersenneTwister(123)
    in_size = p.m * p.n
    out_size = p.m * (p.n + p.pad)
    h_input = rand(rng, Float64, in_size)
    h_out = zeros(Float64, out_size)
    d_input = CuArray(h_input)
    d_out = CuArray(h_out)

    total_iters = p.n_warmup + p.n_reps
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:total_iters
        CUDA.fill!(d_out, 0.0)
        @cuda threads=p.n_gpu_threads blocks=p.n_gpu_blocks padding_kernel!(
            d_out, d_input, Int32(p.n), Int32(p.m), Int32(p.pad), Int32(out_size))
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6
    @printf("Total padding execution time for %d iterations: %f (ms)\n", total_iters, elapsed_ms)
    h_out = Array(d_out)
    ref = cpu_padding(h_input, p.n, p.m, p.n + p.pad)
    ok = compare_output(h_out, ref)
    ok || exit(1)
end

main()
