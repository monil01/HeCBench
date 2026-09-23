using CUDA
using Printf
using Random

const REGS = 32
const REMOVE_VALUE = Int32(0)

mutable struct Params
    n_gpu_threads::Int
    n_gpu_blocks::Int
    n_threads::Int
    n_warmup::Int
    n_reps::Int
    alpha::Float64
    in_size::Int
    compaction_factor::Int
end

divceil(n, m) = (n - 1) ÷ m + 1

function parse_args()
    p = Params(256, 1024, 4, 5, 100, 0.1, 8_388_608, 50)
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "-h"
            println("Usage:  ./sc [options]")
            exit(0)
        elseif arg == "-i"
            i += 1; p.n_gpu_threads = parse(Int, ARGS[i])
        elseif arg == "-g"
            i += 1; p.n_gpu_blocks = parse(Int, ARGS[i])
        elseif arg == "-t"
            i += 1; p.n_threads = parse(Int, ARGS[i])
        elseif arg == "-w"
            i += 1; p.n_warmup = parse(Int, ARGS[i])
        elseif arg == "-r"
            i += 1; p.n_reps = parse(Int, ARGS[i])
        elseif arg == "-a"
            i += 1; p.alpha = parse(Float64, ARGS[i])
        elseif arg == "-n"
            i += 1; p.in_size = parse(Int, ARGS[i])
        elseif arg == "-c"
            i += 1; p.compaction_factor = parse(Int, ARGS[i])
        else
            println("\nUnrecognized option!")
            exit(0)
        end
        i += 1
    end
    return p
end

function read_input(p::Params)
    input = fill(REMOVE_VALUE, p.in_size)
    target = (p.in_size * p.compaction_factor) ÷ 100
    rng = MersenneTwister(123)
    remaining = target
    while remaining > 0
        x = rand(rng, 0:(p.in_size - 1))
        idx = x + 1
        if input[idx] == REMOVE_VALUE
            input[idx] = Int32(x + 2)
            remaining -= 1
        end
    end
    return input
end

function scatter_kernel!(output, input, flags, prefix, value::Int32, n::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    i = tid
    while i <= n
        keep = flags[i]
        if keep != Int32(0)
            pos = prefix[i]
            output[pos] = input[i]
        end
        i += stride
    end
    return
end

function compact_gpu(input::CuArray{Int32}, output::CuArray{Int32}, p::Params)
    flags = Int32.(input .!= REMOVE_VALUE)
    prefix = accumulate(+, flags)
    @cuda blocks=p.n_gpu_blocks threads=p.n_gpu_threads scatter_kernel!(
        output, input, flags, prefix, REMOVE_VALUE, Int32(length(input)))
    return output
end

function cpu_streamcompaction(input::Vector{Int32}, value::Int32)
    pos = 1
    for x in input
        if x != value
            input[pos] = x
            pos += 1
        end
    end
    return input
end

function verify(input::Vector{Int32}, backup::Vector{Int32}, p::Params)
    ref = cpu_streamcompaction(copy(backup), REMOVE_VALUE)
    size_compact = (p.in_size * p.compaction_factor) ÷ 100
    sum_delta = 0.0
    sum_ref = 0.0
    @inbounds for i in 1:size_compact
        sum_delta += abs(Float64(input[i] - ref[i]))
        sum_ref += abs(Float64(ref[i]))
    end
    if sum_ref == 0
        sum_ref = 1
    end
    if sum_delta / sum_ref >= 1e-6
        println("Test failed")
    else
        println("Test Passed")
    end
end

function main()
    p = parse_args()
    n_tasks = divceil(p.in_size, p.n_gpu_threads * REGS)
    padded = n_tasks * p.n_gpu_threads * REGS

    host = Vector{Int32}(undef, padded)
    input = read_input(p)
    host[1:p.in_size] .= input
    if padded > p.in_size
        host[p.in_size+1:end] .= REMOVE_VALUE
    end
    backup = copy(host)

    d_input = CuArray(host)
    d_output = CUDA.zeros(Int32, padded)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:(p.n_warmup + p.n_reps)
        copyto!(d_input, backup)
        CUDA.fill!(d_output, REMOVE_VALUE)
        compact_gpu(d_input, d_output, p)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1e-6
    @printf("Total stream compaction time for %d iterations: %f (ms)\n",
            p.n_reps + p.n_warmup, elapsed_ms)

    result = Array(d_output)
    verify(result, backup, p)
    return 0
end

exit(main())
