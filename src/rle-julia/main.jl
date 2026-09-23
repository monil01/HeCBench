using CUDA
using Printf
using Random

const THREADS = 256

function parse_args()
    num_items = -1
    timing_iterations = 0
    verbose = false
    for arg in ARGS
        if arg == "--help"
            println("main.jl [--n=<input items> [--i=<timing iterations> [--v]")
            exit(0)
        elseif arg == "--v"
            verbose = true
        elseif startswith(arg, "--n=")
            num_items = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--i=")
            timing_iterations = parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return num_items < 0 ? 1_000_000 : num_items, timing_iterations, verbose
end

function initialize_input(num_items::Int, entropy_reduction::Int, max_segment::Int)
    h_in = Vector{Int32}(undef, num_items)
    max_int = typemax(UInt32)
    key = Int32(0)
    i = 1
    while i <= num_items
        repeat = if max_segment < 0
            num_items
        elseif max_segment < 2
            1
        else
            rng = MersenneTwister(19937 + (i - 1))
            r = rand(rng, UInt32) & UInt32(typemax(Int32))
            for _ in 1:entropy_reduction
                r &= rand(rng, UInt32) & UInt32(typemax(Int32))
            end
            max(1, Int(floor(Float64(r) * Float64(max_segment) / Float64(max_int))))
        end
        stop = min(i + repeat - 1, num_items)
        h_in[i:stop] .= key
        i = stop + 1
        key += Int32(1)
    end
    return h_in
end

function solve_cpu(h_in::Vector{Int32})
    num_items = length(h_in)
    if num_items == 0
        return Int32[], Int32[], 0
    end
    unique = Int32[]
    lengths = Int32[]
    previous = h_in[1]
    len = Int32(1)
    for i in 2:num_items
        if h_in[i] != previous
            push!(unique, previous)
            push!(lengths, len)
            previous = h_in[i]
            len = Int32(1)
        else
            len += Int32(1)
        end
    end
    push!(unique, previous)
    push!(lengths, len)
    return unique, lengths, length(unique)
end

function run_start_kernel!(flags, input, n::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    i = tid
    while i <= n
        flags[i] = (i == Int32(1) || input[i] != input[i - Int32(1)]) ? Int32(1) : Int32(0)
        i += stride
    end
    return
end

function scatter_runs_kernel!(unique_out, lengths_out, input, flags, prefix, n::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    i = tid
    while i <= n
        if flags[i] == Int32(1)
            run = prefix[i]
            unique_out[run] = input[i]
            if run > Int32(1)
                prev_start = find_prev_start(flags, i - Int32(1))
                lengths_out[run - Int32(1)] = i - prev_start
            end
        end
        i += stride
    end
    return
end

function finish_lengths_kernel!(lengths_out, flags, prefix, n::Int32)
    if threadIdx().x == Int32(1) && blockIdx().x == Int32(1)
        num_runs = prefix[n]
        last_start = find_prev_start(flags, n)
        lengths_out[num_runs] = n - last_start + Int32(1)
    end
    return
end

function find_prev_start(flags, i::Int32)
    while i > Int32(1) && flags[i] == Int32(0)
        i -= Int32(1)
    end
    return i
end

function encode_gpu(d_in, num_items::Int)
    blocks = max(1, cld(num_items, THREADS))
    flags = CUDA.zeros(Int32, num_items)
    @cuda blocks=blocks threads=THREADS run_start_kernel!(flags, d_in, Int32(num_items))
    prefix = accumulate(+, flags)
    num_runs = Int(Array(prefix[num_items:num_items])[1])
    d_unique = CUDA.zeros(Int32, num_items)
    d_lengths = CUDA.zeros(Int32, num_items)
    @cuda blocks=blocks threads=THREADS scatter_runs_kernel!(
        d_unique, d_lengths, d_in, flags, prefix, Int32(num_items))
    @cuda blocks=1 threads=1 finish_lengths_kernel!(d_lengths, flags, prefix, Int32(num_items))
    return d_unique, d_lengths, num_runs
end

function run_test(h_in::Vector{Int32}, timing_iterations::Int)
    ref_unique, ref_lengths, num_runs = solve_cpu(h_in)
    d_in = CuArray(h_in)
    d_unique, d_lengths, got_runs = encode_gpu(d_in, length(h_in))
    unique = Array(d_unique)[1:num_runs]
    lengths = Array(d_lengths)[1:num_runs]
    compare0 = unique == ref_unique ? 0 : 1
    compare1 = lengths == ref_lengths ? 0 : 1
    compare2 = got_runs == num_runs ? 0 : 1
    println("\t Keys ", compare0 != 0 ? "FAIL" : "PASS")
    println("\t Lengths ", compare1 != 0 ? "FAIL" : "PASS")
    println("\t Count ", compare2 != 0 ? "FAIL" : "PASS")

    if timing_iterations > 0
        CUDA.synchronize()
        start = time_ns()
        encode_gpu(d_in, length(h_in))
        CUDA.synchronize()
        avg_ms = (time_ns() - start) * 1e-6
        giga_rate = Float64(length(h_in)) / avg_ms / 1000.0 / 1000.0
        bytes_moved = length(h_in) * sizeof(Int32) + num_runs * (sizeof(Int32) + sizeof(Int32))
        giga_bandwidth = Float64(bytes_moved) / avg_ms / 1000.0 / 1000.0
        @printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s",
                avg_ms, giga_rate, giga_bandwidth)
    end
    println("\n")
    return compare0 == 0 && compare1 == 0 && compare2 == 0
end

function test_iterator(num_items::Int, timing_iterations::Int)
    h_in = fill(Int32(1), num_items)
    _, _, num_runs = solve_cpu(h_in)
    @printf("\nTest iterator: on %d items, %d segments (avg run length %.3f), {i key, i offset, i length}\n",
            num_items, num_runs, Float64(num_items) / num_runs)
    return run_test(h_in, timing_iterations)
end

function test_pointer(num_items::Int, entropy_reduction::Int, max_segment::Int, timing_iterations::Int)
    h_in = initialize_input(num_items, entropy_reduction, max_segment)
    _, _, num_runs = solve_cpu(h_in)
    @printf("\nTest pointer: %d items, %d segments (avg run length %.3f), {i key, i offset, i length}, max_segment %d, entropy_reduction %d\n",
            num_items, num_runs, Float64(num_items) / num_runs, max_segment, entropy_reduction)
    return run_test(h_in, timing_iterations)
end

function main()
    num_items, timing_iterations, _ = parse_args()
    println()
    ok = test_iterator(num_items, timing_iterations)
    max_seg_limit = min(num_items, 1 << 16)
    entropy_reduction = 0
    max_segment = 1
    while max_segment <= max_seg_limit
        ok &= test_pointer(num_items, entropy_reduction, max(1, max_segment), timing_iterations)
        max_segment <<= 4
        entropy_reduction += 1
    end
    return ok ? 0 : 1
end

exit(main())
