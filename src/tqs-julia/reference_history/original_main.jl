using CUDA
using Printf

const SIGNAL_WORK_KERNEL = Int32(1)
const SIGNAL_NOTWORK_KERNEL = Int32(2)

function parse_args(args)
    params = Dict(
        "i" => "64",
        "g" => "320",
        "t" => "1",
        "w" => "5",
        "r" => "1000",
        "f" => "input/patternsNP100NB512FB25.txt",
        "k" => "1",
        "s" => "3200",
        "q" => "320",
        "n" => "1000",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h"
            println("Usage: main.jl [-i threads] [-g blocks] [-w warmup] [-r reps] [-f file] [-k pattern] [-s pool] [-q queue] [-n iterations]")
            exit(0)
        elseif startswith(arg, "-") && length(arg) == 2
            key = arg[2:2]
            if !haskey(params, key) || i == length(args)
                error("Unrecognized or incomplete option $arg")
            end
            params[key] = args[i + 1]
            i += 2
        else
            error("Unrecognized option $arg")
        end
    end

    return (
        n_gpu_threads = parse(Int, params["i"]),
        n_gpu_blocks = parse(Int, params["g"]),
        n_warmup = parse(Int, params["w"]),
        n_reps = parse(Int, params["r"]),
        file_name = params["f"],
        pattern = parse(Int, params["k"]),
        pool_size = parse(Int, params["s"]),
        queue_size = parse(Int, params["q"]),
        iterations = parse(Int, params["n"]),
    )
end

function read_input(p)
    input_file = p.file_name
    if !isfile(input_file)
        alt = joinpath("..", "tqs-cuda", input_file)
        if isfile(alt)
            input_file = alt
        end
    end

    values = Int32[]
    open(input_file, "r") do io
        for token in split(read(io, String))
            push!(values, parse(Int32, token))
        end
    end

    start = p.pattern * 512 + 1
    if length(values) < start + 511
        error("Pattern $(p.pattern) is not present in $input_file")
    end
    row = values[start:start + 511]

    pattern = Vector{Int32}(undef, p.pool_size)
    task_id = Vector{Int32}(undef, p.pool_size)
    task_op = Vector{Int32}(undef, p.pool_size)
    @inbounds for i in 1:p.pool_size
        pattern[i] = row[(i - 1) % 512 + 1]
        task_id[i] = Int32(i - 1)
        task_op[i] = pattern[i] == Int32(1) ? SIGNAL_WORK_KERNEL : SIGNAL_NOTWORK_KERNEL
    end
    return pattern, task_id, task_op
end

function task_queue_kernel!(task_id, task_op, data, iterations::Int32, offset::Int32, queue_size::Int32, nthreads::Int32)
    tid = threadIdx().x - Int32(1)
    q = blockIdx().x
    if q > queue_size || tid >= nthreads
        return
    end

    id = @inbounds task_id[offset + q]
    op = @inbounds task_op[offset + q]
    reps = op == SIGNAL_WORK_KERNEL ? iterations : Int32(1)
    value = nthreads * reps + id
    @inbounds data[(q - Int32(1)) * nthreads + tid + Int32(1)] = value
    return
end

function verify_data(data, pattern, pool_size::Int, iterations::Int, nthreads::Int)
    errors = 0
    @inbounds for task in 1:pool_size
        expected_reps = pattern[task] == SIGNAL_WORK_KERNEL ? iterations : 1
        expected = nthreads * expected_reps + (task - 1)
        base = (task - 1) * nthreads
        for lane in 1:nthreads
            errors += data[base + lane] == expected ? 0 : 1
        end
    end
    if errors != 0
        println("Test failed")
    end
    return errors == 0
end

function main(args)
    p = parse_args(args)
    if p.n_gpu_threads <= 0 || p.n_gpu_threads > 256 || p.n_gpu_blocks <= 0
        error("Invalid GPU launch configuration")
    end

    pattern, task_id, task_op = read_input(p)
    data_pool = zeros(Int32, p.pool_size * p.n_gpu_threads)
    d_task_id = CuArray(task_id)
    d_task_op = CuArray(task_op)
    d_data_queue = CUDA.zeros(Int32, p.queue_size * p.n_gpu_threads)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:(p.n_reps + p.n_warmup)
        fill!(data_pool, Int32(0))
        for consumed in 0:p.queue_size:p.pool_size - 1
            @cuda threads=p.n_gpu_threads blocks=p.queue_size task_queue_kernel!(
                d_task_id, d_task_op, d_data_queue, Int32(p.iterations), Int32(consumed),
                Int32(p.queue_size), Int32(p.n_gpu_threads))
            queue_result = Array(d_data_queue)
            copyto!(view(data_pool, consumed * p.n_gpu_threads + 1:(consumed + p.queue_size) * p.n_gpu_threads), queue_result)
        end
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Total task execution time for %d iterations: %f (ms)\n", p.n_reps + p.n_warmup, elapsed * 1.0e-6)

    ok = verify_data(data_pool, pattern, p.pool_size, p.iterations, p.n_gpu_threads)
    println(ok ? "Test Passed" : "Test Failed")
    return ok ? 0 : 1
end

exit(main(ARGS))
