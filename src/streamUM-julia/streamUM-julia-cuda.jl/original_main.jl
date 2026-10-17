using CUDA
using Printf
using Random

const THREADS = 256

mutable struct TaskData
    size::Int
    id::Int
    data::Vector{Float64}
    result::Vector{Float64}
    vector::Vector{Float64}
end

function gemv_host!(result, data, vector, n)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:n
            acc += data[(i - 1) * n + j] * vector[j]
        end
        result[i] = acc
    end
end

function ref_gemv!(result, data, vector, n)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:n
            acc += data[(j - 1) * n + i] * vector[j]
        end
        result[i] = acc
    end
end

function gemv_column_kernel!(result, data, vector, n::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    while i <= n
        acc = 0.0
        @inbounds for j in Int32(1):n
            acc += data[(j - Int32(1)) * n + i] * vector[j]
        end
        @inbounds result[i] = acc
        i += stride
    end
    return
end

function initialise_tasks(n::Int)
    rng = MersenneTwister(48)
    tasks = Vector{TaskData}(undef, n)
    for i in 1:n
        sz = max(Int(floor(rand(rng) * 1000.0)), 64)
        data = rand(rng, Float64, sz * sz)
        result = zeros(Float64, sz)
        vector = rand(rng, Float64, sz)
        tasks[i] = TaskData(sz, i - 1, data, result, vector)
    end
    return tasks
end

function execute!(task::TaskData)
    n = task.size
    if n < 100
        gemv_host!(task.result, task.data, task.vector, n)
    else
        d_data = CuArray(task.data)
        d_vector = CuArray(task.vector)
        d_result = CUDA.zeros(Float64, n)
        blocks = max(cld(n, THREADS), 1)
        @cuda threads=THREADS blocks=blocks gemv_column_kernel!(d_result, d_data, d_vector, Int32(n))
        task.result .= Array(d_result)
    end
    return
end

function check(tasks)
    ok = true
    for task in tasks
        if task.size >= 100
            ref = zeros(Float64, task.size)
            ref_gemv!(ref, task.data, task.vector, task.size)
            if any(abs.(task.result .- ref) .> 1.0e-3)
                ok = false
                break
            end
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of host threads> <number of tasks> <verify>")
        return 1
    end
    nthreads = parse(Int, args[1])
    ntasks = parse(Int, args[2])
    verify = parse(Int, args[3])
    _ = nthreads
    tasks = initialise_tasks(ntasks)

    println("Executing tasks on host / device")
    CUDA.synchronize()
    t0 = time_ns()
    for task in tasks
        execute!(task)
    end
    CUDA.synchronize()
    @printf("Task execution time : %f (s)\n", (time_ns() - t0) * 1.0e-9)

    ok = true
    if verify != 0
        ok = check(tasks)
    end
    return ok ? 0 : 1
end

exit(main(ARGS))
