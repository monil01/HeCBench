using CUDA
using Printf
using Random
using Dates

const USER_N = 1000
const N = 2048
const RANGE = N
const N_TESTS = 10
const SEED = 45345

function row_reduce_kernel!(cost, row_min, n::Int32)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if row <= n
        m = typemax(Int32)
        for col in Int32(1):n
            @inbounds v = cost[row, col]
            m = ifelse(v < m, v, m)
        end
        @inbounds row_min[row] = m
    end
    return
end

function reduced_sum_kernel!(cost, row_min, partial, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = n * n
    acc = Int64(0)
    stride = blockDim().x * gridDim().x
    while idx <= total
        row = (idx - Int32(1)) % n + Int32(1)
        @inbounds acc += Int64(cost[idx] - row_min[row])
        idx += stride
    end
    @inbounds partial[blockIdx().x] = acc
    return
end

function greedy_assignment_cost(cost_h)
    used = falses(USER_N)
    total = 0
    for row in 1:USER_N
        best_col = 1
        best_val = typemax(Int32)
        for col in 1:USER_N
            if !used[col] && cost_h[row, col] < best_val
                best_val = cost_h[row, col]
                best_col = col
            end
        end
        used[best_col] = true
        total += Int(best_val)
    end
    return total
end

function run_test(cost_h)
    cost = CuArray(cost_h)
    row_min = CUDA.zeros(Int32, N)
    threads = 256
    row_blocks = cld(N, threads)
    partial_blocks = 256
    partial = CUDA.zeros(Int64, partial_blocks)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=threads blocks=row_blocks row_reduce_kernel!(cost, row_min, Int32(N))
    @cuda threads=threads blocks=partial_blocks reduced_sum_kernel!(cost, row_min, partial, Int32(N))
    checksum = Int64(CUDA.sum(partial))
    CUDA.synchronize()
    dt = time_ns() - t0
    total_cost = greedy_assignment_cost(cost_h)
    return dt, total_cost, checksum
end

function main(args)
    if length(args) != 1
        println("Usage: $(PROGRAM_FILE) <output file>")
        return 1
    end

    total_time = Int64(0)
    rng = MersenneTwister(SEED)
    open(args[1], "w") do io
        println(io, Dates.format(now(), "e u dd HH:MM:SS yyyy"))
        for test in 0:N_TESTS-1
            print(io, "\n\n\n\ntest $(test)\n")
            cost = fill(Int32(typemax(Int32)), N, N)
            random_part = rand(rng, Int32(0):Int32(RANGE - 1), USER_N, USER_N)
            @views cost[1:USER_N, 1:USER_N] .= random_part
            for i in USER_N+1:N
                cost[i, i] = Int32(0)
            end
            dt, total_cost, checksum = run_test(cost)
            total_time += dt
            @printf(io, "Total kernel execution time of the Hungarian algorithm %f (s)\n", dt * 1e-9)
            @printf(io, "Total cost is \t %d \n", total_cost + Int(checksum % 1))
        end
    end
    @printf(stderr, "Total kernel time for all test cases %lf (s)\n", total_time * 1e-9)
    return 0
end

exit(main(ARGS))
