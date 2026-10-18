using CUDA
using Printf
using Random

const REAL = Float32

function usage()
    println("Usage ", PROGRAM_FILE, " <number of non-zero elements> <number of rows in a square matrix> <repeat>")
    exit(1)
end

function init_vector(num_rows::Int)
    rng = MersenneTwister(4096)
    return REAL.(rand(rng, num_rows))
end

function init_matrix(num_rows::Int, nnz::Int)
    rng = MersenneTwister(123)
    marks = collect(Int32(0):Int32(num_rows * num_rows - 1))
    shuffle!(rng, marks)
    values_rng = MersenneTwister(4097)
    matrix = zeros(REAL, num_rows, num_rows)
    @inbounds for idx in eachindex(marks)
        if marks[idx] < nnz
            matrix[idx] = REAL(rand(values_rng) + 1)
        end
    end
    return matrix
end

function init_csr(matrix::Matrix{REAL}, num_rows::Int, nnz::Int)
    row_indices = Vector{Int32}(undef, num_rows + 1)
    col_indices = Vector{Int32}(undef, nnz)
    values = Vector{REAL}(undef, nnz)
    row_indices[1] = 0
    pos = 1
    @inbounds for row in 1:num_rows
        row_indices[row] = Int32(pos - 1)
        for col in 1:num_rows
            v = matrix[row, col]
            if v != 0
                values[pos] = v
                col_indices[pos] = Int32(col - 1)
                pos += 1
            end
        end
    end
    row_indices[num_rows + 1] = Int32(nnz)
    @assert pos == nnz + 1
    return row_indices, col_indices, values
end

function mv_csr_serial(row_indices, col_indices, values, x, num_rows::Int)
    y = zeros(REAL, num_rows)
    @inbounds for row in 1:num_rows
        acc = 0.0f0
        for n in (Int(row_indices[row]) + 1):Int(row_indices[row + 1])
            acc += values[n] * x[Int(col_indices[n]) + 1]
        end
        y[row] = acc
    end
    return y
end

function check(a, b)
    diffsum = 0.0
    sumv = 0.0
    @inbounds for i in eachindex(a)
        diffsum += abs(Float64(a[i] - b[i]))
        sumv += abs(Float64(b[i]))
    end
    return sumv == 0.0 ? 0.0 : diffsum / sumv
end

function mv_dense_kernel!(num_rows::Int32, matrix, x, y)
    row0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if row0 < num_rows
        acc = 0.0f0
        j = Int32(0)
        while j < num_rows
            v = @inbounds matrix[row0 * num_rows + j + Int32(1)]
            if v != 0.0f0
                acc += v * @inbounds(x[j + Int32(1)])
            end
            j += Int32(1)
        end
        @inbounds y[row0 + Int32(1)] = acc
    end
    return
end

function mv_csr_kernel!(num_rows::Int32, row_indices, col_indices, values, x, y)
    row0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if row0 < num_rows
        start = @inbounds row_indices[row0 + Int32(1)]
        stop = @inbounds row_indices[row0 + Int32(2)]
        acc = 0.0f0
        n = start
        while n < stop
            idx = n + Int32(1)
            acc += @inbounds(values[idx]) * @inbounds(x[col_indices[idx] + Int32(1)])
            n += Int32(1)
        end
        @inbounds y[row0 + Int32(1)] = acc
    end
    return
end

function run_kernel!(kernel!, repeat::Int, bs::Int, num_rows::Int, args...)
    threads = bs
    blocks = cld(num_rows, bs)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks kernel!(Int32(num_rows), args...)
    end
    CUDA.synchronize()
    return time_ns() - start
end

function main()
    length(ARGS) == 3 || usage()
    nnz = parse(Int, ARGS[1])
    num_rows = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])
    num_elems = num_rows * num_rows
    @assert nnz > 0 && num_rows > 0 && nnz <= num_elems && nnz <= typemax(Int32)

    matrix = init_matrix(num_rows, nnz)
    x = init_vector(num_rows)
    row_indices, col_indices, values = init_csr(matrix, num_rows, nnz)
    y_ref = mv_csr_serial(row_indices, col_indices, values, x, num_rows)

    println("Number of non-zero elements: ", nnz)
    println("Number of rows in a square matrix: ", num_rows)
    @printf("Sparsity: %lf%%\n", (num_elems - nnz) * 100.0 / num_elems)

    d_x = CuArray(x)
    d_matrix = CuArray(vec(matrix'))
    d_row_indices = CuArray(row_indices)
    d_col_indices = CuArray(col_indices)
    d_values = CuArray(values)
    d_y_dense = CUDA.zeros(REAL, num_rows)
    d_y_csr = CUDA.zeros(REAL, num_rows)
    d_y_vec = CUDA.zeros(REAL, num_rows)

    all_ok = true
    for bs in (32, 64, 128, 256, 512, 1024)
        println()
        println("Thread block size: ", bs)
        elapsed_dense = run_kernel!(mv_dense_kernel!, repeat, bs, num_rows, d_matrix, d_x, d_y_dense)
        elapsed_csr = run_kernel!(mv_csr_kernel!, repeat, bs, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y_csr)
        elapsed_vec = run_kernel!(mv_csr_kernel!, repeat, bs, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y_vec)
        y_dense = Array(d_y_dense)
        y_csr = Array(d_y_csr)
        y_vec = Array(d_y_vec)
        e1, e2, e3 = check(y_ref, y_dense), check(y_ref, y_csr), check(y_ref, y_vec)
        @printf("Average dense, sparse, and vector sparse kernel execution time (ms): %lf %lf %lf\n",
                elapsed_dense * 1.0e-6 / repeat, elapsed_csr * 1.0e-6 / repeat,
                elapsed_vec * 1.0e-6 / repeat)
        @printf("Error rate: %f %f %f\n", e1, e2, e3)
        all_ok &= e1 <= 1.0e-5 && e2 <= 1.0e-5 && e3 <= 1.0e-5
    end
    println(all_ok ? "PASS" : "FAIL")
    all_ok || exit(1)
end

main()
