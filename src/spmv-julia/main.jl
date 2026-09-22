using CUDA
using Printf
using Random

function init_vector(n::Int)
    rng = MersenneTwister(4096)
    return rand(rng, Float32, n)
end

function init_matrix(num_rows::Int, nnz::Int)
    total = num_rows * num_rows
    rng_perm = MersenneTwister(123)
    order = collect(0:total-1)
    for i in total:-1:2
        j = rand(rng_perm, 1:i)
        order[i], order[j] = order[j], order[i]
    end
    rng_vals = MersenneTwister(8192)
    matrix = Vector{Float32}(undef, total)
    @inbounds for i in 1:total
        matrix[i] = order[i] >= nnz ? 0.0f0 : rand(rng_vals, Float32) + 1.0f0
    end
    return matrix
end

function init_csr(matrix, num_rows::Int, nnz::Int)
    rows = zeros(Int32, num_rows + 1)
    cols = Vector{Int32}(undef, nnz)
    vals = Vector{Float32}(undef, nnz)
    pos = 1
    @inbounds for r in 1:num_rows
        rows[r] = Int32(pos - 1)
        for c in 1:num_rows
            v = matrix[(r - 1) * num_rows + c]
            if v != 0.0f0
                vals[pos] = v
                cols[pos] = Int32(c - 1)
                pos += 1
            end
        end
    end
    rows[num_rows + 1] = Int32(nnz)
    return rows, cols, vals
end

function init_coo(matrix, num_rows::Int, nnz::Int)
    rows = Vector{Int32}(undef, nnz)
    cols = Vector{Int32}(undef, nnz)
    vals = Vector{Float32}(undef, nnz)
    pos = 1
    @inbounds for r in 1:num_rows, c in 1:num_rows
        v = matrix[(r - 1) * num_rows + c]
        if v != 0.0f0
            rows[pos] = Int32(r - 1)
            cols[pos] = Int32(c - 1)
            vals[pos] = v
            pos += 1
        end
    end
    return rows, cols, vals
end

function dense_kernel!(matrix, x, y, n::Int32)
    r = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if r <= n
        acc = 0.0f0
        @inbounds for c in Int32(1):n
            v = matrix[(r - Int32(1)) * n + c]
            if v != 0.0f0
                acc += v * x[c]
            end
        end
        @inbounds y[r] = acc
    end
    return
end

function csr_kernel!(rowptr, cols, vals, x, y, n::Int32)
    r = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if r <= n
        acc = 0.0f0
        start = rowptr[r] + Int32(1)
        stop = rowptr[r + Int32(1)]
        @inbounds for p in start:stop
            acc += vals[p] * x[cols[p] + Int32(1)]
        end
        @inbounds y[r] = acc
    end
    return
end

function coo_kernel!(rows, cols, vals, x, y, nnz::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    while i <= nnz
        @inbounds CUDA.@atomic y[rows[i] + Int32(1)] += vals[i] * x[cols[i] + Int32(1)]
        i += stride
    end
    return
end

function error_rate(a, b)
    return sum(abs.(a .- b)) / max(sum(abs.(b)), eps(Float32))
end

function main(args)
    if length(args) != 3
        println("Usage main.jl <number of non-zero elements> <number of rows in a square matrix> <repeat>")
        return 1
    end
    nnz = parse(Int, args[1])
    num_rows = parse(Int, args[2])
    repeat = parse(Int, args[3])
    @assert nnz > 0 && num_rows > 0 && nnz <= num_rows * num_rows
    matrix = init_matrix(num_rows, nnz)
    x = init_vector(num_rows)

    d_matrix, d_x = CuArray(matrix), CuArray(x)
    d_ref = CUDA.zeros(Float32, num_rows)
    threads = 256
    row_blocks = max(cld(num_rows, threads), 1)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=row_blocks dense_kernel!(d_matrix, d_x, d_ref, Int32(num_rows))
    end
    CUDA.synchronize()
    y_ref = Array(d_ref)

    println("Number of non-zero elements: $nnz")
    println("Number of rows in a square matrix: $num_rows")
    @printf("Sparsity: %lf%%\n", (num_rows * num_rows - nnz) / (num_rows * num_rows) * 100.0)

    rowptr, csr_cols, csr_vals = init_csr(matrix, num_rows, nnz)
    d_rowptr, d_csr_cols, d_csr_vals = CuArray(rowptr), CuArray(csr_cols), CuArray(csr_vals)
    d_y = CUDA.zeros(Float32, num_rows)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=row_blocks csr_kernel!(d_rowptr, d_csr_cols, d_csr_vals, d_x, d_y, Int32(num_rows))
    end
    CUDA.synchronize()
    elapsed = time_ns() - t0
    @printf("Average kernel (CSR) execution time (ms): %lf\n", elapsed * 1e-6 / repeat)
    @printf("Error rate: %f\n", error_rate(Array(d_y), y_ref))

    coo_rows, coo_cols, coo_vals = init_coo(matrix, num_rows, nnz)
    d_coo_rows, d_coo_cols, d_coo_vals = CuArray(coo_rows), CuArray(coo_cols), CuArray(coo_vals)
    coo_blocks = max(cld(nnz, threads), 1)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        fill!(d_y, 0.0f0)
        @cuda threads=threads blocks=coo_blocks coo_kernel!(d_coo_rows, d_coo_cols, d_coo_vals, d_x, d_y, Int32(nnz))
    end
    CUDA.synchronize()
    elapsed = time_ns() - t0
    @printf("Average kernel (COO) execution time (ms): %lf\n", elapsed * 1e-6 / repeat)
    @printf("Error rate: %f\n", error_rate(Array(d_y), y_ref))
    return 0
end

exit(main(ARGS))
