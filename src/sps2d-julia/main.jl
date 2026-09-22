using CUDA
using CUDA.CUSPARSE
using Printf
using Random
using SparseArrays

function init_matrix(num_rows::Int, num_cols::Int, nnz::Int)
    n = num_rows * num_cols
    order = collect(Float32, 0:n-1)
    rng = Random.MersenneTwister(123)
    for i in n:-1:1
        b = rand(rng, 1:i)
        if i != b
            order[i], order[b] = order[b], order[i]
        end
    end

    vals = Vector{Float32}(undef, n)
    rng_vals = Random.MersenneTwister(123)
    @inbounds for i in 1:n
        vals[i] = order[i] >= nnz ? 0.0f0 : Float32(rand(rng_vals) + 1)
    end
    return reshape(vals, num_cols, num_rows)'
end

function init_csr(matrix::AbstractMatrix{Float32}, nnz::Int)
    num_rows, num_cols = size(matrix)
    row_offsets = Vector{Int64}(undef, num_rows + 1)
    columns = Vector{Int64}(undef, nnz)
    values = Vector{Float32}(undef, nnz)

    row_offsets[1] = 0
    pos = 1
    @inbounds for i in 1:num_rows
        row_nnz = 0
        for j in 1:num_cols
            v = matrix[i, j]
            if v != 0.0f0
                values[pos] = v
                columns[pos] = j - 1
                pos += 1
                row_nnz += 1
            end
        end
        row_offsets[i + 1] = row_offsets[i] + row_nnz
    end
    row_offsets[end] = nnz
    return row_offsets, columns, values
end

function main()
    if length(ARGS) != 4
        println("The function converts a sparse matrix into a MxN dense matrix")
        println("The sparse matrix is represented in CSR (Compressed Sparse Row) storage format")
        println("Usage ", PROGRAM_FILE, " <M> <N> <nnz> <repeat>")
        println("nnz is the number of non-zero elements")
        exit(1)
    end

    m = parse(Int, ARGS[1])
    n = parse(Int, ARGS[2])
    h_nnz = parse(Int, ARGS[3])
    repeat = parse(Int, ARGS[4])

    println("Initializing host matrices..")
    h_dense = init_matrix(m, n, h_nnz)
    h_csr_offsets, h_csr_columns0, h_csr_values = init_csr(h_dense, h_nnz)

    d_row_offsets = CuArray(h_csr_offsets)
    d_columns = CuArray(h_csr_columns0)
    d_values = CuArray(h_csr_values)
    csr = CuSparseMatrixCSR(d_row_offsets, d_columns, d_values, (m, n))

    dense = CUSPARSE.sparsetodense(csr, 'Z')
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        dense = CUSPARSE.sparsetodense(csr, 'Z')
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of SparseToDense_convert : %f (us)\n",
            elapsed * 1.0e-3 / repeat)

    h_dense_result = Array(dense)
    correct = true
    found_nnz = count(!=(0.0f0), h_dense_result)
    h_dense_linear = vec(h_dense)
    h_dense_result_linear = vec(h_dense_result)
    @inbounds for i in eachindex(h_dense_linear)
        if h_dense_linear[i] != h_dense_result_linear[i]
            @printf("@%ld: %f != %f\n", i - 1, h_dense_linear[i], h_dense_result_linear[i])
            correct = false
            break
        end
    end
    if found_nnz != h_nnz
        @printf("nnz: %ld != %ld\n", h_nnz, found_nnz)
        correct = false
    end

    if correct
        println("sparse2dense_csr_example test PASSED")
    else
        println("sparse2dense_csr_example test FAILED: wrong result")
    end
end

main()
