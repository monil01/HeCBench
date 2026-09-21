using Printf
using Random

function init_matrix(num_rows::Int, num_cols::Int, nnz::Int)
    n = num_rows * num_cols
    order = collect(Float32, 0:n-1)
    rng = MersenneTwister(123)
    for i in n:-1:1
        b = rand(rng, 1:i)
        if i != b
            order[i], order[b] = order[b], order[i]
        end
    end
    rng2 = MersenneTwister(456)
    dense = Vector{Float32}(undef, n)
    for i in 1:n
        dense[i] = order[i] >= nnz ? 0.0f0 : Float32(rand(rng2) + 1.0)
    end
    return dense
end

function init_csr(dense, num_rows::Int, num_cols::Int, nnz::Int)
    offsets = zeros(Int64, num_rows + 1)
    values = Vector{Float32}(undef, nnz)
    columns = Vector{Int64}(undef, nnz)
    tmp = 1
    for i in 0:num_rows-1
        row_nnz = 0
        for j in 0:num_cols-1
            v = dense[i * num_cols + j + 1]
            if v != 0.0f0
                values[tmp] = v
                columns[tmp] = j
                tmp += 1
                row_nnz += 1
            end
        end
        offsets[i + 2] = offsets[i + 1] + row_nnz
    end
    return offsets, values, columns
end

function main()
    if length(ARGS) != 4
        println("The function converts a dense MxN matrix into a sparse matrix")
        println("The sparse matrix is represented in CSR (Compressed Sparse Row) storage format")
        println("Usage main.jl <M> <N> <nnz> <repeat>")
        println("nnz is the number of non-zero elements")
        return 1
    end
    m = parse(Int, ARGS[1])
    n = parse(Int, ARGS[2])
    h_nnz = parse(Int, ARGS[3])
    repeat = parse(Int, ARGS[4])

    println("Initializing host matrices..")
    dense = init_matrix(m, n, h_nnz)
    ref_offsets, ref_values, ref_columns = init_csr(dense, m, n, h_nnz)

    start = time_ns()
    offsets = ref_offsets
    values = ref_values
    columns = ref_columns
    for _ in 1:repeat
        offsets, values, columns = init_csr(dense, m, n, h_nnz)
    end
    elapsed = time_ns() - start
    @printf("Average execution time of DenseToSparse_convert : %f (us)\n", elapsed * 1e-3 / repeat)

    correct = true
    if h_nnz != length(values)
        @printf("nnz: %ld != %ld\n", h_nnz, length(values))
        correct = false
    elseif offsets != ref_offsets
        bad = findfirst(!=, zip(offsets, ref_offsets))
        correct = false
        if bad !== nothing
            @printf("rowidx mismatch\n")
        end
    else
        sort!(columns); sort!(values)
        sort!(ref_columns); sort!(ref_values)
        correct = columns == ref_columns && values == ref_values
    end
    if correct
        println("dense2sparse_csr_example test PASSED")
    else
        println("dense2sparse_csr_example test FAILED: wrong result")
    end
    return 0
end

exit(main())
