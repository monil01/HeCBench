using CUDA
using Printf
using Random

const THREADS = 256

function init_matrix(num_rows, num_cols, nnz, seed)
    n = num_rows * num_cols
    nnz = min(nnz, n)
    rng = MersenneTwister(seed)
    positions = randperm(rng, n)
    matrix = zeros(Float32, n)
    @inbounds for p in positions[1:nnz]
        matrix[p] = rand(rng, Float32) + 1.0f0
    end
    return matrix
end

function init_csr(matrix, num_rows, num_cols, nnz)
    offsets = Vector{Int32}(undef, num_rows + 1)
    columns = Vector{Int32}(undef, nnz)
    values = Vector{Float32}(undef, nnz)
    offsets[1] = 0
    cursor = 1
    @inbounds for i in 0:(num_rows - 1)
        row_nnz = 0
        for j in 0:(num_cols - 1)
            v = matrix[i * num_cols + j + 1]
            if v != 0.0f0
                values[cursor] = v
                columns[cursor] = Int32(j)
                cursor += 1
                row_nnz += 1
            end
        end
        offsets[i + 2] = offsets[i + 1] + Int32(row_nnz)
    end
    return offsets, values, columns
end

function spmm_dense_reference(a, b, m, k, n)
    c = zeros(Float32, m * n)
    @inbounds for row in 0:(m - 1), col in 0:(n - 1)
        s = 0.0
        for kk in 0:(k - 1)
            s += Float64(a[row * k + kk + 1]) * Float64(b[kk * n + col + 1])
        end
        c[row * n + col + 1] = Float32(s)
    end
    return c
end

function count_nnz(x)
    c = 0
    @inbounds for v in x
        c += v != 0.0f0 ? 1 : 0
    end
    return c
end

function csr_spmm_dense_kernel!(c, a_offsets, a_values, a_columns,
                                b_offsets, b_values, b_columns,
                                m::Int32, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = m * n
    if idx >= total
        return
    end
    row = idx ÷ n
    col = idx - row * n
    sum = 0.0f0
    @inbounds begin
        a_begin = a_offsets[row + Int32(1)]
        a_end = a_offsets[row + Int32(2)]
        for ai in a_begin:(a_end - Int32(1))
            aval = a_values[ai + Int32(1)]
            brow = a_columns[ai + Int32(1)]
            b_begin = b_offsets[brow + Int32(1)]
            b_end = b_offsets[brow + Int32(2)]
            for bi in b_begin:(b_end - Int32(1))
                if b_columns[bi + Int32(1)] == col
                    sum += aval * b_values[bi + Int32(1)]
                    break
                end
            end
        end
        c[idx + Int32(1)] = sum
    end
    return
end

function csr_spmm_dense!(d_c, d_a_offsets, d_a_values, d_a_columns,
                         d_b_offsets, d_b_values, d_b_columns, m, n)
    blocks = cld(m * n, THREADS)
    @cuda threads=THREADS blocks=blocks csr_spmm_dense_kernel!(
        d_c, d_a_offsets, d_a_values, d_a_columns,
        d_b_offsets, d_b_values, d_b_columns, Int32(m), Int32(n))
    return d_c
end

function main(args)
    if length(args) != 7
        println("Single-precision sparse matrix-matrix multiplication into sparse matrix,")
        println("where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format")
        println("Usage main.jl <M> <K> <N> <A_nnz> <B_nnz> <repeat> <verify>")
        println("SPMM (A, B, C) where (A: M * K, B: K * N, C: M * N)")
        return 1
    end

    m, k, n, a_nnz, b_nnz, repeat, verify = parse.(Int, args)
    if any(x -> x <= 0, (m, k, n, a_nnz, b_nnz, repeat)) || !(verify in (0, 1))
        println("All size and repeat arguments must be positive; verify must be 0 or 1.")
        return 1
    end
    a_nnz = min(a_nnz, m * k)
    b_nnz = min(b_nnz, k * n)

    h_a = init_matrix(m, k, a_nnz, 123)
    h_b = init_matrix(k, n, b_nnz, 456)
    h_a_offsets, h_a_values, h_a_columns = init_csr(h_a, m, k, a_nnz)
    h_b_offsets, h_b_values, h_b_columns = init_csr(h_b, k, n, b_nnz)

    d_a_offsets = CuArray(h_a_offsets)
    d_a_values = CuArray(h_a_values)
    d_a_columns = CuArray(h_a_columns)
    d_b_offsets = CuArray(h_b_offsets)
    d_b_values = CuArray(h_b_values)
    d_b_columns = CuArray(h_b_columns)
    d_c = CUDA.zeros(Float32, m * n)

    csr_spmm_dense!(d_c, d_a_offsets, d_a_values, d_a_columns,
                    d_b_offsets, d_b_values, d_b_columns, m, n)
    CUDA.synchronize()

    t0 = time_ns()
    for _ in 1:repeat
        csr_spmm_dense!(d_c, d_a_offsets, d_a_values, d_a_columns,
                        d_b_offsets, d_b_values, d_b_columns, m, n)
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - t0
    @printf("Average execution time of SPMM compute: %f (us)\n", elapsed_ns * 1.0e-3 / repeat)

    if verify == 1
        println("Computing the reference SPMM results..")
        h_c_ref = spmm_dense_reference(h_a, h_b, m, k, n)
        h_c = Array(d_c)
        c_nnz = count_nnz(h_c_ref)
        correct = true
        @inbounds for i in eachindex(h_c_ref)
            if abs(h_c[i] - h_c_ref[i]) > 1.0f-2
                correct = false
                break
            end
        end
        if correct
            println("spgemm_example test PASSED")
            println("PASS")
        else
            println("spgemm_example test FAILED: wrong result")
            println("FAIL")
            return 1
        end
    else
        println("PASS")
    end
    return 0
end

exit(main(ARGS))
