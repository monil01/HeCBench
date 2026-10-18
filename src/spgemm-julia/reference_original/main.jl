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

function init_coo(matrix, num_rows, num_cols, nnz)
    rows = Vector{Int32}(undef, nnz)
    cols = Vector{Int32}(undef, nnz)
    vals = Vector{Float32}(undef, nnz)
    cursor = 1
    @inbounds for i in 0:(num_rows - 1), j in 0:(num_cols - 1)
        v = matrix[i * num_cols + j + 1]
        if v != 0.0f0
            rows[cursor] = Int32(i)
            cols[cursor] = Int32(j)
            vals[cursor] = v
            cursor += 1
        end
    end
    return rows, vals, cols
end

function init_csr(matrix, num_rows, num_cols, nnz)
    offsets = Vector{Int32}(undef, num_rows + 1)
    cols = Vector{Int32}(undef, nnz)
    vals = Vector{Float32}(undef, nnz)
    offsets[1] = 0
    cursor = 1
    @inbounds for i in 0:(num_rows - 1)
        row_nnz = 0
        for j in 0:(num_cols - 1)
            v = matrix[i * num_cols + j + 1]
            if v != 0.0f0
                cols[cursor] = Int32(j)
                vals[cursor] = v
                cursor += 1
                row_nnz += 1
            end
        end
        offsets[i + 2] = offsets[i + 1] + Int32(row_nnz)
    end
    return offsets, vals, cols
end

function dense_gemm(a, b, m, k, n)
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

function coo_spmm_kernel!(c, rows, vals, cols, b, nnz::Int32, n::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = nnz * n
    if tid >= total
        return
    end
    nz = tid ÷ n
    col = tid - nz * n
    r = rows[nz + Int32(1)]
    k = cols[nz + Int32(1)]
    CUDA.@atomic c[r * n + col + Int32(1)] += vals[nz + Int32(1)] * b[k * n + col + Int32(1)]
    return
end

function csr_spmm_kernel!(c, offsets, vals, cols, b, m::Int32, n::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = m * n
    if tid >= total
        return
    end
    row = tid ÷ n
    col = tid - row * n
    s = 0.0f0
    @inbounds for nz in offsets[row + Int32(1)]:(offsets[row + Int32(2)] - Int32(1))
        k = cols[nz + Int32(1)]
        s += vals[nz + Int32(1)] * b[k * n + col + Int32(1)]
    end
    @inbounds c[tid + Int32(1)] = s
    return
end

function run_coo(h_a, h_b, m, k, n, a_nnz, repeat, verify)
    rows, vals, cols = init_coo(h_a, m, k, a_nnz)
    d_rows = CuArray(rows); d_vals = CuArray(vals); d_cols = CuArray(cols)
    d_b = CuArray(h_b)
    d_c = CUDA.zeros(Float32, m * n)
    blocks = cld(a_nnz * n, THREADS)
    @cuda threads=THREADS blocks=blocks coo_spmm_kernel!(d_c, d_rows, d_vals, d_cols, d_b, Int32(a_nnz), Int32(n))
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        fill!(d_c, 0.0f0)
        @cuda threads=THREADS blocks=blocks coo_spmm_kernel!(d_c, d_rows, d_vals, d_cols, d_b, Int32(a_nnz), Int32(n))
    end
    CUDA.synchronize()
    @printf("Average execution time of SPGEMM (COO) compute: %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
    return verify == 1 ? Array(d_c) : Float32[]
end

function run_csr(h_a, h_b, m, k, n, a_nnz, repeat, verify)
    offsets, vals, cols = init_csr(h_a, m, k, a_nnz)
    d_offsets = CuArray(offsets); d_vals = CuArray(vals); d_cols = CuArray(cols)
    d_b = CuArray(h_b)
    d_c = CUDA.zeros(Float32, m * n)
    blocks = cld(m * n, THREADS)
    @cuda threads=THREADS blocks=blocks csr_spmm_kernel!(d_c, d_offsets, d_vals, d_cols, d_b, Int32(m), Int32(n))
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks csr_spmm_kernel!(d_c, d_offsets, d_vals, d_cols, d_b, Int32(m), Int32(n))
    end
    CUDA.synchronize()
    @printf("Average execution time of SPGEMM (CSR) compute: %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
    return verify == 1 ? Array(d_c) : Float32[]
end

function check_result(name, got, ref)
    println("Computing the reference SPGEMM results..")
    ok = true
    @inbounds for i in eachindex(ref)
        if abs(got[i] - ref[i]) > 1.0f-2
            @printf("@%d %f != %f\n", i - 1, got[i], ref[i])
            ok = false
            break
        end
    end
    println(ok ? "spgemm_example test PASSED" : "spgemm_example test FAILED: wrong result")
    return ok
end

function main(args)
    if length(args) != 6
        println("Single-precision sparse matrix-dense matrix multiplication into dense matrix,")
        println("where the sparse matrix is represented in COO and CSR storage format")
        println("Usage main.jl <M> <K> <N> <A_nnz> <repeat> <verify>")
        println("SPMM (A, B, C) where (A: M * K, B: K * N, C: M * N)")
        return 1
    end
    m, k, n, a_nnz, repeat, verify = parse.(Int, args)
    a_nnz = min(a_nnz, m * k)
    h_a = init_matrix(m, k, a_nnz, 123)
    h_b = init_matrix(k, n, k * n, 456)
    coo = run_coo(h_a, h_b, m, k, n, a_nnz, repeat, verify)
    csr = run_csr(h_a, h_b, m, k, n, a_nnz, repeat, verify)
    if verify == 1
        ref = dense_gemm(h_a, h_b, m, k, n)
        ok = check_result("COO", coo, ref) & check_result("CSR", csr, ref)
        println(ok ? "PASS" : "FAIL")
        return ok ? 0 : 1
    end
    println("PASS")
    return 0
end

exit(main(ARGS))
