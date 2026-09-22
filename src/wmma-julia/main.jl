using CUDA
using Printf
using LinearAlgebra

const WMMA_M = 16
const WMMA_N = 16
const WMMA_K = 16
const WAVE_SIZE = 32
const TILE_M = 64
const TILE_N = 64

function fill_linear(::Type{T}, rows::Int, cols::Int) where {T}
    data = Vector{T}(undef, rows * cols)
    @inbounds for i in 0:rows-1, j in 0:cols-1
        value = (i * cols + j) % 13
        signed = (value % 3) != 0 ? -value : value
        data[i * cols + j + 1] = T(signed)
    end
    return data
end

rowmajor_matrix(data::Vector{T}, rows::Int, cols::Int) where {T} =
    permutedims(reshape(data, cols, rows))

column_chunks_matrix(data::Vector{T}, rows::Int, cols::Int) where {T} =
    reshape(data, rows, cols)

function compare_equal(a::AbstractMatrix{Float32}, b::AbstractMatrix{Float32};
                       tolerance::Float64=10.0)
    max_relative_error = 0.0
    @inbounds for j in axes(a, 2), i in axes(a, 1)
        val_a = a[i, j]
        val_b = b[i, j]
        rel = abs(Float64(val_a - val_b)) / (abs(Float64(val_a)) + abs(Float64(val_b)) + 1.0)
        if rel > max_relative_error || isnan(rel)
            max_relative_error = rel
        end
    end
    epsv = eps(Float32)
    if isnan(max_relative_error) || max_relative_error > epsv * tolerance
        println("FAILED")
    else
        println("PASSED")
    end
    println("Max relative error: ", max_relative_error)
    return !(isnan(max_relative_error) || max_relative_error > epsv * tolerance)
end

function usage()
    println(" Incorrect parameters")
    println(" Usage: ")
    println(PROGRAM_FILE, "<implementation> <M> <N> <K> <repeat> <verify>\n")
    println("Dense matrix-matrix multiplication: D = alpha * (A * B) + beta * C")
    println("A: M * K, B: K * N, C: M * N, D: M * N")
    exit(-1)
end

function check_supported(impl::Int, m::Int, n::Int, k::Int)
    if impl == 0
        return !(m < WMMA_M || n < WMMA_N || k < WMMA_K ||
                 m % WMMA_M != 0 || n % WMMA_N != 0 || k % WMMA_K != 0)
    end
    return !((m < TILE_M) || n < TILE_N || k < WMMA_K ||
             m % WMMA_M != 0 || n % WMMA_N != 0 || k % WMMA_K != 0 ||
             (TILE_M ÷ WMMA_M) * 32 * (TILE_N ÷ WMMA_N) > 1024)
end

function gemm_once!(d_d, d_a, d_b, d_c, alpha::Float32, beta::Float32)
    copyto!(d_d, d_c)
    CUDA.CUBLAS.gemmEx!('N', 'N', alpha, d_a, d_b, beta, d_d)
    return d_d
end

function gemm_wmma(impl::Int, m::Int, n::Int, k::Int,
                   alpha::Float32, beta::Float32, repeat::Int, verify::Int)
    if !check_supported(impl, m, n, k)
        println("Unsupported size!")
        return
    end

    lda = k
    ldb = k
    ldc = n
    ldd = ldc

    println("Initializing host data...")
    avec = fill_linear(Float16, m, k)
    bvec = fill_linear(Float16, k, n)
    cvec = fill_linear(Float32, m, n)

    matrix_a = rowmajor_matrix(avec, m, k)
    matrix_b = column_chunks_matrix(bvec, k, n)
    matrix_c = rowmajor_matrix(cvec, m, n)
    matrix_d = fill(Float32(NaN), m, n)

    println("Initializing device data...")
    d_a = CuArray(matrix_a)
    d_b = CuArray(matrix_b)
    d_c = CuArray(matrix_c)
    d_d = CuArray(matrix_d)

    println("Launching GEMM kernel...")
    for _ in 1:30
        gemm_once!(d_d, d_a, d_b, d_c, alpha, beta)
    end

    if verify != 0
        println("Validating result with reference...")
        matrix_d = Array(d_d)
        matrix_d_ref = alpha .* (Float32.(matrix_a) * Float32.(matrix_b)) .+ beta .* matrix_c
        compare_equal(matrix_d, matrix_d_ref)
    else
        println("Skip validating result with reference")
    end

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        gemm_once!(d_d, d_a, d_b, d_c, alpha, beta)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6

    gflops = Float64(m) * n * (1.0 + 2.0 * k) * 1.0e-9
    tflops_per_sec = gflops * repeat / elapsed_ms

    println("BlkM, BlkN, BlkK, MatM, MatN, MatK, alpha, lda, ldb, beta, ldc, ldd, elapsedMs, Problem Size(GFlops), TFlops/s")
    println(WMMA_M, ", ", WMMA_N, ", ", WMMA_K, ", ", m, ", ",
            n, ", ", k, ", ", alpha, ", ", lda, ", ", ldb,
            ", ", beta, ", ", ldc, ", ", ldd, ", ",
            elapsed_ms, ", ", gflops, ", ", tflops_per_sec)
    println("Finished!")
end

function main()
    length(ARGS) == 6 || usage()
    impl = parse(Int, ARGS[1])
    m = parse(Int, ARGS[2])
    n = parse(Int, ARGS[3])
    k = parse(Int, ARGS[4])
    repeat = parse(Int, ARGS[5])
    verify = parse(Int, ARGS[6])
    gemm_wmma(impl, m, n, k, 0.5f0, 2.0f0, repeat, verify)
end

main()
