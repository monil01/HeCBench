using CUDA
using CUDA.CUBLAS
using LinearAlgebra
using Printf
using Random

const N = 48
const BATCH_SIZE = 10_000
const MAX_ERROR = 1.0f-6

function usage()
    println("Usage: ", PROGRAM_FILE, " <repeat>")
    exit(1)
end

function init_random_matrices()
    rng = MersenneTwister(12345)
    a = Array{Float32}(undef, N, N, BATCH_SIZE)
    @inbounds for b in 1:BATCH_SIZE
        for j in 1:N, i in 1:N
            a[i, j, b] = 1.0f0 + rand(rng, Float32)
        end
        for i in 1:N
            a[i, i, b] += Float32(N)
        end
    end
    return a
end

function permutation_matrix_from_pivot(piv)
    pivot = collect(1:N)
    @inbounds for k in 1:N
        q = Int(piv[k])
        pivot[k], pivot[q] = pivot[q], pivot[k]
    end
    pmat = zeros(Float32, N, N)
    @inbounds for i in 1:N
        pmat[i, pivot[i]] = 1.0f0
    end
    return pmat
end

function check_relative_error(a, b, max_error)
    rel_max = 0.0f0
    @inbounds for i in eachindex(a, b)
        ref = abs(a[i])
        err = abs(a[i] - b[i])
        if ref != 0.0f0 && err > 0.0f0
            rel = err / ref
            rel_max = max(rel_max, rel)
            rel_max > max_error && return false
        end
    end
    return true
end

function main()
    length(ARGS) == 1 || usage()
    repeat = parse(Int, ARGS[1])

    println("> initializing..")
    CUDA.device()
    println("> using SINGLE precision..")
    println("> pivot ENABLED..")

    println("> generating random matrices..")
    h_input = init_random_matrices()
    d_array = CuArray(h_input)

    total_ns = 0
    pivot = nothing
    info = nothing
    println("> performing batched LU decomposition..")
    for iter in 0:repeat
        copyto!(d_array, h_input)
        CUDA.synchronize()
        start = time_ns()
        pivot, info = CUBLAS.getrf_strided_batched!(d_array, true)
        CUDA.synchronize()
        iter != 0 && (total_ns += time_ns() - start)
    end
    @printf("Average kernel execution time : %f (us)\n", total_ns * 1.0e-3 / max(1, repeat))

    h_output = Array(d_array)
    h_pivot = Array(pivot)
    h_info = Array(info)

    println("> verifying the result..")
    err_count = 0
    @inbounds for b in 1:BATCH_SIZE
        if h_info[b] == 0
            a = @view h_input[:, :, b]
            lu_mat = @view h_output[:, :, b]
            l = Matrix{Float32}(I, N, N)
            l .+= tril(Matrix(lu_mat), -1)
            u = triu(Matrix(lu_mat))
            pmat = permutation_matrix_from_pivot(@view h_pivot[:, b])
            pxa = pmat * Matrix(a)
            lxu = l * u
            if !check_relative_error(pxa, lxu, MAX_ERROR)
                @printf("> ERROR: accuracy check failed for matrix number %05d..\n", b)
                err_count += 1
            end
        elseif h_info[b] > 0
            @printf("> execution for matrix %05d is successful, but U is singular and U(%d,%d) = 0..\n",
                    b, h_info[b] - 1, h_info[b] - 1)
        else
            @printf("> ERROR: matrix %05d have an illegal value at index %d = %lf..\n",
                    b, -h_info[b], Float64(h_input[-h_info[b], b]))
            err_count += 1
        end
    end

    if err_count > 0
        @printf("> TEST FAILED for %d matrices, with precision: %g\n", err_count, MAX_ERROR)
        exit(1)
    end
    @printf("> TEST SUCCESSFUL, with precision: %g\n", MAX_ERROR)
end

main()
