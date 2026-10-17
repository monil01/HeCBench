using CUDA
using Printf
using Random

function parse_args(args)
    lower = 2
    upper = 100
    num = 25000
    reps = 10
    verbose = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-l"
            i += 1; lower = parse(Int, args[i])
        elseif arg == "-u"
            i += 1; upper = parse(Int, args[i])
        elseif arg == "-n"
            i += 1; num = parse(Int, args[i])
        elseif arg == "-r"
            i += 1; reps = parse(Int, args[i])
        elseif arg == "-v"
            verbose = true
        else
            println(stderr, "invalid argument: ", arg)
            exit(1)
        end
        i += 1
    end
    return lower, upper, num, reps, verbose
end

function cxx_rand_values(::Type{T}, n::Int) where {T}
    rng = MersenneTwister(48)
    return T.(rand(rng, n))
end

function performance(m, n, k, time_ns)
    flops = 2.0 * m * n * k
    gflops = flops / time_ns
    @printf("%f GFLOP/s\n", gflops)
end

function gemm_ref!(result_ref, matrices, vectors, upper, m, k, num)
    @inbounds for batch in 1:num
        a0 = (batch - 1) * upper * upper
        b0 = (batch - 1) * upper
        c0 = (batch - 1) * upper
        for row in 1:m
            acc = 0.0
            for col in 1:k
                acc += Float64(matrices[a0 + row + (col - 1) * upper]) *
                       Float64(vectors[b0 + col])
            end
            result_ref[c0 + row] = acc
        end
    end
end

function gemm_batched(::Type{T}, lower, upper, num, reps, verbose) where {T}
    verbose && println("initializing inputs")
    matrices = cxx_rand_values(T, upper * upper * num)
    vectors = cxx_rand_values(T, upper * num)
    result = Vector{T}(undef, upper * num)
    result_ref = Vector{Float64}(undef, upper * num)

    verbose && println("allocating device variables")
    dev_matrices = CuArray(reshape(matrices, upper, upper, num))
    dev_vectors = CuArray(reshape(vectors, upper, 1, num))
    dev_result = CUDA.zeros(T, upper, 1, num)

    for size in lower:upper
        verbose && println("running with <size x size> x <size x 1> ", size)
        sum_us = 0.0
        m = size
        n = 1
        k = size
        A = @view dev_matrices[1:m, 1:k, :]
        B = @view dev_vectors[1:k, 1:n, :]
        C = @view dev_result[1:m, 1:n, :]
        for rep in 0:reps
            start = time_ns()
            CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, B, zero(T), C)
            CUDA.synchronize()
            elapsed_us = (time_ns() - start) * 1.0e-3
            rep != 0 && (sum_us += elapsed_us)
            verbose && @printf("size %d: %f us; %f us per operation\n",
                               size, elapsed_us, elapsed_us / num)
        end
        avg = sum_us / reps
        @printf("size %d average execution time: %f us; %f us per operation; floating-point operations per second: ",
                size, avg, avg / num)
        performance(m, n, k, 1.0e3 * (avg / num))

        if T == Float64
            copyto!(result, vec(Array(dev_result)))
            gemm_ref!(result_ref, matrices, vectors, upper, m, k, num)
            mismatch_reported = false
            for batch in 1:num, row in 1:m
                idx = (batch - 1) * upper + row
                if abs(Float64(result[idx]) - result_ref[idx]) > 1.0e-6
                    println("Mismatch at batch index ", batch - 1, ": ",
                            result[idx], "!=", result_ref[idx])
                    mismatch_reported = true
                    break
                end
            end
            mismatch_reported && break
        end
    end
end

function main()
    lower, upper, num, reps, verbose = parse_args(ARGS)
    println("running with lower: ", lower, " upper: ", upper, " num: ", num, " reps: ", reps)
    println(">>>>>>>>>>>>>>> Half precision gemmBatched >>>>>>>>>>>>>>> ")
    gemm_batched(Float16, lower, upper, num, reps, verbose)
    println(">>>>>>>>>>>>>>> Single precision gemmBatched >>>>>>>>>>>>>>> ")
    gemm_batched(Float32, lower, upper, num, reps, verbose)
    println(">>>>>>>>>>>>>>> Double precision gemmBatched >>>>>>>>>>>>>>> ")
    gemm_batched(Float64, lower, upper, num, reps, verbose)
end

main()
