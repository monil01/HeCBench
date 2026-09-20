using CUDA
using CUDA.CUBLAS
using Printf
using LinearAlgebra

function convert_value(::Type{T}, x::Real) where {T<:Real}
    return T(x)
end

function convert_value(::Type{T}, x::Real) where {T<:Complex}
    return T(0, x)
end

function print_value(x::Real)
    @printf("%6.2f", x)
end

function print_value(x::Complex)
    @printf("<%6.2f,%6.2f> ", real(x), imag(x))
end

function host_inputs(::Type{T}) where {T}
    v(x) = convert_value(T, x)
    avec = T[
        v(1.0), v(0.0), v(0.0), v(0.0), v(0.0),
        v(1.0), v(0.2), v(-0.4), v(-0.4), v(-0.8),
        v(1.0), v(0.6), v(-0.2), v(0.4), v(-1.2),
        v(1.0), v(1.0), v(-1.0), v(0.6), v(-0.8),
        v(1.0), v(1.8), v(-0.6), v(0.2), v(-0.6),
        v(0.2), v(-0.4), v(-0.4), v(-0.8), v(0.0),
        v(0.4), v(0.2), v(0.8), v(-0.4), v(0.0),
        v(0.4), v(-0.8), v(0.2), v(0.4), v(0.0),
        v(0.8), v(0.4), v(-0.4), v(0.2), v(0.0),
        v(0.0), v(0.0), v(0.0), v(0.0), v(1.0),
    ]
    bvec = T[
        v(5.0), v(3.6), v(-2.2), v(0.8), v(-3.4),
        v(1.8), v(-0.6), v(0.2), v(-0.6), v(1.0),
    ]
    A = [reshape(copy(avec[1:25]), 5, 5), reshape(copy(avec[26:50]), 5, 5)]
    B = [reshape(copy(bvec[1:5]), 5, 1), reshape(copy(bvec[6:10]), 5, 1)]
    return A, B
end

function run_gels_batch_example(::Type{T}, repeat::Int) where {T}
    A_host, B_host = host_inputs(T)
    X = fill(T(1), 10)
    bound = real(T) == Float32 ? 1e-6 : 1e-8

    A_dev = CuArray{T, 2}[]
    B_dev = CuArray{T, 2}[]

    elapsed_ns = 0
    for i in 0:repeat
        A_dev = [CuArray(A_host[1]), CuArray(A_host[2])]
        B_dev = [CuArray(B_host[1]), CuArray(B_host[2])]

        CUDA.synchronize()
        start = time_ns()
        CUBLAS.gels_batched!('N', A_dev, B_dev)
        CUDA.synchronize()
        if i != 0
            elapsed_ns += time_ns() - start
        end
    end
    @printf("Average kernel execution time : %f (us)\n", (elapsed_ns * 1e-3) / repeat)

    B_result = reduce(vcat, vec.(Array.(B_dev)))
    passed = true
    println("Results:")
    for batch in 0:1
        for j in 1:5
            result = B_result[batch * 5 + j]
            residual = result - X[batch * 5 + j]
            passed &= result == result
            passed &= sqrt(abs(real(residual * residual))) < bound
            print_value(result)
        end
        println()
    end

    if passed
        println("Calculations successfully finished")
        return false
    else
        println("ERROR: results mismatch!")
        println("Expected:")
        for batch in 0:1
            for j in 1:5
                print_value(X[batch * 5 + j])
            end
            println()
        end
        return true
    end
end

function print_info()
    println("")
    println("########################################################################")
    println("# Batched strided GELS example:")
    println("# ")
    println("# Computes least squares of a batch of matrices and right hand sides.")
    println("# Supported floating point type precisions:")
    println("#   float")
    println("#   double")
    println("#   std::complex<float>")
    println("#   std::complex<double>")
    println("# ")
    println("########################################################################")
    println("")
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    print_info()

    failed = false
    println("Running with single precision real data type:")
    failed |= run_gels_batch_example(Float32, repeat)
    println("Running with single precision complex data type:")
    failed |= run_gels_batch_example(ComplexF32, repeat)
    println("Running with double precision real data type:")
    failed |= run_gels_batch_example(Float64, repeat)
    println("Running with double precision complex data type:")
    failed |= run_gels_batch_example(ComplexF64, repeat)
    exit(failed ? 1 : 0)
end

main()
