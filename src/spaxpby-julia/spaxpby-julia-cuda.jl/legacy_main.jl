using CUDA
using Printf

function libc_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function libc_srand48(seed::Int32)
    ccall(:srand48, Cvoid, (Clong,), seed)
end

function libc_drand48()
    ccall(:drand48, Cdouble, ())
end

function init_matrix(num_rows::Int, num_cols::Int, nnz::Int)
    total = num_rows * num_cols
    order = Float32.(0:total-1)
    libc_srand(UInt32(123))
    for i in total:-1:1
        a = i
        b = mod(libc_rand(), Cint(i)) + 1
        if a != b
            order[a], order[b] = order[b], order[a]
        end
    end

    vals = Vector{Float32}(undef, total)
    libc_srand48(Int32(123))
    @inbounds for i in 1:total
        vals[i] = order[i] >= nnz ? 0.0f0 : Float32(libc_drand48() + 1.0)
    end
    return vals
end

function scale_kernel!(y, beta::Float32, len::Int32)
    tid = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    i = tid
    @inbounds while i <= len
        y[i] = beta * y[i]
        i += stride
    end
    return
end

function axpby_kernel!(y, indices, values, alpha::Float32, nnz::Int32)
    tid = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    j = tid
    @inbounds while j <= nnz
        y[indices[j] + Int64(1)] += alpha * values[j]
        j += stride
    end
    return
end

function main()
    if length(ARGS) != 4
        println("The function computes the sum of a sparse vector and a dense vector in single-precision floating-point operations")
        println("for i=0 to n-1        ")
        println("    Y[i] = beta * Y[i]")
        println("for i=0 to nnz-1      ")
        println("    Y[X_indices[i]] += alpha * X_values[i]")
        println()
        println("Usage main.jl <M> <N> <nnz> <repeat>")
        println("The size of the vector (n) is M * N")
        println("nnz is the number of non-zero elements")
        return 1
    end

    m = parse(Int, ARGS[1])
    n = parse(Int, ARGS[2])
    nnz = parse(Int, ARGS[3])
    repeat_n = parse(Int, ARGS[4])
    size = m * n

    println("Initializing input matrices..")
    hA = init_matrix(m, n, nnz)
    hA_indices = Int64[]
    hA_values = Float32[]
    @inbounds for i in 1:size
        if hA[i] != 0.0f0
            push!(hA_indices, Int64(i - 1))
            push!(hA_values, hA[i])
        end
    end
    actual_nnz = length(hA_values)
    hB = init_matrix(m, n, size)
    println("Done")

    alpha = 1.0f0
    beta = 1.0f0
    dY = CuArray(hB)
    d_indices = CuArray(hA_indices)
    d_values = CuArray(hA_values)

    threads = 256
    blocks = max(cld(max(size, actual_nnz), threads), 1)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks scale_kernel!(dY, beta, Int32(size))
        @cuda threads=threads blocks=blocks axpby_kernel!(dY, d_indices, d_values, alpha, Int32(actual_nnz))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time of SPAXPBY : %f (us)\n", elapsed_us)

    hY = Array(dY)
    println("Computing the reference results..")
    hRef = copy(hB)
    for _ in 1:repeat_n
        @inbounds for i in 1:size
            hRef[i] = alpha * hA[i] + beta * hRef[i]
        end
    end
    println("Done")

    correct = all(abs.(hY .- hRef) .<= 1.0f-2)
    if correct
        println("axpby_example test PASSED")
        println("PASS")
    else
        println("axpby_example test FAILED: wrong result")
        println("FAIL")
    end
    return correct ? 0 : 1
end

exit(main())
