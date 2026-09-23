using CUDA
using LinearAlgebra
using Printf
using Random

function dot_product_kernel!(a, b, partial, n::Int64)
    tid0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1)))
    stride = Int64(gridDim().x * blockDim().x)
    acc = zero(eltype(a))
    idx = tid0
    while idx < n
        acc += a[idx + 1] * b[idx + 1]
        idx += stride
    end
    partial[tid0 + 1] = acc
    return
end

function round_up(local_size::Int, n::Int)
    return cld(n, local_size) * local_size
end

function make_data(::Type{T}, n::Int, src_size::Int) where {T}
    rng = MersenneTwister(19937)
    a = zeros(T, src_size)
    b = zeros(T, src_size)
    ref = zero(T)
    for i in 1:n
        av = T(rand(rng, -32:32))
        bv = T(rand(rng, -32:32))
        a[i] = av
        b[i] = bv
        ref += av * bv
    end
    return a, b, ref
end

function passclose(x::T, ref::T) where {T}
    if T === Float32
        return abs(x - ref) <= max(T(1), abs(ref)) * T(1.0f-4)
    end
    return abs(x - ref) <= max(T(1), abs(ref)) * T(1.0e-10)
end

function run_dot(::Type{T}, n::Int, repeat::Int) where {T}
    local_size = 1024
    global_size = round_up(local_size, n)
    src_size = global_size
    grid_size = max(1, cld(global_size, local_size * 4))
    work_items = grid_size * local_size

    @printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n", global_size, local_size)

    a, b, ref = make_data(T, n, src_size)
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_partial = CUDA.zeros(T, work_items)

    for _ in 1:min(repeat, 100)
        @cuda threads=local_size blocks=grid_size dot_product_kernel!(d_a, d_b, d_partial, Int64(n))
        CUDA.sum(d_partial)
    end

    CUDA.synchronize()
    start = time_ns()
    dst = zero(T)
    for _ in 1:repeat
        @cuda threads=local_size blocks=grid_size dot_product_kernel!(d_a, d_b, d_partial, Int64(n))
        dst = CUDA.@allowscalar CUDA.sum(d_partial)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6 / repeat
    @printf("Average kernel execution time %f (ms)\n", elapsed_ms)
    println(passclose(dst, ref) ? "PASS" : "FAIL")
    println()

    for _ in 1:min(repeat, 100)
        dot(d_a[1:n], d_b[1:n])
    end
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        dst = CUDA.@allowscalar dot(d_a[1:n], d_b[1:n])
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6 / repeat
    @printf("Average cublasDotEx execution time %f (ms)\n", elapsed_ms)
    println(passclose(dst, ref) ? "PASS" : "FAIL")
    println()

    for _ in 1:min(repeat, 100)
        CUDA.sum(d_a[1:n] .* d_b[1:n])
    end
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        dst = CUDA.@allowscalar CUDA.sum(d_a[1:n] .* d_b[1:n])
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6 / repeat
    @printf("Average std::transform_reduce execution time %f (ms)\n", elapsed_ms)
    println(passclose(dst, ref) ? "PASS" : "FAIL")
    println()
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        exit(1)
    end
    n = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])

    println("------------- Data type is Float32 ---------------")
    run_dot(Float32, n, repeat)
    println("------------- Data type is Float64 ---------------")
    run_dot(Float64, n, repeat)
end

main()
