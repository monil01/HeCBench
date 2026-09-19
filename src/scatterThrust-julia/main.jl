using CUDA
using Printf

function scatter_kernel!(out, src, idx, n::Int64)
    i = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
    stride = Int64(gridDim().x * blockDim().x)
    while i <= n
        @inbounds out[idx[i] + Int64(1)] = src[i]
        i += stride
    end
    return
end

value_for(::Type{Int8}, i::Int64) = reinterpret(Int8, UInt8(i % 256))
value_for(::Type{Int16}, i::Int64) = reinterpret(Int16, UInt16(i % 65536))
value_for(::Type{Int32}, i::Int64) = Int32(i)
value_for(::Type{Int64}, i::Int64) = i
value_for(::Type{Float32}, i::Int64) = Float32(i)
value_for(::Type{Float64}, i::Int64) = Float64(i)

function scatter_case(::Type{T}, n::Int, repeat::Int) where {T}
    src = Vector{T}(undef, n)
    idx = Vector{Int64}(undef, n)
    for i in 1:n
        src[i] = value_for(T, Int64(i - 1))
        idx[i] = Int64(n - i)
    end

    d_src = CuArray(src)
    d_idx = CuArray(idx)
    d_out = CUDA.zeros(T, n)
    threads = 256
    blocks = min(cld(n, threads), 4096)

    @cuda threads=threads blocks=blocks scatter_kernel!(d_out, d_src, d_idx, Int64(n))
    CUDA.synchronize()

    total_ns = 0
    for _ in 1:repeat
        fill!(d_out, zero(T))
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=threads blocks=blocks scatter_kernel!(d_out, d_src, d_idx, Int64(n))
        CUDA.synchronize()
        total_ns += time_ns() - start
    end
    @printf("Average execution time of thrust::scatter: %f (us)\n",
            total_ns * 1.0e-3 / repeat)

    out = Array(d_out)
    ok = true
    for i in 1:n
        if out[i] != value_for(T, Int64(n - i))
            ok = false
            break
        end
    end
    println(ok ? "PASS\n" : "FAIL\n")
    return ok
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    n = parse(Int, args[1])
    repeat = parse(Int, args[2])

    ok = true
    println("INT8 scatter")
    ok &= scatter_case(Int8, n, repeat)
    println("INT16 scatter")
    ok &= scatter_case(Int16, n, repeat)
    println("INT32 scatter")
    ok &= scatter_case(Int32, n, repeat)
    println("INT64 scatter")
    ok &= scatter_case(Int64, n, repeat)
    println("FP32 scatter")
    ok &= scatter_case(Float32, n, repeat)
    println("FP64 scatter")
    ok &= scatter_case(Float64, n, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
