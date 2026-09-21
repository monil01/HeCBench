using CUDA
using Printf

# Julia port of langevin-cuda.  The three kernels match k0/k1/k2 from the
# CUDA source and the host computes the same series-reference error summary.

const THREADS = 256

function k0_kernel!(a, o, n::Int32)
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if t <= n
        x = @inbounds a[t]
        @inbounds o[t] = cosh(x) / sinh(x) - 1.0f0 / x
    end
    return
end

function k1_kernel!(a, o, n::Int32)
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if t <= n
        x = @inbounds a[t]
        @inbounds o[t] = 1.0f0 / tanh(x) - 1.0f0 / x
    end
    return
end

@inline function fmaf32(a::Float32, b::Float32, c::Float32)
    return fma(a, b, c)
end

function k2_kernel!(a, o, n::Int32)
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if t <= n
        x = @inbounds a[t]
        s = x * x
        r = 7.70960469f-8
        r = fmaf32(r, s, -1.65101926f-6)
        r = fmaf32(r, s, 2.03457112f-5)
        r = fmaf32(r, s, -2.10521728f-4)
        r = fmaf32(r, s, 2.11580913f-3)
        r = fmaf32(r, s, -2.22220998f-2)
        r = fmaf32(r, s, 8.33333284f-2)
        r = fmaf32(r, x, 0.25f0 * x)
        @inbounds o[t] = r
    end
    return
end

function series_reference(a::Vector{Float32})
    o = Vector{Float32}(undef, length(a))
    for i in eachindex(a)
        x = a[i]
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        o[i] = x * (1.0f0 / 3.0f0 - 1.0f0 / 45.0f0 * x2 +
                    2.0f0 / 945.0f0 * x4 - 1.0f0 / 4725.0f0 * x6)
    end
    return o
end

function main()
    if length(ARGS) != 2
        println("Usage main.jl <n> <repeat>")
        return 1
    end

    n = parse(Int, ARGS[1])
    repeat_n = parse(Int, ARGS[2])

    a = Vector{Float32}(undef, n)
    step = 1.79999f0 / Float32(n)
    for i in 0:n-1
        a[i + 1] = -1.8f0 + Float32(i) * step
    end

    d_a = CuArray(a)
    d_o0 = CUDA.zeros(Float32, n)
    d_o1 = CUDA.zeros(Float32, n)
    d_o2 = CUDA.zeros(Float32, n)
    blocks = cld(n, THREADS)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=THREADS blocks=blocks k0_kernel!(d_a, d_o0, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average execution time of k0: %f (s)\n",
            (time_ns() - t0) * 1e-9 / repeat_n)

    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=THREADS blocks=blocks k1_kernel!(d_a, d_o1, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average execution time of k1: %f (s)\n",
            (time_ns() - t0) * 1e-9 / repeat_n)

    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=THREADS blocks=blocks k2_kernel!(d_a, d_o2, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average execution time of k2: %f (s)\n",
            (time_ns() - t0) * 1e-9 / repeat_n)

    o = series_reference(a)
    o0 = Array(d_o0)
    o1 = Array(d_o1)
    o2 = Array(d_o2)

    e0 = 0.0f0
    e1 = 0.0f0
    e2 = 0.0f0
    for i in 1:n
        e0 += (o[i] - o0[i]) * (o[i] - o0[i])
        e1 += (o[i] - o1[i]) * (o[i] - o1[i])
        e2 += (o[i] - o2[i]) * (o[i] - o2[i])
    end

    println()
    println("Error statistics for the kernels:")
    @printf("%f %f %f \n", sqrt(e0), sqrt(e1), sqrt(e2))

    cpu0 = similar(a)
    cpu1 = similar(a)
    cpu2 = similar(a)
    for i in eachindex(a)
        x = a[i]
        cpu0[i] = cosh(x) / sinh(x) - 1.0f0 / x
        cpu1[i] = 1.0f0 / tanh(x) - 1.0f0 / x
        s = x * x
        r = 7.70960469f-8
        r = fma(r, s, -1.65101926f-6)
        r = fma(r, s, 2.03457112f-5)
        r = fma(r, s, -2.10521728f-4)
        r = fma(r, s, 2.11580913f-3)
        r = fma(r, s, -2.22220998f-2)
        r = fma(r, s, 8.33333284f-2)
        cpu2[i] = fma(r, x, 0.25f0 * x)
    end

    ok = isapprox(o0, cpu0; rtol=1.0f-5, atol=1.0f-5) &&
         isapprox(o1, cpu1; rtol=1.0f-5, atol=1.0f-5) &&
         isapprox(o2, cpu2; rtol=1.0f-5, atol=1.0f-5)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main())
