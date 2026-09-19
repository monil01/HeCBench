using CUDA
using Printf
using Random

function cross_kernel!(n::Int32, out, x1, x2, ostride::Int32, x1stride::Int32, x2stride::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while i < n
        base = Int32(3) * i + Int32(1)
        o = base
        a = base
        b = base
        @inbounds begin
            val0 = x1[a + Int32(1) * x1stride] * x2[b + Int32(2) * x2stride] -
                   x1[a + Int32(2) * x1stride] * x2[b + Int32(1) * x2stride]
            val1 = x1[a + Int32(2) * x1stride] * x2[b + Int32(0) * x2stride] -
                   x1[a + Int32(0) * x1stride] * x2[b + Int32(2) * x2stride]
            val2 = x1[a + Int32(0) * x1stride] * x2[b + Int32(1) * x2stride] -
                   x1[a + Int32(1) * x1stride] * x2[b + Int32(0) * x2stride]
            out[o + Int32(0) * ostride] = val0
            out[o + Int32(1) * ostride] = val1
            out[o + Int32(2) * ostride] = val2
        end
        i += stride
    end
    return
end

function cross2_kernel!(n::Int32, out, x1, x2, ostride::Int32, x1stride::Int32, x2stride::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while i < n
        base = Int32(3) * i + Int32(1)
        @inbounds begin
            x1c0 = x1[base + Int32(0) * x1stride]
            x1c1 = x1[base + Int32(1) * x1stride]
            x1c2 = x1[base + Int32(2) * x1stride]
            x2c0 = x2[base + Int32(0) * x2stride]
            x2c1 = x2[base + Int32(1) * x2stride]
            x2c2 = x2[base + Int32(2) * x2stride]
            out[base + Int32(0) * ostride] = x1c1 * x2c2 - x1c2 * x2c1
            out[base + Int32(1) * ostride] = x1c2 * x2c0 - x1c0 * x2c2
            out[base + Int32(2) * ostride] = x1c0 * x2c1 - x1c1 * x2c0
        end
        i += stride
    end
    return
end

function cross3_kernel!(n::Int32, out, x1, x2)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while i < n
        base = Int32(3) * i + Int32(1)
        @inbounds begin
            x1c0 = x1[base]
            x1c1 = x1[base + Int32(1)]
            x1c2 = x1[base + Int32(2)]
            x2c0 = x2[base]
            x2c1 = x2[base + Int32(1)]
            x2c2 = x2[base + Int32(2)]
            out[base] = x1c1 * x2c2 - x1c2 * x2c1
            out[base + Int32(1)] = x1c2 * x2c0 - x1c0 * x2c2
            out[base + Int32(2)] = x1c0 * x2c1 - x1c1 * x2c0
        end
        i += stride
    end
    return
end

function timed!(f, args...; repeat)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        f(args...)
    end
    CUDA.synchronize()
    (time_ns() - t0) * 1.0e-3 / repeat
end

function eval_case(::Type{T}, nrows::Int, repeat::Int) where {T}
    rng = MersenneTwister(123)
    num_elems = nrows * 3
    a = T.(rand(rng, Float32, num_elems) .* 4.0f0 .- 2.0f0)
    b = T.(rand(rng, Float32, num_elems) .* 4.0f0 .- 2.0f0)
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_o = CUDA.zeros(T, num_elems)
    threads = 256
    blocks = cld(nrows, threads)

    t = timed!(repeat=repeat) do
        @cuda threads=threads blocks=blocks cross_kernel!(Int32(nrows), d_o, d_a, d_b, Int32(1), Int32(1), Int32(1))
    end
    @printf("Average execution time of cross1 kernel: %f (us)\n", t)
    o = Array(d_o)

    t = timed!(repeat=repeat) do
        @cuda threads=threads blocks=blocks cross2_kernel!(Int32(nrows), d_o, d_a, d_b, Int32(1), Int32(1), Int32(1))
    end
    @printf("Average execution time of cross2 kernel: %f (us)\n", t)
    o2 = Array(d_o)

    t = timed!(repeat=repeat) do
        @cuda threads=threads blocks=blocks cross3_kernel!(Int32(nrows), d_o, d_a, d_b)
    end
    @printf("Average execution time of cross3 kernel: %f (us)\n", t)
    o3 = Array(d_o)

    ok = true
    for i in 1:num_elems
        if abs(o[i] - o2[i]) > T(1.0e-3) || abs(o[i] - o3[i]) > T(1.0e-3)
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <number of rows in a 2D tensor> <repeat>")
        return 1
    end
    nrows = parse(Int, args[1])
    repeat = parse(Int, args[2])

    println("=========== Data type is FP32 ==========")
    ok = eval_case(Float32, nrows, repeat)
    println("=========== Data type is FP64 ==========")
    ok &= eval_case(Float64, nrows, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
