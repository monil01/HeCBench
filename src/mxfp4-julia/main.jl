using CUDA
using Printf
using Random

const TABLE = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
                      0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
                      0.1, 0.3, 0.4, 0.6, 0.9,
                      -0.1, -0.3, -0.4, -0.6, -0.9]

@inline function fp4_round(x::Float32)
    if x == 0.25f0
        return 0.0f0
    elseif x == 0.75f0 || x == 1.25f0
        return 1.0f0
    elseif x == 1.75f0 || x == 2.5f0
        return 2.0f0
    elseif x == 3.5f0 || x == 5.0f0
        return 4.0f0
    elseif x > 0.5f0 && x < 0.75f0
        return 0.5f0
    elseif x > 0.0f0 && x < 0.25f0
        return 0.0f0
    elseif x > 0.25f0 && x < 0.5f0
        return 0.5f0
    elseif x > 0.75f0 && x < 1.0f0
        return 1.0f0
    elseif x < -0.5f0 && x > -0.75f0
        return -0.5f0
    elseif x < 0.0f0 && x > -0.25f0
        return -0.0f0
    elseif x < -0.25f0 && x > -0.5f0
        return -0.5f0
    elseif x < -0.75f0 && x > -1.0f0
        return -1.0f0
    else
        return x
    end
end

function qdq_kernel!(src, dst, n::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    while i <= n
        @inbounds dst[i] = fp4_round(Float32(src[i]))
        i += stride
    end
    return
end

function verify(src, dst)
    ok = true
    @inbounds for i in eachindex(src)
        s = Float32(src[i])
        d = Float32(dst[i])
        expected = fp4_round(s)
        if abs(d - expected) > 1.0f-3
            @printf("%f %f\n", s, d)
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

function run_sim(label::String, numel::Int, niters::Int)
    println()
    println(label)
    rng = MersenneTwister(123)
    src = Vector{Float16}(undef, numel)
    @inbounds for i in 0:(numel ÷ 32 - 1)
        for j in 1:32
            src[i * 32 + j] = Float16(TABLE[rand(rng, 1:length(TABLE))])
        end
        r = rand(rng, 0:31)
        src[i * 32 + r + 1] = Float16(r < 16 ? 6.0f0 : -6.0f0)
    end
    d_src = CuArray(src)
    d_dst = similar(d_src)
    threads = (numel % 128 == 0) ? 128 : 64
    blocks = max(numel ÷ threads, 1)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:niters
        @cuda threads=threads blocks=blocks qdq_kernel!(d_src, d_dst, Int32(numel))
    end
    CUDA.synchronize()
    elapsed = (time_ns() - t0) * 1e-9 / niters
    size_gb = 2.0 * 2 * numel / 1.0e9
    @printf("size(GB):%.2f, average time(sec):%f, Bandwidth (GB/sec):%f\n",
            size_gb, elapsed, size_gb / elapsed)
    verify(src, Array(d_dst))
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    numel = parse(Int, args[1])
    niters = parse(Int, args[2])
    if numel % 64 != 0
        println("Expected qdq_mxfp4 input number of elements to be a multiple of 64, but it is not!")
        return 1
    end
    run_sim("Simulate conversion between FP16 and FP4", numel, niters)
    run_sim("Simulate conversion between BF16 and FP4", numel, niters)
    return 0
end

exit(main(ARGS))
