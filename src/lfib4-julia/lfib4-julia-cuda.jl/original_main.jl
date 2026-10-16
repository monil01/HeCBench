using CUDA
using Printf
using Random

const P1 = 55
const P2 = 119
const P3 = 179
const P4 = 256

function lfib4!(x::Vector{UInt32}, n::Int)
    for k in P4+1:n
        @inbounds x[k] = x[k - P1] + x[k - P2] + x[k - P3] + x[k - P4]
    end
    return x
end

function lfib4_kernel!(x, n::UInt32)
    if blockIdx().x == Int32(1) && threadIdx().x == Int32(1)
        for k in UInt32(P4 + 1):n
            @inbounds x[k] = x[k - UInt32(P1)] + x[k - UInt32(P2)] +
                             x[k - UInt32(P3)] + x[k - UInt32(P4)]
        end
    end
    return
end

function glfib4(seed::Vector{UInt32}, n::Int, total_len::Int)
    d_x = CUDA.zeros(UInt32, total_len)
    CUDA.copyto!(view(d_x, 1:P4), seed)
    CUDA.synchronize()
    start = time_ns()
    @cuda threads=1 blocks=1 lfib4_kernel!(d_x, UInt32(n))
    CUDA.synchronize()
    elapsed = (time_ns() - start) * 1e-9
    return Array(d_x), elapsed
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <n>")
        return 1
    end

    n = parse(Int, ARGS[1])
    rng = MersenneTwister(1234)
    x = Vector{UInt32}(undef, n)

    r = 16
    while r <= 4096
        s = div(n, r)
        s -= (s % 256 == 0 ? 0 : s % 256)
        while s * r < n
            r += 1
        end

        @printf("n=%d r=%d s=%d\n", n, r, s)
        total_len = r * s
        z = zeros(UInt32, total_len)

        for k in 1:P4
            v = rand(rng, UInt32)
            x[k] = v
            z[k] = v
        end

        start = time_ns()
        lfib4!(x, n)
        host_time = (time_ns() - start) * 1e-9

        gpu_out, device_time = glfib4(z[1:P4], n, total_len)
        speedup = device_time == 0 ? Inf : host_time / device_time
        @printf("r = %d | host time = %lf | device time = %lf | speedup = %.1f ",
                r, host_time, device_time, speedup)

        ok = true
        @inbounds for i in 1:n
            if x[i] != gpu_out[i]
                ok = false
                break
            end
        end
        println("check = ", ok ? "PASS" : "FAIL")
        r *= 2
    end
    return 0
end

exit(main())
