using CUDA
using Printf

# Julia port of maxFlops-cuda.  The CUDA version uses macro-generated kernels
# for arithmetic throughput variants; this port preserves the same benchmark
# labels with a generic CUDA.jl arithmetic kernel parameterized by operation and
# independent accumulator count.

const BLOCK_SIZE = 256
const NUM_FLOATS = 2 * 1024 * 1024

function flops_kernel!(data, n_iters::Int32, op::Int32, lanes::Int32, v1, v2)
    gid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if gid > length(data)
        return
    end

    base = @inbounds data[gid]
    s1 = base
    s2 = oftype(base, 10.0) - base
    s3 = oftype(base, 9.0) - base
    s4 = oftype(base, 9.0) - s2
    s5 = oftype(base, 8.0) - base
    s6 = oftype(base, 8.0) - s2
    s7 = oftype(base, 7.0) - base
    s8 = oftype(base, 7.0) - s2
    if op == Int32(2)
        s1 = base - base + oftype(base, 0.999)
        s2 = s1 - oftype(base, 0.0001)
        s3 = s1 - oftype(base, 0.0002)
        s4 = s1 - oftype(base, 0.0003)
        s5 = s1 - oftype(base, 0.0004)
        s6 = s1 - oftype(base, 0.0005)
        s7 = s1 - oftype(base, 0.0006)
        s8 = s1 - oftype(base, 0.0007)
    end

    j = Int32(0)
    while j < n_iters
        rep = Int32(0)
        while rep < Int32(120)
            if op == Int32(1)
                s1 = v1 - s1
                if lanes >= Int32(2); s2 = v1 - s2; end
                if lanes >= Int32(4); s3 = v1 - s3; s4 = v1 - s4; end
                if lanes >= Int32(8); s5 = v1 - s5; s6 = v1 - s6; s7 = v1 - s7; s8 = v1 - s8; end
            elseif op == Int32(2)
                s1 = s1 * s1 * v1
                if lanes >= Int32(2); s2 = s2 * s2 * v1; end
                if lanes >= Int32(4); s3 = s3 * s3 * v1; s4 = s4 * s4 * v1; end
                if lanes >= Int32(8); s5 = s5 * s5 * v1; s6 = s6 * s6 * v1; s7 = s7 * s7 * v1; s8 = s8 * s8 * v1; end
            elseif op == Int32(3)
                s1 = v1 - s1 * v2
                if lanes >= Int32(2); s2 = v1 - s2 * v2; end
                if lanes >= Int32(4); s3 = v1 - s3 * v2; s4 = v1 - s4 * v2; end
                if lanes >= Int32(8); s5 = v1 - s5 * v2; s6 = v1 - s6 * v2; s7 = v1 - s7 * v2; s8 = v1 - s8 * v2; end
            else
                s1 = (v1 - v2 * s1) * s1
                if lanes >= Int32(2); s2 = (v1 - v2 * s2) * s2; end
                if lanes >= Int32(4); s3 = (v1 - v2 * s3) * s3; s4 = (v1 - v2 * s4) * s4; end
                if lanes >= Int32(8); s5 = (v1 - v2 * s5) * s5; s6 = (v1 - v2 * s6) * s6; s7 = (v1 - v2 * s7) * s7; s8 = (v1 - v2 * s8) * s8; end
            end
            rep += Int32(1)
        end
        j += Int32(1)
    end

    out = s1
    if lanes >= Int32(2); out += s2; end
    if lanes >= Int32(4); out += s3 + s4; end
    if lanes >= Int32(8); out += s5 + s6 + s7 + s8; end
    @inbounds data[gid] = out
    return
end

function make_host(::Type{T}, n::Int) where {T}
    data = Vector{T}(undef, n)
    state = UInt64(123)
    for j in 1:(n ÷ 2)
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        v = T(Float64(state >> 11) / Float64(UInt64(1) << 53) * 10.0)
        data[j] = v
        data[n - j + 1] = v
    end
    return data
end

function run_kernel!(label::String, op::Int, lanes::Int, d_mem, host, repeat_n::Int, v1, v2)
    copyto!(d_mem, host)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=BLOCK_SIZE blocks=(length(host) ÷ BLOCK_SIZE) flops_kernel!(
        d_mem, Int32(repeat_n), Int32(op), Int32(lanes), v1, v2)
    CUDA.synchronize()
    @printf("kernel execution time (%s): %f (s)\n", label, (time_ns() - t0) * 1e-9)
end

function test_type(::Type{T}, repeat_n::Int) where {T}
    host = make_host(T, NUM_FLOATS)
    d_mem = CuArray(host)
    for (prefix, op, v1, v2) in (
        ("Add", 1, T(10.0), T(0.0)),
        ("Mul", 2, T(1.01), T(0.0)),
        ("MAdd", 3, T(10.0), T(0.9899)),
        ("MulMAdd", 4, T(3.75), T(0.355)),
    )
        for lanes in (1, 2, 4, 8)
            run_kernel!("$(prefix)$(lanes)", op, lanes, d_mem, host, repeat_n, v1, v2)
        end
    end
    CUDA.synchronize()
    return true
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat_n = parse(Int, ARGS[1])

    println("=== Single-precision floating-point kernels ===")
    ok = test_type(Float32, repeat_n)
    println("=== Double-precision floating-point kernels ===")
    ok &= test_type(Float64, repeat_n)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main())
