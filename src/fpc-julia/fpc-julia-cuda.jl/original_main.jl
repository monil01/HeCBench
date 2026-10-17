using CUDA
using Printf
using Random

@inline function my_abs_i32(x::Int32)
    mask = x < 0 ? UInt32(0xffffffff) : UInt32(0)
    return (reinterpret(UInt32, x) ⊻ mask) - mask
end

@inline function signed_low32(v::UInt64)
    return reinterpret(Int32, UInt32(v & UInt64(0xffffffff)))
end

@inline function fpc_increment(v::UInt64)
    if v == UInt64(0)
        return Int32(1)
    end

    if my_abs_i32(signed_low32(v)) <= UInt32(0xff)
        return Int32(1)
    end

    if my_abs_i32(signed_low32(v)) <= UInt32(0xffff)
        return Int32(2)
    end

    if (v & UInt64(0xffff)) == UInt64(0)
        return Int32(2)
    end

    lo = Int32(UInt32(v & UInt64(0xffff)))
    hi = Int32(UInt32((v >> 16) & UInt64(0xffff)))
    if my_abs_i32(lo) <= UInt32(0xff) && my_abs_i32(hi) <= UInt32(0xff)
        return Int32(2)
    end

    b0 = v & UInt64(0xff)
    b1 = (v >> 8) & UInt64(0xff)
    b2 = (v >> 16) & UInt64(0xff)
    b3 = (v >> 24) & UInt64(0xff)
    if b0 == b1 && b0 == b2 && b0 == b3
        return Int32(1)
    end

    return Int32(4)
end

function fpc_kernel!(values, cmp_size, n::Int32)
    shared = CuStaticSharedArray(Int32, 256)
    tid = threadIdx().x
    lane = tid - Int32(1)
    gid = (blockIdx().x - Int32(1)) * blockDim().x + lane

    inc = Int32(0)
    if gid < n
        @inbounds inc = fpc_increment(values[gid + Int32(1)])
    end
    @inbounds shared[tid] = inc
    sync_threads()

    offset = blockDim().x ÷ Int32(2)
    while offset > 0
        if lane < offset
            @inbounds shared[tid] += shared[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end

    if lane == 0
        CUDA.@atomic cmp_size[1] += shared[1]
    end
    return
end

function convert_buffer_to_values(cbuffer::Vector{UInt8}, step::Int)
    n = length(cbuffer) ÷ step
    values = Vector{UInt64}(undef, n)
    @inbounds for i in 0:n-1
        v = UInt64(0)
        for j in 0:step-1
            v += UInt64(cbuffer[i * step + j + 1]) << (8 * j)
        end
        values[i + 1] = v
    end
    return values
end

function fpc_compress(values)
    total = Int32(0)
    @inbounds for v in values
        total += fpc_increment(v)
    end
    return total
end

function run_kernel!(d_values, d_cmp, values_size::Int, wgs::Int)
    fill!(d_cmp, Int32(0))
    @cuda threads=wgs blocks=(values_size ÷ wgs) fpc_kernel!(
        d_values, d_cmp, Int32(values_size))
    CUDA.synchronize()
    return Array(d_cmp)[1]
end

function run_fpc(wgs::Int, repeat::Int)
    step = 4
    size = wgs * wgs * wgs
    rng = MersenneTwister(2)
    cbuffer = Vector{UInt8}(undef, size * step)
    @inbounds for i in eachindex(cbuffer)
        shift = rand(rng, 0:255)
        cbuffer[i] = shift >= 8 ? UInt8(0) : UInt8((0xff << shift) & 0xff)
    end

    values = convert_buffer_to_values(cbuffer, step)
    values_size = length(values)
    cmp_size = fpc_compress(values)

    d_values = CuArray(values)
    d_cmp = CUDA.zeros(Int32, 1)
    ok = true

    run_kernel!(d_values, d_cmp, values_size, wgs)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        cmp_size_hw = run_kernel!(d_values, d_cmp, values_size, wgs)
        if cmp_size_hw != cmp_size
            @printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size)
            ok = false
            break
        end
    end
    elapsed_s = (time_ns() - start) * 1e-9 / repeat
    @printf("fpc: average device offload time %f (s)\n", elapsed_s)

    run_kernel!(d_values, d_cmp, values_size, wgs)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        cmp_size_hw = run_kernel!(d_values, d_cmp, values_size, wgs)
        if cmp_size_hw != cmp_size
            @printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size)
            ok = false
            break
        end
    end
    elapsed_s = (time_ns() - start) * 1e-9 / repeat
    @printf("fpc2: average device offload time %f (s)\n", elapsed_s)

    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <work-group size> <repeat>")
        return 1
    end
    return run_fpc(parse(Int, ARGS[1]), parse(Int, ARGS[2]))
end

exit(main())
