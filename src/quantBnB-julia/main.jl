using CUDA
using Printf
using Random

const BLOCK_SIZE = 256

function load_codebook()
    src = read(joinpath(@__DIR__, "..", "quantBnB-cuda", "code.h"), String)
    m = match(r"\{(.*)\}", src)
    m === nothing && error("could not parse quantization codebook")
    return Float32.(parse.(Float64, split(m.captures[1], ",")))
end

@inline function dquantize(code, x::Float32)
    pivot = 128
    upper_pivot = 256
    lower_pivot = 1
    lower = Float32(-1)
    upper = Float32(1)
    val = @inbounds code[pivot]

    step = 64
    while step > 0
        if x > val
            lower_pivot = pivot
            lower = val
            pivot += step
        else
            upper_pivot = pivot
            upper = val
            pivot -= step
        end
        val = @inbounds code[pivot]
        step >>>= 1
    end

    if upper_pivot == 256
        upper = @inbounds code[upper_pivot]
    end
    if lower_pivot == 1
        lower = @inbounds code[lower_pivot]
    end

    if x > val
        midpoint = (upper + val) * Float32(0.5)
        return UInt8(x > midpoint ? upper_pivot - 1 : pivot - 1)
    else
        midpoint = (lower + val) * Float32(0.5)
        return UInt8(x < midpoint ? lower_pivot - 1 : pivot - 1)
    end
end

function kquantize!(code, a, out, n::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = gridDim().x * blockDim().x
    while idx0 < n
        @inbounds out[idx0 + Int32(1)] = dquantize(code, a[idx0 + Int32(1)])
        idx0 += stride
    end
    return
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end

    n = parse(Int, args[1])
    repeat = parse(Int, args[2])
    code = load_codebook()

    rng = MersenneTwister(19937)
    a = Float32.(randn(rng, n))
    ref = Vector{UInt8}(undef, n)
    @inbounds for i in eachindex(a)
        ref[i] = dquantize(code, a[i])
    end

    d_code = CuArray(code)
    d_a = CuArray(a)
    d_out = CUDA.zeros(UInt8, n)
    blocks = cld(n, BLOCK_SIZE)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=blocks kquantize!(d_code, d_a, d_out, Int32(n))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1.0e-3 / repeat
    @printf("Average execution time of kQuantize kernel with block size %d: %f (us)\n",
            BLOCK_SIZE, elapsed_us)

    out = Array(d_out)
    println(out == ref ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
