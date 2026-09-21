using CUDA
using Printf

# Julia port of popcount-cuda.  The six kernels preserve the CUDA bit-count
# variants and each result is checked against a host reference.

const M1 = UInt64(0x5555555555555555)
const M2 = UInt64(0x3333333333333333)
const M4 = UInt64(0x0f0f0f0f0f0f0f0f)
const H01 = UInt64(0x0101010101010101)
const BLOCK_SIZE = 256

@inline function pc1_value(x::UInt64)
    x -= (x >> 1) & M1
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    x += x >> 8
    x += x >> 16
    x += x >> 32
    return Int32(x & UInt64(0x7f))
end

@inline function pc2_value(x::UInt64)
    x -= (x >> 1) & M1
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return Int32((x * H01) >> 56)
end

@inline function pc3_value(x::UInt64)
    count = Int32(0)
    while x != 0
        count += Int32(1)
        x &= x - UInt64(1)
    end
    return count
end

@inline function pc4_value(x::UInt64)
    count = Int32(0)
    i = Int32(0)
    while i < Int32(64)
        count += Int32(x & UInt64(1))
        x >>= 1
        i += Int32(1)
    end
    return count
end

@inline function pc5_value(x::UInt64)
    count = Int32(0)
    i = Int32(0)
    while i < Int32(8)
        count += Int32(count_ones(UInt8((x >> (UInt64(i) * UInt64(8))) & UInt64(0xff))))
        i += Int32(1)
    end
    return count
end

@inline pc6_value(x::UInt64) = Int32(count_ones(x))

function pc_kernel!(data, result, length::Int32, which::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length
        x = @inbounds data[i]
        value = which == Int32(1) ? pc1_value(x) :
                which == Int32(2) ? pc2_value(x) :
                which == Int32(3) ? pc3_value(x) :
                which == Int32(4) ? pc4_value(x) :
                which == Int32(5) ? pc5_value(x) :
                pc6_value(x)
        @inbounds result[i] = value
    end
    return
end

function make_input(length::Int)
    data = Vector{UInt64}(undef, length)
    state = UInt64(2)
    for i in 1:length
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        hi = state >> 32
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        lo = state >> 32
        data[i] = (hi << 32) | lo
    end
    return data
end

function check_results(data::Vector{UInt64}, result::Vector{Int32})
    for i in eachindex(data)
        if Int32(count_ones(data[i])) != result[i]
            println("Fail")
            return false
        end
    end
    println("Success")
    return true
end

function run_one!(which::Int, d_data, d_result, data, length::Int, repeat_n::Int)
    blocks = cld(length, BLOCK_SIZE)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=BLOCK_SIZE blocks=blocks pc_kernel!(
            d_data, d_result, Int32(length), Int32(which))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average kernel execution time (pc%d): %f (us)\n", which, elapsed_us)
    result = Array(d_result)
    return check_results(data, result)
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <length> <repeat>")
        return 1
    end

    length_n = parse(Int, ARGS[1])
    repeat_n = parse(Int, ARGS[2])
    data = make_input(length_n)
    d_data = CuArray(data)
    d_result = CUDA.zeros(Int32, length_n)

    ok = true
    for which in 1:6
        ok &= run_one!(which, d_data, d_result, data, length_n, repeat_n)
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main())
