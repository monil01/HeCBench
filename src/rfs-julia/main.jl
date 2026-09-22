using CUDA
using Printf

function truncate_factor(factor::Float32, x::Float32)
    return (factor + x) - factor
end

function init_arrays!(x, total::Int64)
    idx0 = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx0
    while i < total
        raw = Float32((i * Int64(1103515245) + Int64(12345)) & Int64(0x00ffffff)) / 16777216.0f0
        sign = (i & Int64(1)) == 0 ? 1.0f0 : -1.0f0
        x[i + 1] = sign * raw
        i += stride
    end
    return
end

function sum_array!(factor::Float32, len::Int32, x, result)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while i < len
        q = truncate_factor(factor, x[i + Int32(1)])
        CUDA.@atomic result[1] += q
        i += stride
    end
    return
end

function sum_arrays!(narrays::Int32, len::Int32, x, result, factor::Float32)
    arr = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while arr < narrays
        base = Int64(arr) * Int64(len)
        s = 0.0f0
        i = len - Int32(1)
        while i >= 0
            s += truncate_factor(factor, x[base + Int64(i) + 1])
            i -= Int32(1)
        end
        result[arr + Int32(1)] = s
        arr += stride
    end
    return
end

function compare_kernel!(a, b, flag, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        if a[i] != b[i]
            flag[1] = Int32(1)
        end
    end
    return
end

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function create_rounding_factor(maxval::Float32, n::Int)
    delta = (maxval * Float32(n)) / (1.0f0 - 2.0f0 * Float32(n) * eps(Float32))
    _, exp = frexp(delta)
    return ldexp(1.0f0, exp)
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of arrays> <length of each array>")
        return 1
    end
    narrays = parse(Int32, args[1])
    len = parse(Int32, args[2])
    total = Int64(narrays) * Int64(len)
    factor = create_rounding_factor(1.0f0, Int(len))

    arrays = CuArray{Float32}(undef, total)
    init_blocks = min(ceil_div(total, Int64(256)), Int64(65535))
    @cuda threads=256 blocks=Int(init_blocks) init_arrays!(arrays, total)
    CUDA.synchronize()

    result_atomic = CuArray{Float32}(undef, Int(narrays))
    result_reverse = CuArray{Float32}(undef, Int(narrays))

    fill!(result_atomic, 0.0f0)
    CUDA.synchronize()
    start = time_ns()
    for arr in 0:(Int(narrays) - 1)
        arr_view = @view arrays[(Int64(arr) * Int64(len) + 1):(Int64(arr + 1) * Int64(len))]
        result_view = @view result_atomic[arr + 1:arr + 1]
        @cuda threads=256 blocks=256 sum_array!(factor, len, arr_view, result_view)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1.0e-9
    @printf("Average kernel execution time (sumArray): %f (s)\n", elapsed_s / Int(narrays))

    start = time_ns()
    @cuda threads=256 blocks=256 sum_arrays!(narrays, len, arrays, result_reverse, factor)
    CUDA.synchronize()
    elapsed_s2 = (time_ns() - start) * 1.0e-9
    @printf("Kernel execution time (sumArrays): %f (s)\n", elapsed_s2)

    flag = CuArray([Int32(0)])
    @cuda threads=256 blocks=ceil_div(Int(narrays), 256) compare_kernel!(result_atomic, result_reverse, flag, narrays)
    CUDA.synchronize()
    ok = CUDA.@allowscalar flag[1] == Int32(0)
    println(ok ? "PASS" : "FAIL")
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
