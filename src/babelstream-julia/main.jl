using CUDA
using Printf

const TBSIZE = 256
const DOT_NUM_BLOCKS = 256
const SCALAR = 0.4

function init_kernel!(a, b, c, initA, initB, initC)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds begin
        a[i] = initA
        b[i] = initB
        c[i] = initC
    end
    return
end

function copy_kernel!(a, c)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds c[i] = a[i]
    return
end

function mul_kernel!(b, c, scalar)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds b[i] = scalar * c[i]
    return
end

function add_kernel!(a, b, c)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds c[i] = a[i] + b[i]
    return
end

function triad_kernel!(a, b, c, scalar)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds a[i] = b[i] + scalar * c[i]
    return
end

function nstream_kernel!(a, b, c, scalar)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds a[i] += b[i] + scalar * c[i]
    return
end

function dot_kernel!(a::CuDeviceVector{T}, b::CuDeviceVector{T}, partial::CuDeviceVector{T}, array_size::Int32) where {T}
    shared = CUDA.@cuStaticSharedMem(T, TBSIZE)
    local_i = threadIdx().x
    stride = blockDim().x * gridDim().x
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    acc = zero(T)
    @inbounds while i <= array_size
        acc += a[i] * b[i]
        i += stride
    end
    shared[local_i] = acc

    offset = blockDim().x ÷ Int32(2)
    while offset > 0
        sync_threads()
        if local_i <= offset
            @inbounds shared[local_i] += shared[local_i + offset]
        end
        offset ÷= Int32(2)
    end

    if local_i == Int32(1)
        @inbounds partial[blockIdx().x] = shared[1]
    end
    return
end

function launch_init!(a, b, c, initA::T, initB::T, initC::T, array_size::Int) where {T}
    @cuda threads=TBSIZE blocks=(array_size ÷ TBSIZE) init_kernel!(a, b, c, initA, initB, initC)
    CUDA.synchronize()
end

function launch_copy!(a, c, array_size::Int)
    @cuda threads=TBSIZE blocks=(array_size ÷ TBSIZE) copy_kernel!(a, c)
    CUDA.synchronize()
end

function launch_mul!(b, c, scalar, array_size::Int)
    @cuda threads=TBSIZE blocks=(array_size ÷ TBSIZE) mul_kernel!(b, c, scalar)
    CUDA.synchronize()
end

function launch_add!(a, b, c, array_size::Int)
    @cuda threads=TBSIZE blocks=(array_size ÷ TBSIZE) add_kernel!(a, b, c)
    CUDA.synchronize()
end

function launch_triad!(a, b, c, scalar, array_size::Int)
    @cuda threads=TBSIZE blocks=(array_size ÷ TBSIZE) triad_kernel!(a, b, c, scalar)
    CUDA.synchronize()
end

function launch_nstream!(a, b, c, scalar, array_size::Int)
    @cuda threads=TBSIZE blocks=(array_size ÷ TBSIZE) nstream_kernel!(a, b, c, scalar)
    CUDA.synchronize()
end

function launch_dot!(a::CuArray{T}, b::CuArray{T}, partial::CuArray{T}, partial_host::Vector{T}, array_size::Int) where {T}
    @cuda threads=TBSIZE blocks=DOT_NUM_BLOCKS dot_kernel!(a, b, partial, Int32(array_size))
    CUDA.synchronize()
    copyto!(partial_host, partial)
    return sum(partial_host)
end

function parse_args()
    array_size = 33_554_432
    num_times = UInt32(100)
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--arraysize" || arg == "-s"
            i += 1
            i > length(ARGS) && error("Invalid array size.")
            array_size = parse(Int, ARGS[i])
            array_size <= 0 && error("Invalid array size.")
        elseif arg == "--numtimes" || arg == "-n"
            i += 1
            i > length(ARGS) && error("Invalid number of times.")
            num_times = parse(UInt32, ARGS[i])
            num_times < 2 && error("Number of times must be 2 or more")
        elseif arg == "--help" || arg == "-h"
            println()
            println("Usage: main.jl [OPTIONS]")
            println()
            println("Options:")
            println("  -h  --help               Print the message")
            println("  -s  --arraysize  SIZE    Use SIZE elements in the array")
            println("  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)")
            println()
            exit(0)
        else
            error("Unrecognized argument '$arg' (try '--help')")
        end
        i += 1
    end
    return array_size, Int(num_times)
end

function run_stream(::Type{T}, array_size::Int, num_times::Int) where {T}
    println("Running kernels $num_times times")
    if array_size % TBSIZE != 0
        error("Array size must be a multiple of $TBSIZE")
    end

    a = CuArray{T}(undef, array_size)
    b = CuArray{T}(undef, array_size)
    c = CuArray{T}(undef, array_size)
    partial = CuArray{T}(undef, DOT_NUM_BLOCKS)
    partial_host = Vector{T}(undef, DOT_NUM_BLOCKS)
    scalar = T(SCALAR)

    if sizeof(T) == sizeof(Float32)
        println("Precision: float")
    else
        println("Precision: double")
    end
    @printf("Array size: %.1f MB (=%.1f GB)\n", array_size * sizeof(T) * 1.0e-6, array_size * sizeof(T) * 1.0e-9)
    @printf("Total size: %.1f MB (=%.1f GB)\n", 3.0 * array_size * sizeof(T) * 1.0e-6, 3.0 * array_size * sizeof(T) * 1.0e-9)

    launch_init!(a, b, c, T(0.1), T(0.2), T(0.0), array_size)

    launch_copy!(a, c, array_size)
    launch_mul!(b, c, scalar, array_size)
    launch_add!(a, b, c, array_size)
    launch_triad!(a, b, c, scalar, array_size)
    sum_d = launch_dot!(a, b, partial, partial_host, array_size)
    launch_nstream!(a, b, c, scalar, array_size)

    expected_b = T(0.1) * scalar
    expected_c = T(0.1) + expected_b
    expected_a_before_nstream = expected_b + scalar * expected_c
    expected_sum = Float64(array_size) * Float64(expected_a_before_nstream) * Float64(expected_b)
    expected_a_after_nstream = expected_a_before_nstream + expected_b + scalar * expected_c

    ok = true
    if abs(expected_sum - Float64(sum_d)) >= 1
        println("dot: $expected_sum $sum_d")
        ok = false
    end
    first_a = Array(a[1:1])[1]
    if abs(Float64(first_a - expected_a_after_nstream)) > 1e-3
        println("a: $first_a $expected_a_after_nstream")
        ok = false
    end
    println(ok ? "PASS" : "FAIL")

    timings = [Float64[] for _ in 1:6]
    for _ in 1:num_times
        t = time()
        launch_copy!(a, c, array_size)
        push!(timings[1], time() - t)

        t = time()
        launch_mul!(b, c, scalar, array_size)
        push!(timings[2], time() - t)

        t = time()
        launch_add!(a, b, c, array_size)
        push!(timings[3], time() - t)

        t = time()
        launch_triad!(a, b, c, scalar, array_size)
        push!(timings[4], time() - t)

        t = time()
        launch_dot!(a, b, partial, partial_host, array_size)
        push!(timings[5], time() - t)

        t = time()
        launch_nstream!(a, b, c, scalar, array_size)
        push!(timings[6], time() - t)
    end

    println(rpad("Function", 12), rpad("MBytes/sec", 12), rpad("Min (sec)", 12), rpad("Max", 12), rpad("Average", 12))
    labels = ["Copy", "Mul", "Add", "Triad", "Dot", "Nstream"]
    sizes = [
        2 * sizeof(T) * array_size,
        2 * sizeof(T) * array_size,
        3 * sizeof(T) * array_size,
        3 * sizeof(T) * array_size,
        2 * sizeof(T) * array_size,
        4 * sizeof(T) * array_size,
    ]
    for i in eachindex(timings)
        measured = timings[i][2:end]
        min_t = minimum(measured)
        max_t = maximum(measured)
        avg_t = sum(measured) / length(measured)
        bandwidth = 1.0e-6 * sizes[i] / min_t
        @printf("%-12s%-12.3f%-12.5f%-12.5f%-12.5f\n", labels[i], bandwidth, min_t, max_t, avg_t)
    end
    println()
end

function main()
    array_size, num_times = parse_args()
    run_stream(Float32, array_size, num_times)
    run_stream(Float64, array_size, num_times)
end

main()
