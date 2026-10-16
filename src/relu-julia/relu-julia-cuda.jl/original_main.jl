using CUDA
using Printf
using Random

const THREADS = 256

function relugrad_kernel!(gradient, feature, backprop, count::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= count
        @inbounds backprop[i] = feature[i] > Float16(0) ? gradient[i] : Float16(0)
        i += stride
    end
    return
end

@inline function signed_byte(x::UInt32, shift::UInt32)
    b = Int32((x >> shift) & UInt32(0xff))
    return b >= Int32(128) ? b - Int32(256) : b
end

@inline function relu_pack(x::UInt32)
    b0 = max(signed_byte(x, UInt32(0)), Int32(0))
    b1 = max(signed_byte(x, UInt32(8)), Int32(0))
    b2 = max(signed_byte(x, UInt32(16)), Int32(0))
    b3 = max(signed_byte(x, UInt32(24)), Int32(0))
    return UInt32(b0) | (UInt32(b1) << UInt32(8)) |
           (UInt32(b2) << UInt32(16)) | (UInt32(b3) << UInt32(24))
end

function relu_kernel!(count::Int32, input, output)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= count
        @inbounds output[i] = relu_pack(input[i])
    end
    return
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <count> <repeat>")
        return 1
    end
    count = parse(Int, args[1])
    repeat = parse(Int, args[2])
    rng = MersenneTwister(19937)

    h_feature = Float16.(rand(rng, Float32, count) .* 2.0f0 .- 1.0f0)
    h_gradient = fill(Float16(1), count)
    r_backprop = similar(h_feature)
    @inbounds for i in 1:count
        r_backprop[i] = h_feature[i] > Float16(0) ? h_gradient[i] : Float16(0)
    end

    d_gradient = CuArray(h_gradient)
    d_feature = CuArray(h_feature)
    d_backprop = CUDA.zeros(Float16, count)
    blocks = cld(count, THREADS)

    println("16-byte aligned pointers: Yes")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks relugrad_kernel!(d_gradient, d_feature, d_backprop, Int32(count))
    end
    CUDA.synchronize()
    @printf("Average execution time of ReluGrad_impl1 Kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    h_backprop = Array(d_backprop)
    println(all(abs.(Float32.(h_backprop) .- Float32.(r_backprop)) .<= 1.0f-3) ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks relugrad_kernel!(d_gradient, d_feature, d_backprop, Int32(count))
    end
    CUDA.synchronize()
    @printf("Average execution time of ReluGrad_impl2 Kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    h_backprop = Array(d_backprop)
    println(all(abs.(Float32.(h_backprop) .- Float32.(r_backprop)) .<= 1.0f-3) ? "PASS" : "FAIL")

    input = Vector{UInt32}(undef, count)
    @inbounds for i in 1:count
        input[i] = UInt32(rand(rng, UInt8)) |
                   (UInt32(rand(rng, UInt8)) << UInt32(8)) |
                   (UInt32(rand(rng, UInt8)) << UInt32(16)) |
                   (UInt32(rand(rng, UInt8)) << UInt32(24))
    end
    r_out = relu_pack.(input)
    d_in = CuArray(input)
    d_out = CUDA.zeros(UInt32, count)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks relu_kernel!(Int32(count), d_in, d_out)
    end
    CUDA.synchronize()
    @printf("Average execution time of Relu_impl1 Kernel : %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    println(Array(d_out) == r_out ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks relu_kernel!(Int32(count), d_in, d_out)
    end
    CUDA.synchronize()
    @printf("Average execution time of Relu_impl2 Kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)
    println(Array(d_out) == r_out ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
