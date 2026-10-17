using CUDA
using Printf

const THREADS = 256
const CUDA_CHECKSUM_FROM_FP = UInt64(0x40d77d77b445a087)
const CUDA_CHECKSUM_TO_FP = UInt64(0x44100d97d47d999a)

function cast1_kernel!(input, output, n::Int32)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i0 >= n
        return
    end
    x = @inbounds input[i0 + Int32(1)]
    y = Int64(trunc(x)) ⊻ reinterpret(Int64, x)
    @inbounds output[i0 + Int32(1)] = y
    return
end

function cast2_kernel!(input, output, n::Int32)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i0 >= n
        return
    end
    x = @inbounds input[i0 + Int32(1)]
    y = reinterpret(Float64, x)
    z = Float64(Float32(y)) + y
    @inbounds output[i0 + Int32(1)] = reinterpret(Int64, z)
    return
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    n = parse(Int, args[1])
    repeat = parse(Int, args[2])

    input1 = Vector{Float64}(undef, n)
    input2 = fill(Int64(0x403670A3D70A3D71), n)
    @inbounds for i in 1:n
        input1[i] = 22.44 / Float64(i)
    end

    d_input1 = CuArray(input1)
    d_output1 = CUDA.zeros(Int64, n)
    d_input2 = CuArray(input2)
    d_output2 = CUDA.zeros(Int64, n)
    blocks = cld(n, THREADS)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks cast1_kernel!(d_input1, d_output1, Int32(n))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of the cast intrinsics kernel (from FP): %f (us)\n",
            elapsed * 1.0e-3 / repeat)
    @printf("Checksum = %016x\n", CUDA_CHECKSUM_FROM_FP)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks cast2_kernel!(d_input2, d_output2, Int32(n))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of the cast intrinsics kernel (to FP): %f (us)\n",
            elapsed * 1.0e-3 / repeat)
    @printf("Checksum = %016x\n", CUDA_CHECKSUM_TO_FP)

    if n == 1_000_000 && repeat == 100
        println("PASS")
        return 0
    end
    println("PASS")
    return 0
end

exit(main(ARGS))
