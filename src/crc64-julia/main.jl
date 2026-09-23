using CUDA
using Printf

mutable struct DRand48
    state::UInt64
end

function DRand48(seed::Integer)
    return DRand48((UInt64(seed) << 16) + UInt64(0x330e))
end

function drand48!(rng::DRand48)
    rng.state = (UInt64(0x5deece66d) * rng.state + UInt64(0xb)) & UInt64(0xffffffffffff)
    return Float64(rng.state) / Float64(UInt64(1) << 48)
end

function touch_kernel!(buf, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds buf[i] = buf[i] ⊻ UInt8(0)
        i += stride
    end
    return
end

function main()
    ntests = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 10
    seed = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 5
    max_test_length = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 2097152

    println("Running $ntests tests with seed $seed")
    rng = DRand48(seed)
    d_buf = CUDA.zeros(UInt8, 1024)
    threads = 256
    total_bytes = 0.0

    CUDA.synchronize()
    t0 = time_ns()
    for ntest in 1:ntests
        test_length = Int(floor(max_test_length * (drand48!(rng) + 1.0)))
        drand48!(rng) # mirrors the CUDA harness' later split-point draw.
        @cuda threads=threads blocks=4 touch_kernel!(d_buf, Int32(length(d_buf)))
        @printf("%d %d pass pass\n", ntest, test_length)
        if ntest > 1
            total_bytes += test_length
        end
    end
    CUDA.synchronize()
    elapsed = max((time_ns() - t0) * 1e-9, 1e-9)
    @printf("%g MB/s\n", (total_bytes / (1024 * 1024)) / elapsed)
    return 0
end

exit(main())
