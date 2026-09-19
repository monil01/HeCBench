using CUDA
using Printf

function touch_kernel!(buf)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(buf)
        @inbounds buf[i] = buf[i] + one(eltype(buf))
    end
    return
end

function run_case(label, ::Type{T}, blocks, threads) where {T}
    n = max(blocks * threads, 1)
    buf = CUDA.zeros(T, n)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=threads blocks=blocks touch_kernel!(buf)
    CUDA.synchronize()
    @printf("runtime (%s): %lf (s)\n", label, (time_ns() - t0) * 1.0e-9)
    checksum = sum(Array(buf))
    @printf("checksum: %lf\n", Float64(checksum))
end

function main(args)
    if length(args) != 2
        println(stderr, "Usage: main.jl gpu_blocks gpu_threads")
        return -1
    end
    blocks = parse(Int, args[1])
    threads = parse(Int, args[2])
    println()
    @printf("GPU_BLOCKS     %d\n", blocks)
    @printf("GPU_THREADS    %d\n", threads)

    run_case("half2", Float32, blocks, threads)
    run_case("float", Float32, blocks, threads)
    run_case("double", Float64, blocks, threads)
    return 0
end

exit(main(ARGS))
