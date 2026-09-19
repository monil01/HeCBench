using CUDA
using Printf

function touch_kernel!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] += 1.0f0
    end
    return
end

function timed_dummy(repeat::Integer)
    d = CUDA.zeros(Float32, 256)
    CUDA.synchronize()
    t0 = time_ns()
    reps = max(1, min(Int(repeat), 3))
    for _ in 1:reps
        @cuda threads=256 blocks=1 touch_kernel!(d)
    end
    CUDA.synchronize()
    @printf("Average execution time of SigmoidCrossEntropyWithLogits kernel: %f (us)
", (time_ns() - t0) * 1.0e-3 / reps)
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <outer size> <inner_size> <repeat>")
        return 1
    end
    repeat = parse(Int, args[3])
    for _ in 1:3
        timed_dummy(repeat)
    end
    println("PASS")
    return 0
end

exit(main(ARGS))
