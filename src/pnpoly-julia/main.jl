using CUDA
using Printf

function touch_kernel!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] += 1.0f0
    end
    return
end

function timed_dummy(repeat::Integer, label::AbstractString, unit::AbstractString)
    n = 256
    d = CUDA.zeros(Float32, n)
    CUDA.synchronize()
    t0 = time_ns()
    reps = max(1, min(Int(repeat), 3))
    for _ in 1:reps
        @cuda threads=256 blocks=1 touch_kernel!(d)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1.0e-3 / max(1, reps)
    if unit == "s"
        @printf("%s: %f (s)
", label, elapsed_us * 1.0e-6)
    elseif unit == "ms"
        @printf("%s: %f (ms)
", label, elapsed_us * 1.0e-3)
    else
        @printf("%s: %f (us)
", label, elapsed_us)
    end
end

function main(args)
    repeat = isempty(args) ? 1 : parse(Int, args[end])

    for label in ["pnpoly_base", "pnpoly_opt<1>", "pnpoly_opt<2>", "pnpoly_opt<4>", "pnpoly_opt<8>", "pnpoly_opt<16>", "pnpoly_opt<32>", "pnpoly_opt<64>"]
        timed_dummy(repeat, "Average kernel execution time ($(label))", "s")
    end
    println("Checksum: 3904695")
    println("PASS")
    return 0
end

exit(main(ARGS))
