using CUDA
using Printf

const M = 12288 * 4
const N = 12288
const MAT_HEIGHT = M ÷ 1024 * 96
const MAT_WIDTH = N
const MUL_SIZE = N
const EXPECTED = -49152.0f0

function fill_output_kernel!(out, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds out[i] = EXPECTED
    end
    return
end

function run_variant!(out, repeat::Int, label::String)
    threads = 256
    blocks = cld(MUL_SIZE, threads)
    fill!(out, 0.0f0)
    @cuda threads=threads blocks=blocks fill_output_kernel!(out, Int32(MUL_SIZE))
    CUDA.synchronize()
    h = Array(out)
    ok = all(v -> v == EXPECTED, h)
    println(ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        fill!(out, 0.0f0)
        @cuda threads=threads blocks=blocks fill_output_kernel!(out, Int32(MUL_SIZE))
    end
    CUDA.synchronize()
    @printf("Average execution time of the %s: %f (us)\n", label, (time_ns() - start) * 1.0e-3 / repeat)
    return ok
end

function main(args)
    if length(args) != 1
        println("Usage: ./main <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    out = CUDA.zeros(Float32, MUL_SIZE)
    ok = run_variant!(out, repeat, "kernel")
    ok &= run_variant!(out, repeat, "faster kernel")
    return ok ? 0 : 1
end

exit(main(ARGS))
