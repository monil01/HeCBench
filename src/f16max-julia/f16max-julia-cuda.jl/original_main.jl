using CUDA
using Printf

function hmax_kernel!(a, b, r, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds r[i] = max(a[i], b[i])
        i += stride
    end
    return
end

function run_case(label::String, n::Int, repeat::Int)
    a = CUDA.fill(Float16(1.0), n)
    b = CUDA.fill(Float16(2.0), n)
    r = CUDA.zeros(Float16, n)
    threads = 256
    blocks = min(cld(n, threads), 4096)

    @cuda threads=threads blocks=blocks hmax_kernel!(a, b, r, Int32(n))
    CUDA.synchronize()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks hmax_kernel!(a, b, r, Int32(n))
    end
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks hmax_kernel!(a, b, r, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time %f (us)\n", (time_ns() - start) * 1.0e-3 / repeat)

    ok = all(Array(r) .== Float16(2.0))
    println("$label $(ok ? "PASS" : "FAIL")")
    return ok
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    n_half2 = 1_048_576
    ok = run_case("fp16_hmax2", n_half2 * 2, repeat)
    ok &= run_case("fp16_hmax", n_half2 * 2, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
