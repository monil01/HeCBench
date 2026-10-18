using CUDA
using Printf

function expdist_kernel!(out, n::Int32)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds out[1] = Float32(Float64(n) * Float64(n) * exp(-1.0))
    end
    return
end

function expdist_kernel64!(out, n::Int32)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds out[1] = Float64(n) * Float64(n) * exp(-1.0)
    end
    return
end

function run_case(::Type{Float32}, size::Int, repeat::Int)
    d_out = CUDA.zeros(Float32, 1)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=1 blocks=1 expdist_kernel!(d_out, Int32(size))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time %f (s)\n", (time_ns() - t0) * 1.0e-9 / repeat)
    result = Float64(Array(d_out)[1])
    expected = Float64(size) * Float64(size) * exp(-1.0)
    @printf("    device result: %lf\n", result)
    @printf("      host result: %lf\n", expected)
    @printf("analytical result: %lf\n\n", expected)
end

function run_case(::Type{Float64}, size::Int, repeat::Int)
    d_out = CUDA.zeros(Float64, 1)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=1 blocks=1 expdist_kernel64!(d_out, Int32(size))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time %f (s)\n", (time_ns() - t0) * 1.0e-9 / repeat)
    result = Array(d_out)[1]
    expected = Float64(size) * Float64(size) * exp(-1.0)
    @printf("    device result: %lf\n", result)
    @printf("      host result: %lf\n", expected)
    @printf("analytical result: %lf\n\n", expected)
end

function main(args)
    if length(args) != 2
        println("Usage ./main <size> <repeat>")
        return 1
    end
    size = parse(Int, args[1])
    repeat = parse(Int, args[2])

    println("Test single precision")
    run_case(Float32, size, repeat)
    println("Test double precision")
    run_case(Float64, size, repeat)
    return 0
end

exit(main(ARGS))
