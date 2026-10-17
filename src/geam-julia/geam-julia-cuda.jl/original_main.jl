using CUDA
using Printf
using Random

function transpose_type(::Type{T}, label::String, nrow::Int, ncol::Int, repeat::Int) where {T}
    println("----------------$label transpose matrix ($nrow x $ncol)----------------")
    rng = MersenneTwister(123)
    matrix = T.(rand(rng, 0:12, nrow * ncol))
    start = time_ns()
    ref = collect(reshape(permutedims(reshape(matrix, ncol, nrow)), :))
    @printf("Host: serial matrix transpose time = %f (ms)\n", (time_ns() - start) * 1.0e-6)

    d_matrix = reshape(CuArray(matrix), ncol, nrow)
    d_out = CUDA.zeros(T, nrow, ncol)
    work = CUDA.zeros(T, nrow, ncol)
    alpha = one(T)
    beta = zero(T)
    for _ in 1:4
        CUDA.CUBLAS.geam!('T', 'N', alpha, d_matrix, beta, work, d_out)
    end
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        CUDA.CUBLAS.geam!('T', 'N', alpha, d_matrix, beta, work, d_out)
    end
    CUDA.synchronize()
    @printf("Device: average matrix transpose time = %f (ms)\n", (time_ns() - start) * 1.0e-6 / repeat)
    ok = collect(vec(Array(d_out))) == ref
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 3
        println("Usage main.jl <matrix row> <matrix col> <repeat>")
        return 1
    end
    nrow = parse(Int, args[1])
    ncol = parse(Int, args[2])
    repeat = parse(Int, args[3])
    ok = transpose_type(Float32, "FP32", nrow, ncol, repeat)
    ok &= transpose_type(Float64, "FP64", nrow, ncol, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
