using CUDA
using LinearAlgebra
using Printf

function matrix_a()
    return Float32[
         2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2;
        -2  4  0  0  0  0  0  0  0  0  0;
        -2  0  6  2  2  2  2  2  2  2  2;
        -2  0  2  8  4  4  4  4  4  4  4;
        -2  0  2  4 10 10 10 10 10 10 10;
        -2  0  2  4 10 12 12 12 12 12 12;
        -2  0  2  4 10 12 14 14 14 14 14;
        -2  0  2  4 10 12 14 16 16 16 16;
        -2  0  2  4 10 12 14 16 18 18 18;
        -2  0  2  4 10 12 14 16 18 20 20;
        -2  0  2  4 10 12 14 16 18 20 22
    ]
end

function determinant_root(a)
    d_a = CuArray(a)
    fact = cholesky!(Hermitian(d_a, :L))
    return prod(diag(Array(fact.L)))
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    host_a = matrix_a()
    det_root = determinant_root(host_a)
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        det_root = determinant_root(host_a)
    end
    CUDA.synchronize()
    @printf("Average execution time: %f (us)\n", (time_ns() - start) * 1.0e-3 / repeat)

    result = Float32(det_root * det_root)
    @printf("determinant = %f\n", result)
    println(abs(result - 2048.0f0) < 0.05f0 ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
