using CUDA
using Printf

function parse_args()
    if length(ARGS) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        exit(1)
    end
    return parse(Int, ARGS[1]), parse(Int, ARGS[2])
end

function main()
    n, repeat = parse_args()
    matched = false

    println("Thrust version: CUDA.jl accumulate")
    CUDA.synchronize()
    start = time_ns()

    for _ in 1:repeat
        d_vec = CuArray(Int32.(0:(n - 1)))
        d_res = accumulate(+, d_vec)
        reduced = CUDA.sum(d_vec)
        last_scan = CUDA.@allowscalar d_res[end]
        wrapped_reduced = reinterpret(Int32, UInt32(mod(reduced, 1 << 32)))
        matched = wrapped_reduced == last_scan
    end

    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time: %f (us)\n", elapsed_us)
    println(matched ? "PASS" : "FAIL")
end

main()
