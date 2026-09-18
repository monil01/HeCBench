using CUDA
using Printf
using Random

function reverse_kernel!(d, len::Int32)
    shared = CuStaticSharedArray(Int32, 256)
    t = threadIdx().x - Int32(1)
    @inbounds shared[t + Int32(1)] = d[t + Int32(1)]
    sync_threads()
    @inbounds d[t + Int32(1)] = shared[len - t]
    return
end

function run_reverse(iteration::Int)
    len = 256
    gold_even = Int32.(collect(0:len-1))
    gold_odd = Int32.(collect(len-1:-1:0))
    test = Vector{Int32}(undef, len)
    d_test = CuArray(gold_even)

    rng = MersenneTwister(123)
    total_ns = 0
    error = false

    for _ in 1:iteration
        count = rand(rng, 100:9999)
        copyto!(d_test, gold_even)

        CUDA.synchronize()
        start = time_ns()
        for _ in 1:count
            @cuda threads=len blocks=1 reverse_kernel!(d_test, Int32(len))
        end
        CUDA.synchronize()
        total_ns += time_ns() - start

        copyto!(test, d_test)
        expected = iseven(count) ? gold_even : gold_odd
        if test != expected
            error = true
            break
        end
    end

    @printf("Total kernel execution time: %f (s)\n", total_ns * 1e-9)
    println(error ? "FAIL" : "PASS")
    return error ? 1 : 0
end

function main()
    if length(ARGS) != 1
        println("Usage: ./main.jl <iterations>")
        return 1
    end

    iteration = parse(Int, ARGS[1])
    return run_reverse(iteration)
end

exit(main())
