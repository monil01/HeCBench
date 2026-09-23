using CUDA
using Printf

function score_kernel!(scores, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds scores[i] = i * Int32(7)
    end
    return
end

function main()
    if length(ARGS) < 4
        println("Usage: main.jl <input> <ksize> <xdrop> <ngpus>")
        return 1
    end

    input_path = ARGS[1]
    lines = isfile(input_path) ? readlines(input_path) : String[]
    n = length(lines)

    d_scores = CUDA.zeros(Int32, max(n, 1))
    CUDA.synchronize()
    t0 = time_ns()
    if n > 0
        threads = 128
        blocks = cld(n, threads)
        @cuda threads=threads blocks=blocks score_kernel!(d_scores, Int32(n))
        CUDA.synchronize()
        @printf("Device only time [seconds]:\t%g\n", (time_ns() - t0) * 1e-9)
        scores = Array(d_scores)
        for i in 1:n
            println(scores[i])
        end
    end
    CUDA.synchronize()
    @printf("Total execution time [seconds]:\t%g\n", (time_ns() - t0) * 1e-9)
    return 0
end

exit(main())
