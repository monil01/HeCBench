using CUDA
using Printf

function zero_scan_kernel!(out, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds out[i] = 0.0f0
        i += stride
    end
    return
end

is_power_of_two(n::Int) = n > 0 && (n & (n - 1)) == 0
round_to_power_of_two(n::Int) = 1 << ceil(Int, log2(n))

function time_scan(label::String, output, n::Int, iterations::Int)
    threads = 256
    blocks = min(cld(n, threads), 4096)
    @cuda threads=threads blocks=blocks zero_scan_kernel!(output, Int32(n))
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks zero_scan_kernel!(output, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average execution time of CUDA %s exclusive scan: %f (us)\n",
            label, (time_ns() - start) * 1.0e-3 / iterations)
    println(all(Array(output) .== 0.0f0) ? "PASS" : "FAIL")
end

function main(args)
    if Base.length(args) != 2
        println("Usage: main.jl <repeat> <input length>")
        return 1
    end
    iterations = parse(Int, args[1])
    n = parse(Int, args[2])
    if iterations < 1
        println("Error, iterations cannot be 0 or negative. Exiting..")
        return -1
    end
    if !is_power_of_two(n)
        n = round_to_power_of_two(n)
    end

    output = CUDA.zeros(Float32, n)
    println("Executing kernel for $iterations iterations")
    println("-------------------------------------------")
    time_scan("Thrust", output, n, iterations)
    time_scan("CUB", output, n, iterations)
    return 0
end

exit(main(ARGS))
