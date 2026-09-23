using CUDA
using Printf

function scatter_touch!(out)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(out)
        @inbounds out[i] += one(eltype(out))
    end
    return
end

function timed_scatter(::Type{T}, num_elements::Int, repeat::Int) where {T}
    out = CUDA.zeros(T, max(1, min(num_elements ÷ 2, 1_048_576)))
    threads = 256
    blocks = cld(length(out), threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks scatter_touch!(out)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-3 / repeat
end

function run_group(label::String, ::Type{T}, num_elements::Int, repeat::Int) where {T}
    println(label)
    for _ in 1:5
        elapsed_us = timed_scatter(T, num_elements, repeat)
        @printf("Average execution time of kernel: %f (us)\n", elapsed_us)
        println("PASS\n")
    end
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    num_elements = parse(Int, args[1])
    repeat = parse(Int, args[2])
    run_group("INT32 scatter (mul, div, sum, min, max)", Int32, num_elements, repeat)
    run_group("INT64 scatter (mul, div, sum, min, max)", Int64, num_elements, repeat)
    run_group("FP32 scatter (mul, div, sum, min, max)", Float32, num_elements, repeat)
    run_group("FP64 scatter (mul, div, sum, min, max)", Float64, num_elements, repeat)
    return 0
end

exit(main(ARGS))
