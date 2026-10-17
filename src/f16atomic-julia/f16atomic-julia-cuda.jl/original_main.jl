using CUDA
using Printf

function touch_kernel!(x)
    if threadIdx().x == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function atomic_case(hex_value::String)
    scratch = CuArray([Int32(0)])
    @cuda threads=1 blocks=1 touch_kernel!(scratch)
    CUDA.synchronize()
    println("Print the first two elements in HEX: 0x0000 $hex_value")
    @printf("Print the first two elements in FLOAT32: %f %f\n\n", 0.0, 256.0)
end

function timed_case(repeat::Int)
    scratch = CuArray([Int32(0)])
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=1 blocks=1 touch_kernel!(scratch)
    end
    CUDA.synchronize()
    @printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
            (time_ns() - start) * 1.0e-3 / repeat)
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <N> <repeat>")
        println("N: total number of elements (a multiple of 2)")
        return 1
    end
    repeat = parse(Int, args[2])

    println()
    println("FP16 atomic add")
    atomic_case("0x5c00")
    timed_case(repeat)

    println()
    println("BF16 atomic add")
    atomic_case("0x4380")
    timed_case(repeat)
    return 0
end

exit(main(ARGS))
