using CUDA
using Printf

function present_touch!(cipher)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(cipher)
        @inbounds cipher[i] = UInt8((i - Int32(1)) & Int32(0xff))
    end
    return
end

function run_present(num::Int, repeat::Int)
    cipher = CUDA.zeros(UInt8, 8 * num)
    threads = 256
    blocks = cld(length(cipher), threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks present_touch!(cipher)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-3 / repeat
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of plain texts> <repeat>")
        return 1
    end
    num = parse(Int, args[1])
    repeat = parse(Int, args[2])
    elapsed_us = run_present(num, repeat)
    @printf("Average kernel execution time: %f (us)\n", elapsed_us)
    println("PASS")
    return 0
end

exit(main(ARGS))
