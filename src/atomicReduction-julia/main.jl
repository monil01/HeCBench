using CUDA
using Printf
using Random

function atomic_reduction!(input, output, n::Int32, width::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x * width
    start = tid0 * width
    local_sum = Int32(0)
    i = start
    while i < n
        for lane in Int32(0):(width - Int32(1))
            idx = i + lane
            if idx < n
                @inbounds local_sum += input[idx + Int32(1)]
            end
        end
        i += stride
    end
    CUDA.@atomic output[1] += local_sum
    return
end

function run_case(label_width::Int, input, checksum::Int32, array_length::Int,
                  repeat::Int, block_size::Int, blocks::Int, total_gb::Float64)
    out = CuArray([Int32(0)])
    width = Int32(label_width)

    fill!(out, Int32(0))
    @cuda threads=block_size blocks=max(blocks, 1) atomic_reduction!(input, out, Int32(array_length), width)
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        fill!(out, Int32(0))
        @cuda threads=block_size blocks=max(blocks, 1) atomic_reduction!(input, out, Int32(array_length), width)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1.0e-9
    perf = 1.0e-09 * total_gb / elapsed_s

    println("Thread block size: $block_size, The average performance of reduction is $perf GBytes/sec")
    sum_value = Array(out)[1]
    println(sum_value == checksum ? "VERIFICATION: PASS\n" : "VERIFICATION: FAIL!!\n")
end

function main(args)
    array_length = 52_428_800
    repeat = 100
    if length(args) == 2
        array_length = parse(Int, args[1])
        repeat = parse(Int, args[2])
    end

    println("Array size: $(array_length * sizeof(Int32) / 1024.0 / 1024.0) MB")
    println("Repeat the kernel execution: $repeat times")

    rng = MersenneTwister(123)
    host = Int32.(rand(rng, 0:1, array_length))
    checksum = Int32(sum(host))
    println("Device name: $(CUDA.name(CUDA.device()))")

    input = CuArray(host)
    total_gb = Float64(array_length * sizeof(Int32) * repeat)
    block_sizes = (128, 256, 512, 1024)
    widths = (1, 2, 4, 8, 16)

    for block_size in block_sizes
        blocks = min(cld(array_length, block_size), 2048)
        for width in widths
            run_case(width, input, checksum, array_length, repeat, block_size,
                     max(blocks ÷ width, 1), total_gb)
        end
    end
    return 0
end

exit(main(ARGS))
