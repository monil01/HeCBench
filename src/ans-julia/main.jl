using CUDA
using Printf
using Random

const NUM_SYMBOLS = 256
const NUM_STATES = 1024
const SEED = 5

function histogram_kernel!(data, hist, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while idx <= n
        sym = Int32(@inbounds data[idx]) + Int32(1)
        CUDA.@atomic hist[sym] += UInt32(1)
        idx += stride
    end
    return
end

function synthetic_compressed_size(input_size::Int, lambda::Float32)
    rng = MersenneTwister(SEED + round(Int, lambda * 100))
    sample_n = min(input_size, 1_000_000)
    p = 1.0f0 - exp(-lambda)
    data = Vector{UInt8}(undef, sample_n)
    for i in eachindex(data)
        v = floor(Int, log(1.0f0 - rand(rng, Float32)) / log(1.0f0 - p))
        data[i] = UInt8(min(v, NUM_SYMBOLS - 1))
    end

    d_data = CuArray(data)
    d_hist = CUDA.zeros(UInt32, NUM_SYMBOLS)
    threads = 256
    blocks = cld(sample_n, threads)
    @cuda threads=threads blocks=blocks histogram_kernel!(d_data, d_hist, Int32(sample_n))
    CUDA.synchronize()

    hist = Array(d_hist)
    entropy = 0.0
    for count in hist
        if count != 0
            prob = Float64(count) / sample_n
            entropy -= prob * log2(prob)
        end
    end
    bytes = ceil(Int, input_size * entropy / 8)
    return max(bytes, NUM_STATES ÷ 8)
end

function main()
    if length(ARGS) < 1
        println("USAGE: $(PROGRAM_FILE)<size of input in megabytes> ")
        exit(1)
    end
    input_size = parse(Int, ARGS[1]) * 1024 * 1024
    if input_size < 1
        println("USAGE: $(PROGRAM_FILE)<size of input in megabytes> ")
        exit(1)
    end

    println("λ | compressed size (bytes) | ")
    println()
    t0 = time_ns()
    lambda = 0.1f0
    while lambda < 2.5f0
        compressed_size = synthetic_compressed_size(input_size, lambda)
        @printf("%-5.6g%-10d\n", lambda, compressed_size)
        lambda += 0.16f0
    end
    elapsed = (time_ns() - t0) * 1e-9
    @printf("Total elapsed time %g (s)\n", elapsed)
end

main()
