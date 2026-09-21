using CUDA
using Printf
using Random

const GROUP_SIZE = 256

is_power_of_2(v::Integer) = v != 0 && (v & -v) == v

function round_to_power_of_2(v::Int)
    v -= 1
    shift = 1
    while shift < sizeof(Int) * 8
        v |= v >> shift
        shift <<= 1
    end
    return v + 1
end

function scan_kernel!(output, input, n::Int32)
    if blockIdx().x == Int32(1) && threadIdx().x == Int32(1)
        acc = 0.0f0
        for i in Int32(1):n
            @inbounds output[i] = acc
            @inbounds acc += input[i]
        end
    end
    return
end

function cpu_scan(input)
    out = similar(input)
    out[1] = 0.0f0
    for i in 2:length(input)
        out[i] = input[i - 1] + out[i - 1]
    end
    return out
end

function compare(ref, data, epsilon=0.001f0)
    err = 0.0f0
    refnorm = 0.0f0
    for i in 2:length(ref)
        diff = ref[i] - data[i]
        err += diff * diff
        refnorm += ref[i] * ref[i]
    end
    abs(refnorm) < epsilon && return false
    return sqrt(err) / sqrt(refnorm) < epsilon
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <repeat> <input length> <block size>")
        return 1
    end
    iterations = parse(Int, ARGS[1])
    nitems = parse(Int, ARGS[2])
    block_size = parse(Int, ARGS[3])
    if iterations < 1
        println("Error, iterations cannot be 0 or negative. Exiting..")
        return -1
    end
    if !is_power_of_2(nitems)
        nitems = round_to_power_of_2(nitems)
    end
    if (nitems ÷ block_size > GROUP_SIZE) && !is_power_of_2(nitems)
        println("Invalid length: $nitems")
        return -1
    end
    block_size = block_size < nitems ÷ 2 ? block_size : nitems ÷ 2
    pass = floor(Int, log(nitems) / log(block_size))
    if abs(log(nitems) / log(block_size) - pass) < 1e-7
        pass -= 1
    end
    temp_length = floor(Int, nitems / block_size^pass)

    rng = MersenneTwister(123)
    input = rand(rng, Float32, nitems) .* 256.0f0
    d_input = CuArray(input)
    d_output = CUDA.zeros(Float32, nitems)

    println("Executing kernel for $iterations iterations")
    println("-------------------------------------------")
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:iterations
        @cuda threads=1 blocks=1 scan_kernel!(d_output, d_input, Int32(nitems))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    println("Average execution time of scan kernels: $(elapsed * 1e-3 / iterations) (us)")

    output = Array(d_output)
    verification = cpu_scan(input)
    println(compare(output, verification) ? "PASS" : "FAIL")
    return 0
end

exit(main())
