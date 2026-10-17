using CUDA
using Printf

function usage()
    println("Usage: ", PROGRAM_FILE, " <signal length> <repeat>")
    exit(1)
end

function round_to_power_of_2(v::UInt32)
    v -= UInt32(1)
    for i in 0:3
        v |= v >> (UInt32(1) << i)
    end
    return v + UInt32(1)
end

function get_levels(length::UInt32)
    for i in UInt32(0):UInt32(23)
        length == (UInt32(1) << i) && return Int(i)
    end
    return -1
end

function init_signal(n::Int)
    ccall(:srand, Cvoid, (Cuint,), Cuint(2))
    data = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        data[i] = Float32(mod(ccall(:rand, Cint, ()), 10))
    end
    return data
end

function haar_reference(input::Vector{Float32})
    n = length(input)
    temp = input ./ sqrt(Float32(n))
    out = zeros(Float32, n)
    len = n
    while len > 1
        @inbounds for i0 in 0:(len ÷ 2 - 1)
            data0 = temp[2 * i0 + 1]
            data1 = temp[2 * i0 + 2]
            out[i0 + 1] = (data0 + data1) / sqrt(2.0f0)
            out[len ÷ 2 + i0 + 1] = (data0 - data1) / sqrt(2.0f0)
        end
        copyto!(temp, out)
        len >>= 1
    end
    return out
end

function haar_level_kernel!(data, scratch, len::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    half = len ÷ Int32(2)
    if idx0 < half
        data0 = @inbounds data[Int32(2) * idx0 + Int32(1)]
        data1 = @inbounds data[Int32(2) * idx0 + Int32(2)]
        @inbounds scratch[idx0 + Int32(1)] = (data0 + data1) / sqrt(2.0f0)
        @inbounds scratch[half + idx0 + Int32(1)] = (data0 - data1) / sqrt(2.0f0)
    end
    return
end

function run_transform(input::Vector{Float32}, repeat::Int)
    n = length(input)
    d_data = CuArray(input ./ sqrt(Float32(n)))
    d_scratch = CUDA.zeros(Float32, n)
    threads = 256
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        copyto!(d_data, input ./ sqrt(Float32(n)))
        len = n
        while len > 1
            @cuda threads=threads blocks=cld(len ÷ 2, threads) haar_level_kernel!(d_data, d_scratch, Int32(len))
            copyto!(view(d_data, 1:len), view(d_scratch, 1:len))
            len >>= 1
        end
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    return Array(d_data), elapsed
end

function main()
    length(ARGS) == 2 || usage()
    signal_length = round_to_power_of_2(UInt32(parse(Int, ARGS[1])))
    repeat = parse(Int, ARGS[2])
    levels = get_levels(signal_length)
    if levels < 0
        println(stderr, "signalLength > 2 ^ 23 not supported")
        exit(1)
    end
    n = Int(signal_length)
    input = init_signal(n)

    println("Executing kernel for ", repeat, " iterations")
    println("-------------------------------------------")
    output_gpu, elapsed = run_transform(input, repeat)
    @printf("Average device offload time %.9f (s)\n", elapsed * 1.0e-9 / repeat)

    output_cpu = haar_reference(input)
    ok = all(abs.(output_gpu .- output_cpu) .<= 0.1f0)
    println(ok ? "PASS" : "FAIL")
end

main()
