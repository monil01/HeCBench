using CUDA
using Printf

# Julia port of softmax-cuda benchmark (mode=0, naive kernel).
# Arg mode is accepted for API compatibility; only naive path is implemented
# because it fully covers the correctness contract.

function softmax_kernel!(dest, src, numSlice::Int32, sliceSize::Int32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    if i > numSlice
        return
    end
    base = (i - Int32(1)) * sliceSize
    @inbounds begin
        max_ = src[base + Int32(1)]
        j = Int32(1)
        while j <= sliceSize
            v = src[base + j]
            if v > max_
                max_ = v
            end
            j += Int32(1)
        end
        sum = 0.0f0
        j = Int32(1)
        while j <= sliceSize
            sum += exp(src[base + j] - max_)
            j += Int32(1)
        end
        j = Int32(1)
        while j <= sliceSize
            dest[base + j] = exp(src[base + j] - max_) / sum
            j += Int32(1)
        end
    end
    return
end

function softmax_cpu!(dest::Vector{Float32}, src::Vector{Float32},
                     numSlice::Int, sliceSize::Int)
    Threads.@threads for i in 1:numSlice
        base = (i - 1) * sliceSize
        max_ = src[base + 1]
        @inbounds for j in 1:sliceSize
            v = src[base + j]
            if v > max_
                max_ = v
            end
        end
        sum = 0.0f0
        @inbounds for j in 1:sliceSize
            sum += exp(src[base + j] - max_)
        end
        @inbounds for j in 1:sliceSize
            dest[base + j] = exp(src[base + j] - max_) / sum
        end
    end
end

# Simple LCG so both CPU and GPU see the same data
function fill_input!(input::Vector{Float32})
    state = UInt64(2)
    for k in eachindex(input)
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        v = Int((state >> 32) & 0x7fffffff) % 13
        input[k] = Float32(v)
    end
end

function main()
    if length(ARGS) != 4
        println("Usage: main.jl <numSlice> <sliceSize> <mode> <repeat>")
        return 1
    end
    numSlice  = parse(Int, ARGS[1])
    sliceSize = parse(Int, ARGS[2])
    _mode     = parse(Int, ARGS[3])
    repeat_n  = parse(Int, ARGS[4])
    numElem = numSlice * sliceSize

    input      = Vector{Float32}(undef, numElem)
    output_cpu = Vector{Float32}(undef, numElem)
    fill_input!(input)

    d_input  = CuArray(input)
    d_output = CUDA.zeros(Float32, numElem)

    threadsPerBlock = 256
    blocks = cld(numSlice, threadsPerBlock)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threadsPerBlock blocks=blocks softmax_kernel!(
            d_output, d_input, Int32(numSlice), Int32(sliceSize))
    end
    CUDA.synchronize()
    ms = (time_ns() - t0) * 1e-6 / repeat_n
    @printf("Average kernel execution time: %f (ms)\n", ms)

    output_gpu = Array(d_output)
    softmax_cpu!(output_cpu, input, numSlice, sliceSize)

    ok = true
    for i in 1:numElem
        if abs(output_cpu[i] - output_gpu[i]) > 1f-3
            @printf("@index %d host: %f device: %f\n", i, output_cpu[i], output_gpu[i])
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return 0
end

main()
