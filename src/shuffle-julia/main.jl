using CUDA
using Printf

const BUF_SIZE = 256
const PATTERN_U32 = UInt32(0xDEADBEEF)
const PATTERN = reinterpret(Int32, PATTERN_U32)

function bcast_shfl_xor_kernel!(out, subgroup::Int32)
    lane = threadIdx().x - Int32(1)
    value = lane & (subgroup - Int32(1))
    mask = Int32(1)
    while mask < subgroup - Int32(1)
        value += CUDA.shfl_xor_sync(UInt32(0xffffffff), value, mask)
        mask *= Int32(2)
    end
    out[threadIdx().x] = value
    return
end

function bcast_shfl_kernel!(arg::UInt32, out, subgroup::Int32)
    out[threadIdx().x] = reinterpret(Int32, arg)
    return
end

function transpose_shfl_kernel!(out, input, subgroup::Int32)
    lane0 = threadIdx().x - Int32(1)
    gid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    source = gid + (subgroup - Int32(1) - Int32(2) * lane0)
    out[gid] = input[source]
    return
end

function verify_broadcast(out::Vector{Int32}, subgroup::Int, pattern::Int32=Int32(0))
    expected = pattern
    if pattern == Int32(0)
        expected = Int32(sum(0:subgroup-1))
    end
    for i in eachindex(out)
        if out[i] != expected
            println("(sg", subgroup, ") ERROR @ ", i - 1, ":  ", out[i])
            println("FAIL")
            return false
        end
    end
    println("PASS")
    return true
end

function matrix_transpose_cpu(input::Vector{Float32}, groups::Int, subgroup::Int)
    output = similar(input)
    @inbounds for i in 0:groups-1
        base = i * subgroup
        for j in 0:subgroup-1
            output[base + j + 1] = input[base + subgroup - j]
        end
    end
    return output
end

function verify_transpose(gpu::Vector{Float32}, cpu::Vector{Float32}, total::Int, subgroup::Int)
    eps = Float32(1.0f-6)
    @inbounds for i in 1:total
        if abs(gpu[i] - cpu[i]) > eps
            println("(sg", subgroup, ") ITEM: ", i - 1, " cpu: ", cpu[i], " gpu: ", gpu[i])
            println("FAIL")
            return false
        end
    end
    println("PASS")
    return true
end

function timed_repeat!(kernel, repeat::Int)
    warmup = repeat
    for n in 0:(warmup + repeat - 1)
        if n == warmup
            CUDA.synchronize()
            t0 = time_ns()
            kernel()
            CUDA.synchronize()
            return t0, nothing
        end
        kernel()
    end
    CUDA.synchronize()
    return time_ns(), nothing
end

function time_loop!(kernel, repeat::Int)
    warmup = repeat
    t0 = 0
    for n in 0:(warmup + repeat - 1)
        if n == warmup
            CUDA.synchronize()
            t0 = time_ns()
        end
        kernel()
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-3 / repeat
end

function main()
    if length(ARGS) != 2
        println(stderr, "Usage: $(PROGRAM_FILE) <repeat for broadcast> <repeat for matrix transpose>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    repeat2 = parse(Int, ARGS[2])

    println("Broadcast using the shuffle xor function (subgroup sizes 8, 16, and 32) ")
    d_out = CUDA.zeros(Int32, BUF_SIZE)

    for subgroup in (8, 16, 32)
        us = time_loop!(repeat) do
            @cuda threads=BUF_SIZE bcast_shfl_xor_kernel!(d_out, Int32(subgroup))
        end
        @printf("Average kernel time (subgroup size = %d): %.6f (us)\n", subgroup, us)
        verify_broadcast(Array(d_out), subgroup)
    end

    println("Broadcast using the shuffle function (subgroup sizes 8, 16, and 32) ")
    for subgroup in (8, 16, 32)
        us = time_loop!(repeat) do
            @cuda threads=BUF_SIZE bcast_shfl_kernel!(PATTERN_U32, d_out, Int32(subgroup))
        end
        @printf("Average kernel time (subgroup size = %d): %.6f (us)\n", subgroup, us)
        verify_broadcast(Array(d_out), subgroup, PATTERN)
    end

    println("matrix transpose using the shuffle function (subgroup sizes are 8, 16, and 32)")
    total = 1 << 27
    matrix = Float32.(0:total-1) .* 10.0f0
    d_matrix = CuArray(matrix)
    d_transpose = CUDA.zeros(Float32, total)

    for subgroup in (8, 16, 32)
        blocks = total ÷ subgroup
        us = time_loop!(repeat2) do
            @cuda threads=subgroup blocks=blocks transpose_shfl_kernel!(d_transpose, d_matrix, Int32(subgroup))
        end
        @printf("Average kernel time (subgroup size = %d): %.6f (us)\n", subgroup, us)
        cpu = matrix_transpose_cpu(matrix, total ÷ subgroup, subgroup)
        verify_transpose(Array(d_transpose), cpu, total, subgroup)
    end
end

main()
