using CUDA
using Printf
using Random

const C10_WARP_SIZE = 32
const NUM_THREADS = C10_WARP_SIZE * 4
const THREAD_WORK_SIZE = 4
const BLOCK_WORK_SIZE = THREAD_WORK_SIZE * NUM_THREADS
const ROTARY_DIM = 128
const ROTARY_PAIRS = ROTARY_DIM ÷ 2

function rotary_kernel!(o1, o2, x1, x2, cosv, sinv, n::Int64)
    tid0 = threadIdx().x - Int32(1)
    block_start = Int64(blockIdx().x - Int32(1)) * Int64(BLOCK_WORK_SIZE)
    lane = Int64(tid0)
    @inbounds for work in Int64(0):(Int64(THREAD_WORK_SIZE) - Int64(1))
        idx0 = block_start + lane + work * Int64(NUM_THREADS)
        if idx0 < n
            idx = idx0 + Int64(1)
            xv1 = x1[idx]
            xv2 = x2[idx]
            c = cosv[idx]
            s = sinv[idx]
            o1[idx] = xv1 * c - xv2 * s
            o2[idx] = xv1 * s + xv2 * c
        end
    end
    return
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, ARGS[1])

    num_tokens = BLOCK_WORK_SIZE * 10000
    @printf("Number of tokens: %d\n", num_tokens)
    @printf("Rotary dimensions per token: %d\n", ROTARY_DIM)

    numel = num_tokens * ROTARY_PAIRS
    Random.seed!(12345)
    x1 = randn(Float32, numel)
    x2 = randn(Float32, numel)
    cosv = Vector{Float32}(undef, numel)
    sinv = Vector{Float32}(undef, numel)
    @inbounds for i0 in 0:(numel - 1)
        position = Float32(i0 ÷ ROTARY_PAIRS)
        pair_index = Float32(i0 % ROTARY_PAIRS)
        angle = position / (Float32(10000.0) ^ (Float32(2.0) * pair_index / Float32(ROTARY_DIM)))
        cosv[i0 + 1] = cos(angle)
        sinv[i0 + 1] = sin(angle)
    end

    d_x1 = CuArray(x1)
    d_x2 = CuArray(x2)
    d_cos = CuArray(cosv)
    d_sin = CuArray(sinv)
    d_o1 = CUDA.zeros(Float32, numel)
    d_o2 = CUDA.zeros(Float32, numel)

    blocks = cld(numel, BLOCK_WORK_SIZE)
    @printf("Number of thread blocks: %d, thread block size: %d\n", blocks, NUM_THREADS)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=NUM_THREADS blocks=blocks rotary_kernel!(d_o1, d_o2, d_x1, d_x2, d_cos, d_sin, Int64(numel))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1.0e-3 / repeat
    @printf("Average execution time: %f (us)\n", elapsed_us)

    o1 = Array(d_o1)
    o2 = Array(d_o2)
    ok = true
    @inbounds for i in eachindex(o1)
        r1 = x1[i] * cosv[i] - x2[i] * sinv[i]
        r2 = x1[i] * sinv[i] + x2[i] * cosv[i]
        if abs(r1 - o1[i]) > Float32(1.0e-3) || abs(r2 - o2[i]) > Float32(1.0e-3)
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
