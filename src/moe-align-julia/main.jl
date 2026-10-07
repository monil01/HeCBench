using CUDA
using Printf
using Random

const BLOCK_SIZE = 32

ceildiv(x::Integer, y::Integer) = (x + y - 1) ÷ y

function count_experts_kernel!(topk_ids, counts, numel::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= numel
        expert = topk_ids[i] + Int32(1)
        CUDA.atomic_add!(pointer(counts, expert), Int32(1))
        i += stride
    end
    return
end

function fill_outputs_kernel!(sorted_ids, expert_ids, total_tokens, cumsum, counts,
                              num_experts::Int32, block_size::Int32,
                              numel::Int32, max_num_tokens_padded::Int32,
                              max_num_m_blocks::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    while i <= max_num_tokens_padded
        sorted_ids[i] = numel
        i += stride
    end

    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    while j <= max_num_m_blocks
        expert_ids[j] = Int32(0)
        j += stride
    end

    if blockIdx().x == Int32(1) && threadIdx().x == Int32(1)
        total = cumsum[num_experts + Int32(1)]
        total_tokens[1] = total
        for e in Int32(0):(num_experts - Int32(1))
            first_block = cumsum[e + Int32(1)] ÷ block_size
            last_block = (cumsum[e + Int32(2)] - Int32(1)) ÷ block_size
            for b in first_block:last_block
                expert_ids[b + Int32(1)] = e
            end
            counts[e + Int32(1)] = Int32(0)
        end
    end
    return
end

function scatter_sorted_kernel!(topk_ids, sorted_ids, cumsum, offsets, numel::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= numel
        expert = topk_ids[i]
        pos = CUDA.atomic_add!(pointer(offsets, expert + Int32(1)), Int32(1))
        sorted_ids[cumsum[expert + Int32(1)] + pos + Int32(1)] = i - Int32(1)
        i += stride
    end
    return
end

function randperm_topk(topk::Int, m::Int, n::Int)
    rng = MersenneTwister(19937)
    out = Vector{Int32}(undef, m * topk)
    v = collect(Int32(0):Int32(n - 1))
    for row in 0:(m - 1)
        shuffle!(rng, v)
        for j in 1:topk
            out[row * topk + j] = v[j]
        end
    end
    return out
end

function moe_align_block_size!(sorted_ids, expert_ids, total_tokens, counts,
                               cumsum, offsets, d_topk_ids, h_cumsum,
                               num_experts::Int, block_size::Int,
                               topk_ids_size::Int, max_num_tokens_padded::Int)
    threads = 256
    blocks = min(ceildiv(topk_ids_size, threads), 65_535)
    CUDA.fill!(counts, Int32(0))
    @cuda threads=threads blocks=blocks count_experts_kernel!(d_topk_ids, counts, Int32(topk_ids_size))
    h_counts = Array(counts)

    padded_counts = Int32.(ceildiv.(Int.(h_counts), block_size) .* block_size)
    h_cumsum[1] = Int32(0)
    for i in 1:num_experts
        h_cumsum[i + 1] = h_cumsum[i] + padded_counts[i]
    end
    copyto!(cumsum, h_cumsum)

    max_num_m_blocks = length(expert_ids)
    fill_blocks = min(ceildiv(max(max_num_tokens_padded, max_num_m_blocks), threads), 65_535)
    @cuda threads=threads blocks=fill_blocks fill_outputs_kernel!(
        sorted_ids, expert_ids, total_tokens, cumsum, offsets, Int32(num_experts),
        Int32(block_size), Int32(topk_ids_size), Int32(max_num_tokens_padded),
        Int32(max_num_m_blocks))
    @cuda threads=threads blocks=blocks scatter_sorted_kernel!(
        d_topk_ids, sorted_ids, cumsum, offsets, Int32(topk_ids_size))
    CUDA.synchronize()

    return nothing
end

function verify_case(h_topk_ids, sorted_ids, expert_ids, total_tokens,
                     topk_ids_size::Int, num_experts::Int, block_size::Int,
                     max_num_tokens_padded::Int)
    actual_num_tokens = Int(Array(total_tokens)[1])
    actual_expert_ids = Array(expert_ids)
    actual_sorted_ids = Array(sorted_ids)

    ok = true
    if actual_num_tokens % block_size != 0
        println("Error: num_tokens_post_pad should be divisible by block_size")
        ok = false
    end
    if actual_num_tokens < topk_ids_size
        println("Error: num_tokens_post_pad should be at least total_tokens")
        ok = false
    end
    for eid in actual_expert_ids
        if eid < 0 || eid >= num_experts
            println("Error: expert_ids should contain valid expert indices")
            ok = false
            break
        end
    end

    ei = 1
    for t in 1:block_size:max_num_tokens_padded
        eid = actual_expert_ids[ei]
        ei += 1
        for b in 0:(block_size - 1)
            v = actual_sorted_ids[t + b]
            if v == topk_ids_size
                continue
            end
            if eid != h_topk_ids[Int(v) + 1]
                ok = false
                break
            end
        end
    end
    return ok
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])

    tokens = [1, 3, 256, 4096, 8192]
    experts = [32, 128]
    topks = [2, 3, 4]
    block_size = BLOCK_SIZE

    for m in tokens, num_experts in experts, topk in topks
        topk_ids_size = m * topk
        h_topk_ids = randperm_topk(topk, m, num_experts)
        d_topk_ids = CuArray(h_topk_ids)

        max_num_tokens_padded = topk_ids_size + num_experts * (block_size - 1)
        max_num_tokens_padded = ceildiv(max_num_tokens_padded, block_size) * block_size
        if topk_ids_size < num_experts
            max_num_tokens_padded = min(topk_ids_size * block_size, max_num_tokens_padded)
        end

        counts = CUDA.zeros(Int32, num_experts)
        cumsum = CUDA.zeros(Int32, num_experts + 1)
        offsets = CUDA.zeros(Int32, num_experts)
        sorted_ids = CUDA.zeros(Int32, max_num_tokens_padded)
        max_num_m_blocks = ceildiv(max_num_tokens_padded, block_size)
        expert_ids = CUDA.zeros(Int32, max_num_m_blocks)
        total_tokens = CUDA.zeros(Int32, 1)
        h_cumsum = Vector{Int32}(undef, num_experts + 1)

        moe_align_block_size!(sorted_ids, expert_ids, total_tokens, counts,
                              cumsum, offsets, d_topk_ids, h_cumsum,
                              num_experts, block_size, topk_ids_size,
                              max_num_tokens_padded)

        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            moe_align_block_size!(sorted_ids, expert_ids, total_tokens, counts,
                                  cumsum, offsets, d_topk_ids, h_cumsum,
                                  num_experts, block_size, topk_ids_size,
                                  max_num_tokens_padded)
        end
        CUDA.synchronize()
        elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
        @printf("Average execution time of the kernels (tokens %d, topk: %d, expert: %d, block_size %d): %f (us)\n",
                m, topk, num_experts, block_size, elapsed_us)

        ok = verify_case(h_topk_ids, sorted_ids, expert_ids, total_tokens,
                         topk_ids_size, num_experts, block_size,
                         max_num_tokens_padded)
        println(ok ? "PASS" : "FAIL")
    end
end

main()
