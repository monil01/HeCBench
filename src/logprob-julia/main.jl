using CUDA
using Printf
using Random

function log_probs_kernel!(log_probs, logits, ids, lengths, max_input_length::Int32,
                           batch_size::Int32, vocab_size::Int32, vocab_size_padded::Int32)
    tid0 = threadIdx().x - Int32(1)
    step0 = blockIdx().x - Int32(1)
    bidx0 = blockIdx().y - Int32(1)
    if tid0 == Int32(0) && bidx0 < batch_size && step0 < lengths[bidx0 + Int32(1)] - Int32(1)
        step_offset = step0 * vocab_size_padded
        batch_offset = bidx0 * max_input_length * vocab_size_padded
        max_val = -typemax(Float32)
        for i0 in Int32(0):(vocab_size - Int32(1))
            @inbounds max_val = max(max_val, logits[batch_offset + step_offset + i0 + Int32(1)])
        end
        sum_exp = 0.0f0
        for i0 in Int32(0):(vocab_size - Int32(1))
            @inbounds sum_exp += exp(logits[batch_offset + step_offset + i0 + Int32(1)] - max_val)
        end
        idx = step0 + bidx0 * (max_input_length - Int32(1)) + Int32(1)
        token_idx = step0 + Int32(1) + bidx0 * max_input_length + Int32(1)
        @inbounds token = ids[token_idx]
        @inbounds log_probs[idx] = logits[batch_offset + step_offset + token + Int32(1)] -
                                   max_val - log(sum_exp + 1.0f-9)
    end
    return
end

function accumulate_kernel!(cum_log_probs, log_probs, lengths, max_input_length::Int32, batch_size::Int32)
    bidx0 = blockIdx().x - Int32(1)
    if threadIdx().x == Int32(1) && bidx0 < batch_size
        length = lengths[bidx0 + Int32(1)]
        acc = 0.0f0
        base = bidx0 * (max_input_length - Int32(1))
        for step0 in Int32(0):(length - Int32(2))
            @inbounds acc += log_probs[base + step0 + Int32(1)]
        end
        @inbounds cum_log_probs[bidx0 + Int32(1)] = acc
    end
    return
end

function log_probs_cpu(logits, ids, lengths, max_input_length, batch_size, vocab_size, vocab_size_padded)
    log_probs = Vector{Float32}(undef, batch_size * (max_input_length - 1))
    cum = Vector{Float32}(undef, batch_size)
    for b in 0:batch_size-1
        acc = 0.0f0
        for step in 0:lengths[b+1]-2
            offset = b * max_input_length * vocab_size_padded + step * vocab_size_padded
            max_val = maximum(@view logits[offset+1:offset+vocab_size])
            sum_exp = 0.0f0
            for i in 1:vocab_size
                sum_exp += exp(logits[offset+i] - max_val)
            end
            token_idx = step + 1 + b * max_input_length + 1
            idx = step + b * (max_input_length - 1) + 1
            log_probs[idx] = logits[offset + ids[token_idx] + 1] - max_val - log(sum_exp + 1.0f-9)
            acc += log_probs[idx]
        end
        cum[b+1] = acc
    end
    return log_probs, cum
end

function main()
    if length(ARGS) != 4
        println("Usage: main.jl <maximum sequence length> <batch size> <vocabulary size> <repeat>")
        return 1
    end
    max_length = parse(Int, ARGS[1])
    batch_size = parse(Int, ARGS[2])
    vocab_size = parse(Int, ARGS[3])
    repeat = parse(Int, ARGS[4])
    vocab_size_padded = cld(vocab_size, 32) * 32

    rng = MersenneTwister(123)
    logits = rand(rng, Float32, batch_size * max_length * vocab_size_padded) .* 12.0f0 .- 6.0f0
    lengths = fill(Int32(max_length), batch_size)
    ids = Int32.(rand(rng, 0:vocab_size-1, batch_size * max_length))

    d_logits = CuArray(logits)
    d_ids = CuArray(ids)
    d_lengths = CuArray(lengths)
    d_log_probs = CUDA.zeros(Float32, batch_size * (max_length - 1))
    d_cum = CUDA.zeros(Float32, batch_size)
    block_size = vocab_size < 1024 ? cld(vocab_size, 32) * 32 : 1024
    grid = (max_length - 1, batch_size)

    @cuda threads=block_size blocks=grid log_probs_kernel!(
        d_log_probs, d_logits, d_ids, d_lengths, Int32(max_length), Int32(batch_size),
        Int32(vocab_size), Int32(vocab_size_padded))
    @cuda threads=block_size blocks=batch_size accumulate_kernel!(
        d_cum, d_log_probs, d_lengths, Int32(max_length), Int32(batch_size))
    CUDA.synchronize()

    ref_log_probs, ref_cum = log_probs_cpu(logits, ids, lengths, max_length, batch_size, vocab_size, vocab_size_padded)
    got_log_probs = Array(d_log_probs)
    got_cum = Array(d_cum)
    error = false
    for i in eachindex(ref_log_probs)
        if abs(got_log_probs[i] - ref_log_probs[i]) > 1.0f-3
            @printf("log_probs: @%zu %f != %f\n", i - 1, got_log_probs[i], ref_log_probs[i])
            error = true
            break
        end
    end
    for i in eachindex(ref_cum)
        if abs(got_cum[i] - ref_cum[i]) > 1.0f-1
            @printf("cum_log_probs: @%d %f != %f\n", i - 1, got_cum[i], ref_cum[i])
            error = true
        end
    end
    println(error ? "FAIL" : "PASS")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block_size blocks=grid log_probs_kernel!(
            d_log_probs, d_logits, d_ids, d_lengths, Int32(max_length), Int32(batch_size),
            Int32(vocab_size), Int32(vocab_size_padded))
        @cuda threads=block_size blocks=batch_size accumulate_kernel!(
            d_cum, d_log_probs, d_lengths, Int32(max_length), Int32(batch_size))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of kernels: %f (us)\n", elapsed * 1e-3 / repeat)
    return 0
end

exit(main())
