using CUDA
using Printf
using Random

function mha_kernel!(q, k, v, beam_size::Int32, n_steps::Int32, qk_col::Int32,
                     v_col::Int32, nhead::Int32, scale::Float32,
                     threshold::Int32, dst)
    dim_per_head = qk_col ÷ nhead
    block0 = blockIdx().x - Int32(1)
    tid0 = threadIdx().x - Int32(1)
    candidate_id = block0 ÷ nhead
    head_id = block0 % nhead

    buffer = @cuDynamicSharedMem(Float32, Int(dim_per_head + n_steps))
    sq = buffer
    logits_offset = Int(dim_per_head)

    if tid0 < dim_per_head
        pos = candidate_id * qk_col + head_id * dim_per_head + tid0
        @inbounds sq[Int(tid0) + 1] = q[Int(pos) + 1]
    end
    sync_threads()

    if tid0 < n_steps
        summ = Float32(0)
        k_base = candidate_id * qk_col * n_steps + head_id * dim_per_head + tid0 * qk_col
        for i0 in Int32(0):(dim_per_head - Int32(1))
            @inbounds summ += sq[Int(i0) + 1] * k[Int(k_base + i0) + 1]
        end
        @inbounds buffer[logits_offset + Int(tid0) + 1] = summ * scale
    end
    sync_threads()

    if tid0 == 0
        max_val = Float32(-1.0f-20)
        for t0 in Int32(0):(n_steps - Int32(1))
            @inbounds max_val = max(max_val, buffer[logits_offset + Int(t0) + 1])
        end

        denom = Float32(0)
        for t0 in Int32(0):(n_steps - Int32(1))
            @inbounds local_i = buffer[logits_offset + Int(t0) + 1] - max_val
            if local_i < -Float32(threshold)
                local_i = -Float32(threshold)
            end
            local_o = exp(local_i)
            @inbounds buffer[logits_offset + Int(t0) + 1] = local_o
            denom += local_o
        end

        for t0 in Int32(0):(n_steps - Int32(1))
            @inbounds buffer[logits_offset + Int(t0) + 1] /= denom
        end
    end
    sync_threads()

    if tid0 < dim_per_head
        summ = Float32(0)
        tid = candidate_id * v_col * n_steps + head_id * dim_per_head + tid0
        for i0 in Int32(0):(n_steps - Int32(1))
            @inbounds summ += buffer[logits_offset + Int(i0) + 1] * v[Int(tid + i0 * v_col) + 1]
        end
        out = candidate_id * v_col + head_id * dim_per_head + tid0
        @inbounds dst[Int(out) + 1] = summ
    end
    return
end

function mha_reference(q, k, v, beam_size, n_steps, qk_col, v_col, nhead, scale, threshold)
    dim_per_head = qk_col ÷ nhead
    dst = Vector{Float32}(undef, beam_size * v_col)
    sq = Vector{Float32}(undef, dim_per_head)
    logits = Vector{Float32}(undef, n_steps)
    dp = Vector{Float32}(undef, n_steps)

    for candidate_id in 0:(beam_size - 1), head_id in 0:(nhead - 1)
        for t in 0:(dim_per_head - 1)
            pos = candidate_id * qk_col + head_id * dim_per_head + t
            sq[t + 1] = q[pos + 1]
        end

        for t in 0:(n_steps - 1)
            summ = Float32(0)
            k_base = candidate_id * qk_col * n_steps + head_id * dim_per_head + t * qk_col
            for i in 0:(dim_per_head - 1)
                summ += sq[i + 1] * k[k_base + i + 1]
            end
            dp[t + 1] = summ * scale
        end

        max_val = Float32(-1.0f-20)
        for t in 1:n_steps
            max_val = max(max_val, dp[t])
        end

        denom = Float32(0)
        for t in 1:n_steps
            local_i = dp[t] - max_val
            if local_i < -Float32(threshold)
                local_i = -Float32(threshold)
            end
            logits[t] = exp(local_i)
            denom += logits[t]
        end

        for t in 1:n_steps
            logits[t] /= denom
        end

        for t in 0:(dim_per_head - 1)
            summ = Float32(0)
            tid = candidate_id * v_col * n_steps + head_id * dim_per_head + t
            for i in 0:(n_steps - 1)
                summ += logits[i + 1] * v[tid + i * v_col + 1]
            end
            dst[candidate_id * v_col + head_id * dim_per_head + t + 1] = summ
        end
    end
    return dst
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end

    repeat = parse(Int, args[1])
    beamsize = 4
    nhead = 16
    dim_feature = nhead * 256
    n_steps = 9
    scaler = Float32(sqrt(nhead / dim_feature))
    qk_col = dim_feature
    v_col = dim_feature
    threshold = 64

    q_size = beamsize * dim_feature
    k_size = beamsize * dim_feature * n_steps
    v_size = beamsize * dim_feature * n_steps

    rng = MersenneTwister(123)
    hq = rand(rng, Float32, q_size)
    hk = rand(rng, Float32, k_size)
    hv = rand(rng, Float32, v_size)

    dq = CuArray(hq)
    dk = CuArray(hk)
    dv = CuArray(hv)
    dst = CUDA.zeros(Float32, q_size)

    grid = nhead * beamsize
    block = qk_col ÷ nhead
    shmem = sizeof(Float32) * (block + n_steps)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block blocks=grid shmem=shmem mha_kernel!(
            dq, dk, dv, Int32(beamsize), Int32(n_steps), Int32(qk_col),
            Int32(v_col), Int32(nhead), scaler, Int32(threshold), dst)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time: %f (us)\n", elapsed * 1e-3 / repeat)

    h_dst = Array(dst)
    r_dst = mha_reference(hq, hk, hv, beamsize, n_steps, qk_col, v_col,
                          nhead, scaler, threshold)
    ok = all(abs.(h_dst .- r_dst) .<= Float32(1.0f-3))
    println(ok ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
